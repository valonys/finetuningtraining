"""
Unit tests for app.uploads — filename sanitization (path-traversal guard),
unique-target suffixing, and the list/delete/clear flow on a tmp dir.
"""
from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────
# Filename sanitization — the security-critical path
# ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw, expected_contains", [
    # Basic cases
    ("report.pdf",                     "report.pdf"),
    ("My File With Spaces.pdf",        "My_File_With_Spaces.pdf"),
    ("résumé.docx",                    "r_sum_.docx"),
    # Path traversal attempts — the sanitizer must strip every dir component
    ("../../../etc/passwd",            "passwd"),
    ("..\\..\\windows\\system32.dll",  "system32.dll"),
    ("/absolute/path/file.txt",        "file.txt"),
    ("foo/bar/baz.pdf",                "baz.pdf"),
    # Hidden / dot files shouldn't strip to empty
    (".env",                           "env"),
    ("...",                            "upload"),
    # Empty / None-ish
    ("",                               "upload"),
    ("   ",                            "upload"),
    # `/` is a path separator; everything before the last `/` is stripped
    ("file;rm -rf /.pdf",              "pdf"),
    ("$(whoami).txt",                  "whoami_.txt"),     # `$(` → `__` → leading _s stripped
])
def test_safe_filename_strips_paths_and_unsafe_chars(raw, expected_contains):
    from app.uploads import safe_filename
    out = safe_filename(raw)
    # Must not contain path separators
    assert "/" not in out
    assert "\\" not in out
    # Must not start with `..` (traversal)
    assert not out.startswith("..")
    # The sanitizer must produce the expected shape
    assert out == expected_contains, f"safe_filename({raw!r}) → {out!r} != {expected_contains!r}"


# ──────────────────────────────────────────────────────────────
# Full upload lifecycle — save / list / delete / clear
# ──────────────────────────────────────────────────────────────
def _isolated_uploads(monkeypatch, tmp_path):
    """Point UPLOADS_DIR at a tmp dir for the duration of one test."""
    import app.uploads as u
    monkeypatch.setattr(u, "UPLOADS_DIR", tmp_path.resolve())
    return u


def test_unique_target_suffixes_on_collision(monkeypatch, tmp_path):
    u = _isolated_uploads(monkeypatch, tmp_path)
    # First call → no collision → base name preserved
    p1 = u.unique_target("doc.pdf")
    assert p1.name == "doc.pdf"
    # Create it, then ask again
    p1.write_text("x")
    p2 = u.unique_target("doc.pdf")
    assert p2.name == "doc_1.pdf"
    p2.write_text("x")
    p3 = u.unique_target("doc.pdf")
    assert p3.name == "doc_2.pdf"


def test_list_uploads_and_clear(monkeypatch, tmp_path):
    u = _isolated_uploads(monkeypatch, tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world!")
    (tmp_path / ".hidden").write_text("skip-me")

    files = u.list_uploads()
    names = [f.name for f in files]
    assert "a.txt" in names
    assert "b.txt" in names
    assert ".hidden" not in names          # dotfiles skipped
    assert all(f.size > 0 for f in files)
    assert u.total_upload_bytes() == len("hello") + len("world!")

    deleted = u.clear_uploads()
    assert deleted == 2                     # dotfile not counted
    assert u.list_uploads() == []


def test_delete_upload_single(monkeypatch, tmp_path):
    u = _isolated_uploads(monkeypatch, tmp_path)
    (tmp_path / "report.pdf").write_text("pdf-contents")

    assert u.delete_upload("report.pdf") is True
    assert not (tmp_path / "report.pdf").exists()
    # Deleting something that doesn't exist returns False
    assert u.delete_upload("ghost.pdf") is False


def test_delete_refuses_path_traversal(monkeypatch, tmp_path):
    u = _isolated_uploads(monkeypatch, tmp_path)
    # Even if someone passes `../../something`, safe_filename neutralises it,
    # and the delete either fails-safely (file doesn't exist in the uploads
    # dir) or raises UploadError if the resolved path escapes.
    ok = u.delete_upload("../../../etc/passwd")
    assert ok is False                      # sanitized to "passwd", which doesn't exist


@pytest.mark.asyncio
async def test_save_upload_streams_and_records_size(monkeypatch, tmp_path):
    u = _isolated_uploads(monkeypatch, tmp_path)

    class _FakeUpload:
        """Minimal stand-in for FastAPI's UploadFile."""
        def __init__(self, name, payload):
            self.filename = name
            self._payload = payload
            self._pos = 0

        async def read(self, n: int) -> bytes:
            chunk = self._payload[self._pos : self._pos + n]
            self._pos += len(chunk)
            return chunk

    payload = b"hello world" * 100
    uf = _FakeUpload("my report.pdf", payload)
    rec = await u.save_upload(uf)

    assert rec.name == "my_report.pdf"      # sanitized
    assert rec.size == len(payload)
    assert (tmp_path / "my_report.pdf").read_bytes() == payload


@pytest.mark.asyncio
async def test_save_upload_rejects_oversize(monkeypatch, tmp_path):
    u = _isolated_uploads(monkeypatch, tmp_path)
    monkeypatch.setattr(u, "MAX_FILE_BYTES", 50)

    class _FakeUpload:
        def __init__(self, name, payload):
            self.filename = name
            self._payload = payload
            self._pos = 0

        async def read(self, n: int) -> bytes:
            chunk = self._payload[self._pos : self._pos + n]
            self._pos += len(chunk)
            return chunk

    oversize = _FakeUpload("big.bin", b"x" * 200)
    with pytest.raises(u.UploadError) as exc:
        await u.save_upload(oversize)
    assert "exceeds" in str(exc.value).lower()
    # Partial file must have been cleaned up
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_save_upload_rejects_empty(monkeypatch, tmp_path):
    u = _isolated_uploads(monkeypatch, tmp_path)

    class _FakeUpload:
        filename = "empty.txt"
        async def read(self, _n):
            return b""

    with pytest.raises(u.UploadError) as exc:
        await u.save_upload(_FakeUpload())
    assert "empty" in str(exc.value).lower()
    assert list(tmp_path.iterdir()) == []
