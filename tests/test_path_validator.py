"""
Unit tests for app.security.paths.validated_path.

The validator is the S06 chokepoint for every endpoint that takes a
user-supplied filesystem path. These tests exercise its three
guarantees: only allowlisted roots accept, escapes reject (including
symlink redirection), and ``must_exist`` reports missing files.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.security import (
    PathValidationError,
    default_allowlist,
    validated_path,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def env_roots(tmp_path: Path, monkeypatch):
    """Point all three default roots at tmp_path subdirs."""
    uploads = tmp_path / "uploads"
    processed = tmp_path / "processed"
    outputs = tmp_path / "outputs"
    for p in (uploads, processed, outputs):
        p.mkdir()
    monkeypatch.setenv("VALONY_UPLOADS_DIR", str(uploads))
    monkeypatch.setenv("VALONY_PROCESSED_DIR", str(processed))
    monkeypatch.setenv("VALONY_OUTPUTS_DIR", str(outputs))
    return {"uploads": uploads, "processed": processed, "outputs": outputs}


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────
def test_path_under_each_root_accepted(env_roots, tmp_path):
    for label, root in env_roots.items():
        f = root / f"file_{label}.txt"
        f.write_text("hi")
        result = validated_path(str(f))
        assert result == f.resolve()


def test_root_itself_accepted(env_roots):
    """The root directory entry is itself a legitimate target (e.g.
    write a new file directly into outputs/)."""
    for root in env_roots.values():
        assert validated_path(str(root)) == root.resolve()


def test_traversal_with_dotdot_rejected(env_roots, tmp_path):
    sneaky = env_roots["uploads"] / ".." / ".." / "etc_passwd"
    with pytest.raises(PathValidationError, match="outside the allowlist"):
        validated_path(str(sneaky))


def test_absolute_path_outside_allowlist_rejected(tmp_path, env_roots):
    outside = tmp_path / "definitely_not_in_allowlist" / "x.txt"
    outside.parent.mkdir()
    outside.write_text("secrets")
    with pytest.raises(PathValidationError, match="outside the allowlist"):
        validated_path(str(outside))


def test_symlink_escape_rejected(env_roots, tmp_path):
    """A symlink inside the allowlist that points OUT must be rejected
    after resolution."""
    secret = tmp_path / "secret.txt"
    secret.write_text("classified")

    link = env_roots["uploads"] / "innocent.txt"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")

    with pytest.raises(PathValidationError, match="outside the allowlist"):
        validated_path(str(link))


def test_empty_path_rejected(env_roots):
    with pytest.raises(PathValidationError, match="empty"):
        validated_path("")


def test_must_exist_raises_for_missing_file(env_roots):
    target = env_roots["uploads"] / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        validated_path(str(target), must_exist=True)


def test_must_exist_passes_for_existing_file(env_roots):
    target = env_roots["uploads"] / "real.pdf"
    target.write_text("x")
    assert validated_path(str(target), must_exist=True) == target.resolve()


def test_explicit_allowlist_overrides_default(tmp_path):
    custom_root = tmp_path / "custom"
    custom_root.mkdir()
    target = custom_root / "x.txt"
    target.write_text("y")
    # Default allowlist would reject this — explicit overrides it
    result = validated_path(str(target), allowlist_roots=[custom_root])
    assert result == target.resolve()


def test_default_allowlist_reads_env_each_call(tmp_path, monkeypatch):
    a = tmp_path / "first"
    b = tmp_path / "second"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("VALONY_UPLOADS_DIR", str(a))
    roots1 = default_allowlist()
    assert any(r == a.resolve() for r in roots1)

    monkeypatch.setenv("VALONY_UPLOADS_DIR", str(b))
    roots2 = default_allowlist()
    assert any(r == b.resolve() for r in roots2)


def test_relative_path_resolves_against_cwd(env_roots, tmp_path, monkeypatch):
    """Relative inputs resolve via Path.resolve() — they must end up
    inside an allowlist root or be rejected."""
    monkeypatch.chdir(env_roots["uploads"])
    (env_roots["uploads"] / "doc.txt").write_text("z")
    assert validated_path("doc.txt") == (env_roots["uploads"] / "doc.txt").resolve()
