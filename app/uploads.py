"""
app/uploads.py
──────────────
File upload handling for the Data Forge tab.

Files arrive via `POST /v1/forge/upload` (multipart/form-data) and are
streamed to `./data/uploads/<sanitized_filename>`. Subsequent build-dataset
calls reference those server-side paths.

Security:
  - Strip any path components from the incoming filename (no traversal)
  - Allow only ASCII [A-Za-z0-9._-] in the final name; replace everything
    else with `_` (ASCII-only for max cross-platform filesystem safety)
  - Never overwrite an existing file — auto-suffix `_1`, `_2`, ...
  - Size cap per file (default 512 MB) to guard against runaway uploads
  - DELETE endpoints refuse to touch anything outside the uploads dir
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

UPLOADS_DIR = Path(os.environ.get("VALONY_UPLOADS_DIR", "./data/uploads")).resolve()
MAX_FILE_BYTES = int(os.environ.get("VALONY_MAX_UPLOAD_BYTES", 512 * 1024 * 1024))  # 512 MB
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._\-]")   # ASCII-only allowlist


@dataclass
class UploadedFile:
    name: str          # sanitized final name on disk
    path: str          # absolute path
    size: int          # bytes


class UploadError(ValueError):
    """Raised for invalid names, size violations, path traversal attempts."""


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def safe_filename(raw: str) -> str:
    """
    Derive a filesystem-safe filename from a user-supplied string.

    Strips any directory components (`foo/bar.pdf` → `bar.pdf`),
    replaces unsafe chars with `_`, collapses `..` attempts,
    and falls back to `upload` if the name becomes empty.
    """
    if not raw:
        return "upload"
    # Take only the basename — blocks `../../etc/passwd` style attacks
    base = Path(raw.replace("\\", "/")).name
    # Replace anything outside [A-Za-z0-9._-]
    cleaned = _UNSAFE_RE.sub("_", base).strip("._")
    return cleaned or "upload"


def ensure_uploads_dir() -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR


def unique_target(filename: str) -> Path:
    """
    Return a non-existing path in UPLOADS_DIR for `filename`.
    Appends `_1`, `_2`, ... to the stem on collision.
    """
    base = ensure_uploads_dir() / filename
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    i = 1
    while True:
        candidate = base.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _inside_uploads(path: Path) -> bool:
    """Resolve `path` and confirm it sits inside UPLOADS_DIR (no traversal)."""
    try:
        resolved = path.resolve(strict=False)
        return resolved.is_relative_to(UPLOADS_DIR)
    except (ValueError, OSError):
        return False


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
async def save_upload(upload_file) -> UploadedFile:
    """
    Stream one FastAPI `UploadFile` to disk under UPLOADS_DIR.

    Raises UploadError if the file is empty or exceeds MAX_FILE_BYTES.
    """
    name = safe_filename(upload_file.filename)
    target = unique_target(name)

    total = 0
    with open(target, "wb") as out:
        while True:
            chunk = await upload_file.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                out.close()
                target.unlink(missing_ok=True)
                raise UploadError(
                    f"File '{name}' exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB cap"
                )
            out.write(chunk)

    if total == 0:
        target.unlink(missing_ok=True)
        raise UploadError(f"File '{name}' is empty")

    return UploadedFile(name=target.name, path=str(target), size=total)


def list_uploads() -> List[UploadedFile]:
    """List every file currently in the uploads directory (non-recursive)."""
    ensure_uploads_dir()
    out: list[UploadedFile] = []
    for entry in sorted(UPLOADS_DIR.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_file() and not entry.name.startswith("."):
            out.append(UploadedFile(
                name=entry.name,
                path=str(entry),
                size=entry.stat().st_size,
            ))
    return out


def delete_upload(filename: str) -> bool:
    """Delete one upload by (sanitized) filename. Returns True if deleted."""
    safe = safe_filename(filename)
    target = UPLOADS_DIR / safe
    if not _inside_uploads(target):
        raise UploadError(f"Refusing to delete outside uploads dir: {filename}")
    if not target.exists() or not target.is_file():
        return False
    target.unlink()
    return True


def clear_uploads() -> int:
    """Delete every file in the uploads dir (skips dotfiles to stay
    consistent with `list_uploads`). Returns the count deleted."""
    ensure_uploads_dir()
    count = 0
    for entry in UPLOADS_DIR.iterdir():
        if (
            entry.is_file()
            and not entry.name.startswith(".")
            and _inside_uploads(entry)
        ):
            entry.unlink()
            count += 1
    return count


def total_upload_bytes() -> int:
    return sum(f.size for f in list_uploads())
