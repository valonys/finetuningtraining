"""
app/security/paths.py
─────────────────────
Path-traversal hardening for user-supplied filesystem paths reaching
the API surface (S06 of the Lane A sprint plan).

Endpoints that took ``req.paths`` / ``req.path`` / ``req.output_dir``
previously handed them straight to ingest / harvester / export code.
A request like ``{"paths": ["/etc/passwd"]}`` would happily be opened.
This module is the single chokepoint they now route through.

Contract:
  * ``validated_path(raw, allowlist_roots=...)`` returns a fully
    resolved ``Path`` only if it lives under at least one allowlist
    root. Symlink escapes are caught because we resolve first and then
    compare against resolved roots.
  * Default allowlist (used when ``allowlist_roots`` is None) covers the
    three legitimate write-zones: uploads, processed datasets, outputs.
  * ``must_exist=True`` adds an explicit existence check on top.

Override roots via env:
  ``VALONY_UPLOADS_DIR``    (default ``./data/uploads``)
  ``VALONY_PROCESSED_DIR``  (default ``./data/processed``)
  ``VALONY_OUTPUTS_DIR``    (default ``./outputs``)
"""
from __future__ import annotations

import os
from pathlib import Path


class PathValidationError(ValueError):
    """Raised when a user-supplied path resolves outside the allowlist."""


def default_allowlist() -> list[Path]:
    """Resolve the three default roots fresh on each call so env-var
    overrides set in the same process are honored."""
    return [
        Path(os.environ.get("VALONY_UPLOADS_DIR", "./data/uploads")).resolve(),
        Path(os.environ.get("VALONY_PROCESSED_DIR", "./data/processed")).resolve(),
        Path(os.environ.get("VALONY_OUTPUTS_DIR", "./outputs")).resolve(),
    ]


def validated_path(
    raw: str | os.PathLike[str],
    *,
    allowlist_roots: list[Path] | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve ``raw`` and confirm it lives under one of the allowlist roots.

    Args:
        raw: The path string from the request payload.
        allowlist_roots: Override the defaults. Each entry is itself
            ``.resolve()``-d so callers can pass relative roots.
        must_exist: When True, raise ``FileNotFoundError`` if the
            resolved path does not exist on disk.

    Returns:
        The fully resolved ``Path`` if validation passes.

    Raises:
        PathValidationError: empty string, escapes the allowlist, or
            otherwise unresolvable.
        FileNotFoundError: ``must_exist=True`` and the path is missing.
    """
    if raw is None or raw == "":
        raise PathValidationError("path is empty")

    roots = [Path(r).resolve() for r in (allowlist_roots or default_allowlist())]
    if not roots:
        raise PathValidationError("no allowlist roots configured")

    try:
        resolved = Path(raw).expanduser().resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise PathValidationError(f"unresolvable path {raw!r}: {exc}") from exc

    for root in roots:
        try:
            if resolved == root or resolved.is_relative_to(root):
                if must_exist and not resolved.exists():
                    raise FileNotFoundError(f"path does not exist: {resolved}")
                return resolved
        except ValueError:
            # Different drives on Windows; .is_relative_to may raise.
            continue

    raise PathValidationError(
        f"path {raw!r} resolved to {resolved} which is outside the allowlist "
        f"(roots: {[str(r) for r in roots]})"
    )
