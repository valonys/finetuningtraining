"""
app/security — S06 hardening primitives.

Public surface:
    validated_path        Resolve a user-supplied path and confirm it lives
                          under one of the allowlist roots (no traversal).
    PathValidationError   Raised when a path escapes every allowlist root.
    default_allowlist     Lazy default roots (uploads / processed / outputs).
    resolve_cors_origins  Env-driven CORS allowlist resolver
                          (default: localhost:5173 + 127.0.0.1:5173).
"""
from .paths import (
    PathValidationError,
    default_allowlist,
    validated_path,
)
from .cors import resolve_cors_origins

__all__ = [
    "PathValidationError",
    "default_allowlist",
    "validated_path",
    "resolve_cors_origins",
]
