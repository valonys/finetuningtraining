"""
app/security/cors.py
────────────────────
CORS allowlist resolution. Replaces the wildcard ``allow_origins=["*"]``
that lived in ``app/main.py`` until S06.

Resolution order:
  1. ``VALONY_CORS_ORIGINS`` env var — comma-separated origin list
     (e.g. ``https://studio.example.com,https://staging.example.com``).
  2. Empty / unset — fall back to the dev defaults
     (``http://localhost:5173`` + ``http://127.0.0.1:5173``) so the
     Vite dev server keeps working out of the box.
"""
from __future__ import annotations

import os


_DEV_DEFAULT = ["http://localhost:5173", "http://127.0.0.1:5173"]


def resolve_cors_origins() -> list[str]:
    raw = os.environ.get("VALONY_CORS_ORIGINS", "").strip()
    if not raw:
        return list(_DEV_DEFAULT)
    parts = [o.strip() for o in raw.split(",")]
    cleaned = [o for o in parts if o]
    return cleaned or list(_DEV_DEFAULT)
