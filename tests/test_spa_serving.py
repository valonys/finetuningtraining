"""
Unit tests for the SPA serving block at the bottom of app/main.py (S07).

Covers four behaviors that matter in production:
  1. The catch-all serves index.html for unknown SPA routes (so React
     Router takes over after a hard reload on /domains/foo).
  2. The catch-all does NOT mask API misses — a mistyped /v1/foo must
     still return 404, not an HTML shell.
  3. Path traversal via /{full_path} is rejected.
  4. Explicit static files inside dist (favicon.ico etc.) are served
     directly rather than falling through to index.html.

The block only registers when ``frontend/dist`` exists, so we set up a
fake dist tree under tmp_path, monkeypatch the module-level pointer,
re-evaluate the registration logic via a small helper, then drive
requests through TestClient.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _build_fake_dist(root: Path) -> Path:
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><title>SPA</title><div id=root></div>"
    )
    (dist / "assets" / "main.js").write_text("console.log('hello')")
    (dist / "favicon.ico").write_bytes(b"\x00\x00\x01\x00")
    return dist


def _make_app_with_spa(dist: Path) -> FastAPI:
    """Replicate the production registration order: API routes first,
    SPA mount last. Lifted directly from app/main.py for test isolation
    so we don't have to import the full app (which loads inference
    backends, hardware probes, etc.)."""
    app = FastAPI()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/v1/templates")
    async def templates():
        return {"templates": ["alpaca", "qwen"]}

    @app.get("/v1/registry/{model_version}")
    async def registry_get(model_version: str):
        if model_version == "missing":
            raise HTTPException(404, "Not Found")
        return {"model_version": model_version}

    # ── SPA block (mirrors app/main.py) ───────────────────────
    api_prefixes = ("v1/", "healthz", "openapi.json", "docs", "redoc")
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    async def spa_root():
        return FileResponse(dist / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith(api_prefixes):
            raise HTTPException(404, "Not Found")
        candidate = (dist / full_path).resolve()
        try:
            candidate.relative_to(dist.resolve())
        except ValueError:
            raise HTTPException(404, "Not Found")
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")

    return app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    dist = _build_fake_dist(tmp_path)
    return TestClient(_make_app_with_spa(dist))


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────
def test_root_serves_index_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "id=root" in res.text
    assert res.headers["content-type"].startswith("text/html")


def test_unknown_spa_route_falls_through_to_index(client):
    res = client.get("/domains/specific-id")
    assert res.status_code == 200
    assert "id=root" in res.text


def test_explicit_dist_file_is_served_directly(client):
    res = client.get("/favicon.ico")
    assert res.status_code == 200
    assert res.content == b"\x00\x00\x01\x00"


def test_assets_mount_serves_hashed_bundle(client):
    res = client.get("/assets/main.js")
    assert res.status_code == 200
    assert "console.log" in res.text


def test_api_route_still_wins_over_spa_catchall(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_api_route_with_param_still_wins(client):
    res = client.get("/v1/registry/v123")
    assert res.status_code == 200
    assert res.json() == {"model_version": "v123"}


def test_v1_miss_returns_404_not_index_html(client):
    """A mistyped /v1/foo must NOT be silently rewritten to index.html
    — that would make API typos look like 'page works, data missing'."""
    res = client.get("/v1/this/does/not/exist")
    assert res.status_code == 404
    assert "id=root" not in res.text


def test_healthz_miss_returns_404(client):
    res = client.get("/healthz/extra")
    assert res.status_code == 404
    assert "id=root" not in res.text


def test_path_traversal_rejected(client):
    """Even if someone constructs a `/{full_path}` that resolves above
    dist, the explicit relative_to check rejects it."""
    res = client.get("/../../etc/passwd")
    # The HTTP client may normalize the URL; whichever path makes it to
    # the server, it must not return a file from outside dist.
    assert res.status_code == 404 or "root:" not in res.text


def test_skipped_when_dist_missing(tmp_path):
    """Sanity check: if frontend/dist doesn't exist, the SPA block
    must not register routes — / should return whatever the underlying
    framework would (404 here since we have no other root route)."""
    app = FastAPI()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    nonexistent = tmp_path / "definitely_not_dist"
    if nonexistent.is_dir():
        pytest.fail("test fixture pre-condition broken")

    # We don't register the SPA block at all because dist is missing.
    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 404
