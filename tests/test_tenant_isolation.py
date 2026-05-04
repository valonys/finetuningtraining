"""
Cross-cutting tenant isolation acceptance test for A6b.

Drives the FastAPI app via TestClient with two valid tenant tokens
(tenant-A and tenant-B) and asserts that:

  1. Job created by A is invisible to B (`GET /v1/jobs/{id}` 404).
  2. Job list returned to A doesn't contain B's jobs (and vice-versa).
  3. Audit log lines for A's actions are not visible in B's filtered
     query.
  4. Stored job rows on disk carry the correct tenant_id stamp.
  5. Auth-disabled mode (the default dev setting) routes everything
     to the synthetic 'public' tenant — both clients see each other.

This is the test that fires for the SPRINTS.md A6b acceptance gate
("tenant isolation verified in automated tests"). It exercises the
*full stack* — auth middleware → tenant extraction → store scoping
→ audit log filtering — rather than just the storage layer in
isolation (which test_persistence.py covers).
"""
from __future__ import annotations

import time
from pathlib import Path

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient


_SECRET = "tenant-isolation-test-secret-32-byte"


def _mint(tenant_id: str, *, user_id: str = "alice", secret: str = _SECRET) -> str:
    return pyjwt.encode(
        {
            "tenant_id": tenant_id,
            "sub": user_id,
            "roles": ["user"],
            "exp": int(time.time()) + 60,
        },
        secret,
        algorithm="HS256",
    )


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch):
    """Boot a fresh FastAPI app per test with auth ON and persistence
    + audit pointed at tmp dirs. The lazy ``default_store()`` and
    ``default_logger()`` singletons are reset so each test sees a
    clean slate."""
    monkeypatch.setenv("VALONY_AUTH_REQUIRED", "1")
    monkeypatch.setenv("VALONY_JWT_SECRET", _SECRET)
    monkeypatch.setenv("VALONY_PERSISTENCE_PATH", str(tmp_path / "store.db"))
    monkeypatch.setenv("VALONY_AUDIT_DIR", str(tmp_path / "audit"))

    # Reset module-level singletons so they pick up the env overrides.
    import app.persistence.store as store_mod
    import app.audit.logging as audit_mod
    monkeypatch.setattr(store_mod, "_default_store", None)
    monkeypatch.setattr(audit_mod, "_default", None)

    from app.main import app
    return app


def _client(app, tenant_id: str | None) -> TestClient:
    headers = {}
    if tenant_id is not None:
        headers["Authorization"] = f"Bearer {_mint(tenant_id)}"
    c = TestClient(app)
    c.headers.update(headers)
    return c


def _seed_job(app, tenant_id: str, *, status: str = "queued") -> str:
    """Insert a job directly via the store helpers — avoids needing a
    real trainer in tests. We're testing tenant isolation, not
    training mechanics."""
    from app.main import _store_job
    from app.models import JobStatus
    import uuid

    jid = str(uuid.uuid4())
    job = JobStatus(
        job_id=jid,
        status=status,
        progress=0.0,
        method="sft",
        dataset_source="local",
    )
    _store_job(job, tenant_id)
    return jid


# ──────────────────────────────────────────────────────────────
# Acceptance gate
# ──────────────────────────────────────────────────────────────
def test_get_job_returns_404_for_other_tenant(isolated_app):
    """Tenant A creates job-x; tenant B asks for job-x → 404
    (not the payload, not a 403 leaking that the id exists)."""
    jid_a = _seed_job(isolated_app, "tenant-a")

    a = _client(isolated_app, "tenant-a")
    b = _client(isolated_app, "tenant-b")

    assert a.get(f"/v1/jobs/{jid_a}").status_code == 200
    res_b = b.get(f"/v1/jobs/{jid_a}")
    assert res_b.status_code == 404
    # Body shouldn't mention the id existing elsewhere
    assert "tenant" not in res_b.json().get("detail", "").lower()


def test_list_jobs_only_returns_own_tenants(isolated_app):
    a1 = _seed_job(isolated_app, "tenant-a")
    a2 = _seed_job(isolated_app, "tenant-a")
    b1 = _seed_job(isolated_app, "tenant-b")

    a = _client(isolated_app, "tenant-a")
    b = _client(isolated_app, "tenant-b")

    a_jobs = {j["job_id"] for j in a.get("/v1/jobs").json()}
    b_jobs = {j["job_id"] for j in b.get("/v1/jobs").json()}
    assert a_jobs == {a1, a2}
    assert b_jobs == {b1}
    assert not (a_jobs & b_jobs)


def test_stored_payload_carries_tenant_stamp(isolated_app, tmp_path):
    """Verify at the storage level (not just the API) that each
    persisted job's payload includes tenant_id."""
    _seed_job(isolated_app, "tenant-a")
    _seed_job(isolated_app, "tenant-b")

    from app.persistence import default_store
    store = default_store()
    a_jobs = store.list("tenant-a")
    b_jobs = store.list("tenant-b")
    assert all(j["tenant_id"] == "tenant-a" for j in a_jobs)
    assert all(j["tenant_id"] == "tenant-b" for j in b_jobs)


def test_audit_log_filters_by_tenant(isolated_app, tmp_path):
    """Drive a few audit-emitting actions for both tenants, then
    verify each tenant's filtered audit query only sees their own."""
    from app.audit import AuditAction, default_logger

    audit = default_logger()
    audit.log(tenant_id="tenant-a", action=AuditAction.JOB_CREATED, target_id="ja1")
    audit.log(tenant_id="tenant-a", action=AuditAction.CHAT, target_id="ra1")
    audit.log(tenant_id="tenant-b", action=AuditAction.JOB_CREATED, target_id="jb1")
    audit.log(tenant_id="tenant-b", action=AuditAction.UPLOAD, target_id="ub1")

    a_events = audit.query(tenant_id="tenant-a")
    b_events = audit.query(tenant_id="tenant-b")
    assert {e.target_id for e in a_events} == {"ja1", "ra1"}
    assert {e.target_id for e in b_events} == {"jb1", "ub1"}


def test_unauthenticated_request_rejected_in_strict_mode(isolated_app):
    """Sanity: without a token, even a list call gets 401 in strict
    mode. (If this regresses, the whole tenant story collapses.)"""
    nobody = TestClient(isolated_app)
    res = nobody.get("/v1/jobs")
    assert res.status_code == 401
    assert "WWW-Authenticate" in res.headers


def test_invalid_token_rejected(isolated_app):
    forged = pyjwt.encode(
        {"tenant_id": "tenant-a", "exp": int(time.time()) + 60},
        "wrong-secret-not-the-server-secret",
        algorithm="HS256",
    )
    forged_client = TestClient(isolated_app)
    forged_client.headers["Authorization"] = f"Bearer {forged}"
    res = forged_client.get("/v1/jobs")
    assert res.status_code == 401


def test_healthz_reachable_without_token(isolated_app):
    """Liveness must always work — infrastructure can't probe it
    otherwise."""
    nobody = TestClient(isolated_app)
    res = nobody.get("/healthz")
    assert res.status_code == 200


# ──────────────────────────────────────────────────────────────
# Auth-disabled mode (dev) — public tenant routing
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def app_no_auth(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VALONY_AUTH_REQUIRED", raising=False)
    monkeypatch.setenv("VALONY_PERSISTENCE_PATH", str(tmp_path / "store.db"))
    monkeypatch.setenv("VALONY_AUDIT_DIR", str(tmp_path / "audit"))

    import app.persistence.store as store_mod
    import app.audit.logging as audit_mod
    monkeypatch.setattr(store_mod, "_default_store", None)
    monkeypatch.setattr(audit_mod, "_default", None)

    from app.main import app
    return app


def test_dev_mode_routes_everything_to_public_tenant(app_no_auth):
    """In auth-disabled mode (the default dev setting) every request
    gets the synthetic 'public' tenant. Two clients hitting the same
    instance see each other's data — that's the whole point of
    dev mode."""
    jid = _seed_job(app_no_auth, "public")

    one = TestClient(app_no_auth)
    two = TestClient(app_no_auth)

    res1 = one.get(f"/v1/jobs/{jid}")
    res2 = two.get(f"/v1/jobs/{jid}")
    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res1.json()["job_id"] == res2.json()["job_id"] == jid
