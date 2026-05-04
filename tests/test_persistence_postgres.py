"""
Postgres-backend integration tests for app.persistence.PostgresStore (A6b).

Uses ``testcontainers-python`` to spin up a throwaway Postgres
container per test session, so we exercise the real driver path,
real JSONB serialization, real ``ON CONFLICT`` semantics — not a
mock.

Skipped automatically when:
  * ``testcontainers`` isn't installed (lightweight dev environments)
  * ``psycopg`` isn't installed (same)
  * Docker isn't reachable (CI without Docker, or Docker daemon down)

Otherwise: this is the production-profile equivalent of the
SQLite-backed test_persistence.py and asserts the same tenant
isolation guarantees against a real Postgres.
"""
from __future__ import annotations

import os
import threading

import pytest


# ──────────────────────────────────────────────────────────────
# Fixtures — gated on testcontainers + Docker availability
# ──────────────────────────────────────────────────────────────
testcontainers = pytest.importorskip(
    "testcontainers.postgres",
    reason="testcontainers-python not installed",
)
psycopg_pool = pytest.importorskip(
    "psycopg_pool",
    reason="psycopg-pool not installed",
)


@pytest.fixture(scope="module")
def postgres_container():
    """One Postgres container per test module — saves ~15s of cold
    boot per test if we re-spun. Each test cleans up its own rows."""
    PostgresContainer = testcontainers.PostgresContainer

    try:
        container = PostgresContainer("postgres:16-alpine")
        container.start()
    except Exception as exc:
        pytest.skip(f"Docker not reachable: {exc}")

    yield container
    container.stop()


@pytest.fixture
def store(postgres_container, monkeypatch):
    """Per-test PostgresStore that wipes the jobs/runs tables on
    setup so each test sees a clean slate without paying for a fresh
    container."""
    from app.persistence.postgres import PostgresStore

    dsn = postgres_container.get_connection_url()
    # testcontainers-python returns a SQLAlchemy-style URL by default
    # ('postgresql+psycopg2://...'); psycopg wants a plain
    # 'postgresql://...'. Strip the driver prefix if present.
    if "+" in dsn.split("://")[0]:
        scheme, rest = dsn.split("://", 1)
        dsn = scheme.split("+")[0] + "://" + rest

    s = PostgresStore(dsn=dsn)
    # Wipe rows; ON CONFLICT and PK rely on a clean slate per test.
    with s._pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM jobs")
            cur.execute("DELETE FROM runs")
        conn.commit()
    yield s
    s.close()


_T = "tenant-default"
_A = "tenant-a"
_B = "tenant-b"


# ──────────────────────────────────────────────────────────────
# JobStore — happy path against real Postgres
# ──────────────────────────────────────────────────────────────
def test_create_get_roundtrip_postgres(store):
    store.create("job-1", _T, {"job_id": "job-1", "status": "queued", "method": "sft"})
    out = store.get("job-1", _T)
    assert out["job_id"] == "job-1"
    assert out["status"] == "queued"
    assert out["method"] == "sft"
    assert out["tenant_id"] == _T


def test_list_filters_by_status_postgres(store):
    store.create("a", _T, {"job_id": "a", "status": "queued"})
    store.create("b", _T, {"job_id": "b", "status": "training"})
    store.create("c", _T, {"job_id": "c", "status": "completed"})
    queued = {j["job_id"] for j in store.list(_T, status="queued")}
    assert queued == {"a"}


def test_update_fields_patches_payload_postgres(store):
    store.create("j", _T, {"job_id": "j", "status": "queued", "progress": 0.0})
    new = store.update_fields("j", _T, status="training", progress=0.42)
    assert new["status"] == "training"
    assert new["progress"] == 0.42
    assert store.get("j", _T)["progress"] == 0.42


def test_loss_history_round_trips_postgres(store):
    history = [{"step": i, "loss": 2.5 - i * 0.1} for i in range(10)]
    store.create("j", _T, {"job_id": "j", "status": "queued", "loss_history": history})
    assert store.get("j", _T)["loss_history"] == history


def test_delete_returns_true_on_hit_postgres(store):
    store.create("j", _T, {"job_id": "j", "status": "queued"})
    assert store.delete("j", _T) is True
    assert store.delete("j", _T) is False
    assert store.get("j", _T) is None


# ──────────────────────────────────────────────────────────────
# JobStore — tenant isolation against real Postgres
# ──────────────────────────────────────────────────────────────
def test_get_does_not_leak_across_tenants_postgres(store):
    store.create("job-x", _A, {"job_id": "job-x", "status": "queued"})
    assert store.get("job-x", _A) is not None
    assert store.get("job-x", _B) is None


def test_list_only_returns_own_tenant_postgres(store):
    store.create("a1", _A, {"job_id": "a1", "status": "queued"})
    store.create("b1", _B, {"job_id": "b1", "status": "queued"})
    assert {j["job_id"] for j in store.list(_A)} == {"a1"}
    assert {j["job_id"] for j in store.list(_B)} == {"b1"}


def test_update_does_not_cross_tenants_postgres(store):
    store.create("job-x", _A, {"job_id": "job-x", "status": "queued"})
    res = store.update_fields("job-x", _B, status="completed")
    assert res is None
    assert store.get("job-x", _A)["status"] == "queued"


def test_delete_does_not_cross_tenants_postgres(store):
    store.create("job-x", _A, {"job_id": "job-x", "status": "queued"})
    assert store.delete("job-x", _B) is False
    assert store.get("job-x", _A) is not None


# ──────────────────────────────────────────────────────────────
# RunStore — tenant isolation
# ──────────────────────────────────────────────────────────────
def test_run_get_does_not_leak_postgres(store):
    store.upsert_run("run-x", _A, {"run_id": "run-x", "domain": "ai_llm"})
    assert store.get_run("run-x", _A) is not None
    assert store.get_run("run-x", _B) is None


def test_run_upsert_collision_does_not_overwrite_other_tenant_postgres(store):
    """The whole point of the WHERE-clause guard on the ON CONFLICT
    UPDATE: tenant B can't silently overwrite tenant A's row by
    upserting the same run_id."""
    store.upsert_run("shared", _A, {"run_id": "shared", "domain": "ai_llm", "marker": "A"})
    store.upsert_run("shared", _B, {"run_id": "shared", "domain": "x", "marker": "B"})
    assert store.get_run("shared", _A)["marker"] == "A"
    assert store.get_run("shared", _B) is None


# ──────────────────────────────────────────────────────────────
# Concurrency — connection pool under load
# ──────────────────────────────────────────────────────────────
def test_concurrent_writes_serialize_cleanly_postgres(store):
    """50 threads × 10 writes; FOR UPDATE row lock prevents lost
    updates so the counter actually reaches 500."""
    store.create("j", _T, {"job_id": "j", "status": "queued", "counter": 0})

    def worker():
        for _ in range(10):
            current = store.get("j", _T)["counter"]
            store.update_fields("j", _T, counter=current + 1)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads: t.start()
    for t in threads: t.join()

    final = store.get("j", _T)["counter"]
    # Postgres FOR UPDATE in update_fields means no lost updates ->
    # exactly 500. Any value < 500 indicates the row lock isn't
    # actually serializing.
    assert final == 500
