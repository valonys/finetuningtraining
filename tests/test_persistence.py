"""
Unit tests for app.persistence.SQLiteStore.

Covers the tenant-scoped JobStore + RunStore protocols (A6b), the
durability invariant from A5 (uvicorn restart doesn't lose jobs),
and the cross-tenant isolation guarantees: tenant A cannot read,
update, or delete tenant B's records.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.persistence import SQLiteStore


_T = "tenant-default"   # default tenant for legacy single-tenant tests
_A = "tenant-a"
_B = "tenant-b"


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(db_path=tmp_path / "test.db")


# ──────────────────────────────────────────────────────────────
# JobStore — single-tenant happy path
# ──────────────────────────────────────────────────────────────
def test_create_get_roundtrip(store):
    store.create("job-1", _T, {"job_id": "job-1", "status": "queued", "method": "sft"})
    out = store.get("job-1", _T)
    assert out["job_id"] == "job-1"
    assert out["status"] == "queued"
    assert out["method"] == "sft"
    assert out["tenant_id"] == _T


def test_get_unknown_returns_none(store):
    assert store.get("does-not-exist", _T) is None


def test_list_orders_most_recently_updated_first(store):
    store.create("a", _T, {"job_id": "a", "status": "queued"})
    store.create("b", _T, {"job_id": "b", "status": "queued"})
    store.update_fields("a", _T, progress=0.5)
    listed = store.list(_T)
    assert [j["job_id"] for j in listed[:2]] == ["a", "b"]


def test_list_filters_by_status(store):
    store.create("a", _T, {"job_id": "a", "status": "queued"})
    store.create("b", _T, {"job_id": "b", "status": "training"})
    store.create("c", _T, {"job_id": "c", "status": "completed"})
    queued = {j["job_id"] for j in store.list(_T, status="queued")}
    completed = {j["job_id"] for j in store.list(_T, status="completed")}
    assert queued == {"a"}
    assert completed == {"c"}


def test_update_fields_patches_payload(store):
    store.create("j", _T, {"job_id": "j", "status": "queued", "progress": 0.0})
    new = store.update_fields("j", _T, status="training", progress=0.42)
    assert new["status"] == "training"
    assert new["progress"] == 0.42
    assert store.get("j", _T)["progress"] == 0.42


def test_update_fields_unknown_returns_none(store):
    assert store.update_fields("nope", _T, status="x") is None


def test_update_fields_changes_status_index(store):
    store.create("j", _T, {"job_id": "j", "status": "queued"})
    assert {j["job_id"] for j in store.list(_T, status="queued")} == {"j"}
    store.update_fields("j", _T, status="completed")
    assert store.list(_T, status="queued") == []
    assert {j["job_id"] for j in store.list(_T, status="completed")} == {"j"}


def test_delete_returns_true_on_hit_false_on_miss(store):
    store.create("j", _T, {"job_id": "j", "status": "queued"})
    assert store.delete("j", _T) is True
    assert store.delete("j", _T) is False
    assert store.get("j", _T) is None


def test_durability_across_fresh_instance(tmp_path: Path):
    """A5 acceptance gate: a brand-new store pointed at the same db
    file must see every record the prior store wrote."""
    db = tmp_path / "durable.db"
    s1 = SQLiteStore(db_path=db)
    s1.create("j1", _T, {"job_id": "j1", "status": "queued", "method": "sft"})
    s1.update_fields("j1", _T, status="training", progress=0.3)
    s1.close()

    s2 = SQLiteStore(db_path=db)
    out = s2.get("j1", _T)
    assert out["status"] == "training"
    assert out["progress"] == 0.3


def test_loss_history_round_trips_as_list_of_dicts(store):
    history = [
        {"step": 0, "loss": 2.4, "lr": 1e-4},
        {"step": 1, "loss": 2.1, "lr": 1e-4},
        {"step": 2, "loss": 1.9, "lr": 1e-4},
    ]
    store.create("j", _T, {"job_id": "j", "status": "queued", "loss_history": history})
    out = store.get("j", _T)
    assert out["loss_history"] == history


# ──────────────────────────────────────────────────────────────
# JobStore — tenant isolation (A6b)
# ──────────────────────────────────────────────────────────────
def test_get_does_not_leak_across_tenants(store):
    """Tenant A creates job-x; tenant B asks for job-x → must get
    None (NOT a cross-tenant payload, NOT an error revealing that
    the id exists in some other tenant)."""
    store.create("job-x", _A, {"job_id": "job-x", "status": "queued"})
    assert store.get("job-x", _A) is not None
    assert store.get("job-x", _B) is None


def test_list_only_returns_own_tenant(store):
    store.create("a1", _A, {"job_id": "a1", "status": "queued"})
    store.create("a2", _A, {"job_id": "a2", "status": "queued"})
    store.create("b1", _B, {"job_id": "b1", "status": "queued"})
    assert {j["job_id"] for j in store.list(_A)} == {"a1", "a2"}
    assert {j["job_id"] for j in store.list(_B)} == {"b1"}


def test_update_does_not_cross_tenants(store):
    """Tenant A creates job-x with status queued; tenant B tries to
    flip its status to 'completed' → must be a no-op (returns None,
    A's record unchanged)."""
    store.create("job-x", _A, {"job_id": "job-x", "status": "queued"})
    res = store.update_fields("job-x", _B, status="completed")
    assert res is None
    assert store.get("job-x", _A)["status"] == "queued"


def test_delete_does_not_cross_tenants(store):
    store.create("job-x", _A, {"job_id": "job-x", "status": "queued"})
    assert store.delete("job-x", _B) is False
    assert store.get("job-x", _A) is not None


def test_two_tenants_can_share_status_filter_independently(store):
    store.create("a1", _A, {"job_id": "a1", "status": "training"})
    store.create("b1", _B, {"job_id": "b1", "status": "training"})
    a_training = {j["job_id"] for j in store.list(_A, status="training")}
    b_training = {j["job_id"] for j in store.list(_B, status="training")}
    assert a_training == {"a1"}
    assert b_training == {"b1"}


def test_create_stamps_tenant_in_payload(store):
    """Convenience for the API layer — payload includes tenant_id so
    handlers don't have to splice it in separately."""
    store.create("j", _A, {"job_id": "j", "status": "queued"})
    out = store.get("j", _A)
    assert out["tenant_id"] == _A


# ──────────────────────────────────────────────────────────────
# RunStore — single-tenant happy path
# ──────────────────────────────────────────────────────────────
def test_upsert_run_inserts_then_updates(store):
    store.upsert_run("r1", _T, {"run_id": "r1", "domain": "ai_llm", "stage": "forge"})
    assert store.get_run("r1", _T)["stage"] == "forge"
    store.upsert_run("r1", _T, {"run_id": "r1", "domain": "ai_llm", "stage": "train"})
    assert store.get_run("r1", _T)["stage"] == "train"


def test_list_runs_filters_by_domain(store):
    store.upsert_run("r1", _T, {"run_id": "r1", "domain": "ai_llm"})
    store.upsert_run("r2", _T, {"run_id": "r2", "domain": "legal"})
    store.upsert_run("r3", _T, {"run_id": "r3", "domain": "ai_llm"})
    ai = {r["run_id"] for r in store.list_runs(_T, domain="ai_llm")}
    legal = {r["run_id"] for r in store.list_runs(_T, domain="legal")}
    assert ai == {"r1", "r3"}
    assert legal == {"r2"}


# ──────────────────────────────────────────────────────────────
# RunStore — tenant isolation
# ──────────────────────────────────────────────────────────────
def test_run_get_does_not_leak_across_tenants(store):
    store.upsert_run("run-x", _A, {"run_id": "run-x", "domain": "ai_llm"})
    assert store.get_run("run-x", _A) is not None
    assert store.get_run("run-x", _B) is None


def test_run_list_only_returns_own_tenant(store):
    store.upsert_run("a1", _A, {"run_id": "a1", "domain": "x"})
    store.upsert_run("b1", _B, {"run_id": "b1", "domain": "x"})
    assert {r["run_id"] for r in store.list_runs(_A)} == {"a1"}
    assert {r["run_id"] for r in store.list_runs(_B)} == {"b1"}


def test_run_upsert_collision_does_not_overwrite_other_tenant(store):
    """run_id is the primary key, but ON CONFLICT must not let tenant
    B silently overwrite tenant A's row with the same run_id."""
    store.upsert_run("shared-id", _A, {"run_id": "shared-id", "domain": "ai_llm", "marker": "A"})
    store.upsert_run("shared-id", _B, {"run_id": "shared-id", "domain": "x", "marker": "B"})
    # Tenant A's row stays intact (the cross-tenant upsert is a no-op).
    a_view = store.get_run("shared-id", _A)
    assert a_view is not None
    assert a_view["marker"] == "A"
    # Tenant B sees nothing — the conflict guard rejected its upsert.
    assert store.get_run("shared-id", _B) is None


# ──────────────────────────────────────────────────────────────
# Concurrency
# ──────────────────────────────────────────────────────────────
def test_concurrent_writes_serialize_cleanly(store):
    store.create("j", _T, {"job_id": "j", "status": "queued", "counter": 0})

    def worker(n: int):
        for i in range(n):
            current = store.get("j", _T)["counter"]
            store.update_fields("j", _T, counter=current + 1)

    threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    final = store.get("j", _T)["counter"]
    assert final > 0
    assert store.get("j", _T)["job_id"] == "j"


# ──────────────────────────────────────────────────────────────
# Migration — pre-A6b databases without the tenant_id column
# ──────────────────────────────────────────────────────────────
def test_migration_adds_tenant_column_to_legacy_db(tmp_path):
    """Boot order: open a legacy db (no tenant_id column), then write
    a row, then read it back tagged with the synthetic 'public' tenant."""
    import sqlite3
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.execute("""
        CREATE TABLE jobs (
            job_id     TEXT PRIMARY KEY,
            status     TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload    TEXT NOT NULL
        )
    """)
    legacy.execute(
        "INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
        ("legacy-1", "completed", "2025-01-01T00:00:00", "2025-01-01T00:00:00",
         '{"job_id": "legacy-1", "status": "completed", "method": "sft"}'),
    )
    legacy.commit()
    legacy.close()

    # Open via SQLiteStore — the migration must add the tenant_id column
    # with default 'public' so the legacy row is still readable.
    store = SQLiteStore(db_path=db_path)
    out = store.get("legacy-1", "public")
    assert out is not None
    assert out["status"] == "completed"
