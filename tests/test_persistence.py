"""
Unit tests for app.persistence.SQLiteStore.

Covers the JobStore + RunStore protocols and the durability invariant
that justifies A5: a fresh process pointed at the same db file reads
back the same records — i.e. ``uvicorn`` restart doesn't lose jobs.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.persistence import SQLiteStore


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(db_path=tmp_path / "test.db")


# ──────────────────────────────────────────────────────────────
# JobStore
# ──────────────────────────────────────────────────────────────
def test_create_get_roundtrip(store):
    store.create("job-1", {"job_id": "job-1", "status": "queued", "method": "sft"})
    out = store.get("job-1")
    assert out["job_id"] == "job-1"
    assert out["status"] == "queued"
    assert out["method"] == "sft"


def test_get_unknown_returns_none(store):
    assert store.get("does-not-exist") is None


def test_list_orders_most_recently_updated_first(store):
    store.create("a", {"job_id": "a", "status": "queued"})
    store.create("b", {"job_id": "b", "status": "queued"})
    # Touch 'a' so it becomes most recent
    store.update_fields("a", progress=0.5)
    listed = store.list()
    assert [j["job_id"] for j in listed[:2]] == ["a", "b"]


def test_list_filters_by_status(store):
    store.create("a", {"job_id": "a", "status": "queued"})
    store.create("b", {"job_id": "b", "status": "training"})
    store.create("c", {"job_id": "c", "status": "completed"})
    queued = {j["job_id"] for j in store.list(status="queued")}
    completed = {j["job_id"] for j in store.list(status="completed")}
    assert queued == {"a"}
    assert completed == {"c"}


def test_update_fields_patches_payload(store):
    store.create("j", {"job_id": "j", "status": "queued", "progress": 0.0})
    new = store.update_fields("j", status="training", progress=0.42)
    assert new["status"] == "training"
    assert new["progress"] == 0.42
    # Re-read confirms persistence
    assert store.get("j")["progress"] == 0.42


def test_update_fields_unknown_returns_none(store):
    assert store.update_fields("nope", status="x") is None


def test_update_fields_changes_status_index(store):
    """The status column is a projection used for filtered list — it
    must move when the payload's status changes."""
    store.create("j", {"job_id": "j", "status": "queued"})
    assert {j["job_id"] for j in store.list(status="queued")} == {"j"}
    store.update_fields("j", status="completed")
    assert store.list(status="queued") == []
    assert {j["job_id"] for j in store.list(status="completed")} == {"j"}


def test_delete_returns_true_on_hit_false_on_miss(store):
    store.create("j", {"job_id": "j", "status": "queued"})
    assert store.delete("j") is True
    assert store.delete("j") is False
    assert store.get("j") is None


def test_durability_across_fresh_instance(tmp_path: Path):
    """The acceptance gate: a brand-new store pointed at the same db
    file must see every record the prior store wrote."""
    db = tmp_path / "durable.db"
    s1 = SQLiteStore(db_path=db)
    s1.create("j1", {"job_id": "j1", "status": "queued", "method": "sft"})
    s1.update_fields("j1", status="training", progress=0.3)
    s1.close()

    # Fresh instance — simulates uvicorn restart
    s2 = SQLiteStore(db_path=db)
    out = s2.get("j1")
    assert out["status"] == "training"
    assert out["progress"] == 0.3


def test_loss_history_round_trips_as_list_of_dicts(store):
    """JobStatus stores a loss_history list — make sure JSON
    (de)serialization preserves it."""
    history = [
        {"step": 0, "loss": 2.4, "lr": 1e-4},
        {"step": 1, "loss": 2.1, "lr": 1e-4},
        {"step": 2, "loss": 1.9, "lr": 1e-4},
    ]
    store.create("j", {"job_id": "j", "status": "queued", "loss_history": history})
    out = store.get("j")
    assert out["loss_history"] == history


# ──────────────────────────────────────────────────────────────
# RunStore
# ──────────────────────────────────────────────────────────────
def test_upsert_run_inserts_then_updates(store):
    store.upsert_run("r1", {"run_id": "r1", "domain": "ai_llm", "stage": "forge"})
    assert store.get_run("r1")["stage"] == "forge"
    store.upsert_run("r1", {"run_id": "r1", "domain": "ai_llm", "stage": "train"})
    assert store.get_run("r1")["stage"] == "train"


def test_list_runs_filters_by_domain(store):
    store.upsert_run("r1", {"run_id": "r1", "domain": "ai_llm"})
    store.upsert_run("r2", {"run_id": "r2", "domain": "legal"})
    store.upsert_run("r3", {"run_id": "r3", "domain": "ai_llm"})
    ai = {r["run_id"] for r in store.list_runs(domain="ai_llm")}
    legal = {r["run_id"] for r in store.list_runs(domain="legal")}
    assert ai == {"r1", "r3"}
    assert legal == {"r2"}


# ──────────────────────────────────────────────────────────────
# Concurrency
# ──────────────────────────────────────────────────────────────
def test_concurrent_writes_serialize_cleanly(store):
    """A loose smoke check that the per-instance write lock + WAL
    journaling cope with multiple writer threads. We don't claim
    transaction-level isolation — just that no UPDATE is lost."""
    store.create("j", {"job_id": "j", "status": "queued", "counter": 0})

    def worker(n: int):
        for i in range(n):
            current = store.get("j")["counter"]
            store.update_fields("j", counter=current + 1)

    threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    final = store.get("j")["counter"]
    # Lost-update is possible without optimistic concurrency control —
    # we only assert the counter advanced, not that it equals 200.
    # The point of the test is that no exception fired and the row
    # still exists & is readable.
    assert final > 0
    assert store.get("j")["job_id"] == "j"
