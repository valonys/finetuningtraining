"""
Unit tests for app.audit.AuditLogger.

The audit log is the compliance backbone of A6b — every test here
locks down a behavior that a future change would silently break if
not asserted (line atomicity, daily rotation, query filters, the
typed action vocabulary).
"""
from __future__ import annotations

import json
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from app.audit import (
    AuditAction,
    AuditEvent,
    AuditLogger,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
class _StaticClock:
    """Returns a fixed datetime so tests can drive daily rotation."""

    def __init__(self, dt: datetime):
        self.dt = dt

    def __call__(self) -> datetime:
        return self.dt


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────
def test_log_writes_one_jsonl_line(tmp_path):
    audit = AuditLogger(audit_dir=tmp_path)
    event = audit.log(
        tenant_id="acme",
        action=AuditAction.JOB_CREATED,
        user_id="alice",
        target_id="job-1",
    )
    assert isinstance(event, AuditEvent)

    files = sorted(tmp_path.glob("events_*.jsonl"))
    assert len(files) == 1
    payload = _read_lines(files[0])
    assert len(payload) == 1
    assert payload[0]["tenant_id"] == "acme"
    assert payload[0]["action"] == "job_created"
    assert payload[0]["user_id"] == "alice"
    assert payload[0]["target_id"] == "job-1"
    assert payload[0]["success"] is True
    assert payload[0]["metadata"] == {}


def test_log_appends_multiple_events(tmp_path):
    audit = AuditLogger(audit_dir=tmp_path)
    audit.log(tenant_id="t1", action=AuditAction.JOB_CREATED, target_id="j1")
    audit.log(tenant_id="t1", action=AuditAction.JOB_COMPLETED, target_id="j1")
    audit.log(tenant_id="t2", action=AuditAction.CHAT, target_id="r-42")

    files = sorted(tmp_path.glob("events_*.jsonl"))
    assert len(files) == 1
    rows = _read_lines(files[0])
    assert len(rows) == 3
    assert [r["action"] for r in rows] == ["job_created", "job_completed", "chat"]


def test_metadata_round_trips(tmp_path):
    audit = AuditLogger(audit_dir=tmp_path)
    meta = {"latency_ms": 1234, "tokens": {"in": 100, "out": 250}, "model_id": "qwen-7b"}
    audit.log(
        tenant_id="acme",
        action=AuditAction.CHAT,
        user_id="alice",
        model_version="acme-20260504-abc12345",
        metadata=meta,
    )
    rows = _read_lines(next(tmp_path.glob("events_*.jsonl")))
    assert rows[0]["metadata"] == meta
    assert rows[0]["model_version"] == "acme-20260504-abc12345"


def test_failure_event_records_success_false(tmp_path):
    audit = AuditLogger(audit_dir=tmp_path)
    audit.log(
        tenant_id="acme",
        action=AuditAction.REGISTRY_PROMOTE,
        target_id="model-x",
        success=False,
        metadata={"reason": "invalid_transition"},
    )
    rows = _read_lines(next(tmp_path.glob("events_*.jsonl")))
    assert rows[0]["success"] is False
    assert rows[0]["metadata"]["reason"] == "invalid_transition"


def test_daily_rotation_writes_separate_files(tmp_path):
    """Three events on three different UTC days must land in three
    different files. Compliance spot-checks rely on this layout."""
    day1 = datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 5, 5, 10, 0, 0, tzinfo=timezone.utc)
    day3 = datetime(2026, 5, 6, 10, 0, 0, tzinfo=timezone.utc)

    clock = _StaticClock(day1)
    audit = AuditLogger(audit_dir=tmp_path, clock=clock)
    audit.log(tenant_id="t", action=AuditAction.JOB_CREATED, target_id="d1")
    clock.dt = day2
    audit.log(tenant_id="t", action=AuditAction.JOB_CREATED, target_id="d2")
    clock.dt = day3
    audit.log(tenant_id="t", action=AuditAction.JOB_CREATED, target_id="d3")

    files = sorted(p.name for p in tmp_path.glob("events_*.jsonl"))
    assert files == [
        "events_2026-05-04.jsonl",
        "events_2026-05-05.jsonl",
        "events_2026-05-06.jsonl",
    ]


def test_query_filters_by_tenant(tmp_path):
    audit = AuditLogger(audit_dir=tmp_path)
    audit.log(tenant_id="acme", action=AuditAction.JOB_CREATED, target_id="a")
    audit.log(tenant_id="acme", action=AuditAction.CHAT, target_id="b")
    audit.log(tenant_id="other", action=AuditAction.JOB_CREATED, target_id="c")
    audit.log(tenant_id="other", action=AuditAction.UPLOAD, target_id="d")

    acme_only = audit.query(tenant_id="acme")
    assert {e.target_id for e in acme_only} == {"a", "b"}

    other_only = audit.query(tenant_id="other")
    assert {e.target_id for e in other_only} == {"c", "d"}


def test_query_filters_by_action(tmp_path):
    audit = AuditLogger(audit_dir=tmp_path)
    audit.log(tenant_id="t", action=AuditAction.JOB_CREATED, target_id="j1")
    audit.log(tenant_id="t", action=AuditAction.JOB_COMPLETED, target_id="j1")
    audit.log(tenant_id="t", action=AuditAction.CHAT, target_id="r1")

    chats = audit.query(action=AuditAction.CHAT)
    assert len(chats) == 1
    assert chats[0].target_id == "r1"


def test_query_filters_by_target_id(tmp_path):
    audit = AuditLogger(audit_dir=tmp_path)
    audit.log(tenant_id="t", action=AuditAction.JOB_CREATED, target_id="j1")
    audit.log(tenant_id="t", action=AuditAction.JOB_COMPLETED, target_id="j1")
    audit.log(tenant_id="t", action=AuditAction.JOB_CREATED, target_id="j2")

    j1 = audit.query(target_id="j1")
    assert len(j1) == 2


def test_query_filters_by_date(tmp_path):
    day1 = datetime(2026, 5, 4, 10, tzinfo=timezone.utc)
    day2 = datetime(2026, 5, 5, 10, tzinfo=timezone.utc)
    clock = _StaticClock(day1)
    audit = AuditLogger(audit_dir=tmp_path, clock=clock)
    audit.log(tenant_id="t", action=AuditAction.JOB_CREATED, target_id="d1")
    clock.dt = day2
    audit.log(tenant_id="t", action=AuditAction.JOB_CREATED, target_id="d2")

    on_4 = audit.query(on_date=date(2026, 5, 4))
    assert {e.target_id for e in on_4} == {"d1"}
    on_5 = audit.query(on_date=date(2026, 5, 5))
    assert {e.target_id for e in on_5} == {"d2"}
    missing = audit.query(on_date=date(2026, 5, 6))
    assert missing == []


def test_query_combines_filters(tmp_path):
    audit = AuditLogger(audit_dir=tmp_path)
    audit.log(tenant_id="acme", action=AuditAction.JOB_CREATED, target_id="j1")
    audit.log(tenant_id="acme", action=AuditAction.CHAT, target_id="r1")
    audit.log(tenant_id="other", action=AuditAction.JOB_CREATED, target_id="j2")

    res = audit.query(tenant_id="acme", action=AuditAction.JOB_CREATED)
    assert len(res) == 1
    assert res[0].target_id == "j1"


def test_query_limit(tmp_path):
    audit = AuditLogger(audit_dir=tmp_path)
    for i in range(10):
        audit.log(tenant_id="t", action=AuditAction.CHAT, target_id=f"r{i}")
    capped = audit.query(action=AuditAction.CHAT, limit=3)
    assert len(capped) == 3


def test_query_skips_malformed_lines(tmp_path):
    """A truncated line (mid-flush crash) shouldn't crash query()."""
    audit = AuditLogger(audit_dir=tmp_path)
    audit.log(tenant_id="t", action=AuditAction.JOB_CREATED, target_id="j1")

    # Append a deliberately malformed line + another good one
    f = next(tmp_path.glob("events_*.jsonl"))
    with f.open("a") as fh:
        fh.write("{not-valid-json\n")
    audit.log(tenant_id="t", action=AuditAction.JOB_COMPLETED, target_id="j1")

    rows = audit.query()
    assert {(e.action.value, e.target_id) for e in rows} == {
        ("job_created", "j1"),
        ("job_completed", "j1"),
    }


def test_concurrent_writes_serialize_cleanly(tmp_path):
    """50 threads × 10 writes each = 500 lines, all complete and parseable."""
    audit = AuditLogger(audit_dir=tmp_path)

    def worker(idx: int):
        for i in range(10):
            audit.log(
                tenant_id=f"t{idx}",
                action=AuditAction.CHAT,
                target_id=f"r-{idx}-{i}",
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    files = sorted(tmp_path.glob("events_*.jsonl"))
    # Could be 1 or 2 files if the test crosses a UTC midnight — both fine.
    total_lines = 0
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            # Each line must parse as a complete event
            json.loads(line)
            total_lines += 1
    assert total_lines == 500


def test_action_enum_values_are_stable_strings():
    """Enum members serialize to specific strings — changing one would
    silently break every prior audit log entry's queryability."""
    assert AuditAction.JOB_CREATED.value == "job_created"
    assert AuditAction.REGISTRY_PROMOTE.value == "registry_promote"
    assert AuditAction.REGISTRY_ROLLBACK.value == "registry_rollback"
    assert AuditAction.CHAT.value == "chat"
    assert AuditAction.UPLOAD.value == "upload"
