"""
app/persistence/store.py
────────────────────────
SQLite-backed durable state for training jobs and pipeline runs (A5).

Why this exists:
  * Pre-A5, training jobs lived in `app/main.py:job_registry: Dict[str, JobStatus]`
    — a process-local dict. A `uvicorn` restart wiped every in-flight job;
    multiple workers would diverge. The acceptance gate for A5 explicitly
    requires "jobs survive a uvicorn restart".
  * Pre-A5, pipeline runs were file-only (JSONL under `outputs/runs/`).
    That works for a single operator on one machine but isn't queryable.
    We mirror run state into SQLite so observability surfaces (SLO,
    cost) can read from a relational layer without parsing JSONL.

Design choices:
  * One SQLite file at `outputs/.persistence/store.db` with two tables:
    `jobs` and `runs`. Both keep an opaque JSON `payload` column plus a
    handful of indexed projection columns (status, domain, timestamps)
    so simple filtered queries don't need to deserialize every row.
  * WAL journaling so a long-running write (e.g. progress callback
    every training step) doesn't block a concurrent read (the
    `GET /v1/jobs/{id}` poll from the UI).
  * Schema migrations are dead-simple: a single `_init_schema` call
    that runs `CREATE TABLE IF NOT EXISTS`. A real migration framework
    is overkill for two tables; revisit when A6 swaps Postgres in.
  * JobStore / RunStore are typed Protocols so A6 can drop a Postgres
    backend in without changing call sites.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Protocols — what every backend must implement
# ──────────────────────────────────────────────────────────────
class JobStore(Protocol):
    def create(self, job_id: str, payload: dict[str, Any]) -> None: ...
    def get(self, job_id: str) -> dict[str, Any] | None: ...
    def list(self, *, status: str | None = None) -> list[dict[str, Any]]: ...
    def update_fields(self, job_id: str, **fields: Any) -> dict[str, Any] | None: ...
    def delete(self, job_id: str) -> bool: ...


class RunStore(Protocol):
    def upsert_run(self, run_id: str, payload: dict[str, Any]) -> None: ...
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...
    def list_runs(self, *, domain: str | None = None) -> list[dict[str, Any]]: ...


# ──────────────────────────────────────────────────────────────
# SQLite implementation — used for both JobStore and RunStore
# ──────────────────────────────────────────────────────────────
_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id     TEXT PRIMARY KEY,
        status     TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id     TEXT PRIMARY KEY,
        domain     TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload    TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_domain ON runs(domain)",
    "CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_at)",
]


class SQLiteStore:
    """SQLite-backed JobStore + RunStore in one object. Thread-safe via
    a per-instance lock around every operation — SQLite's connection
    object isn't thread-safe for concurrent calls even with
    ``check_same_thread=False``, so we serialize via the lock and let
    WAL handle reader/writer concurrency at the file level."""

    def __init__(self, db_path: Path | str = "outputs/.persistence/store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        # WAL journaling so multiple processes can read while one writes.
        # Within this process we still serialize via _lock — sqlite3's
        # Connection isn't safe to share across threads concurrently.
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)

    @staticmethod
    def _now() -> str:
        # Microsecond precision so two writes in the same second still
        # order deterministically by ``updated_at`` in list queries.
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    # ── JobStore ──────────────────────────────────────────────
    def create(self, job_id: str, payload: dict[str, Any]) -> None:
        ts = self._now()
        status = str(payload.get("status", "queued"))
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (job_id, status, created_at, updated_at, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, status, ts, ts, json.dumps(payload, default=str)),
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list(self, *, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT payload FROM jobs ORDER BY updated_at DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT payload FROM jobs WHERE status = ? ORDER BY updated_at DESC",
                    (status,),
                ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def update_fields(
        self, job_id: str, **fields: Any
    ) -> dict[str, Any] | None:
        """Patch one or more fields on a stored job. Returns the new
        full payload, or None if the job_id is unknown. Read-modify-write
        happens under the lock so concurrent updates don't lose data."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            payload = json.loads(row["payload"])
            payload.update(fields)
            ts = self._now()
            new_status = str(payload.get("status", "queued"))
            self._conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ?, payload = ? "
                "WHERE job_id = ?",
                (new_status, ts, json.dumps(payload, default=str), job_id),
            )
            return payload

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            return cur.rowcount > 0

    # ── RunStore ──────────────────────────────────────────────
    def upsert_run(self, run_id: str, payload: dict[str, Any]) -> None:
        ts = self._now()
        domain = str(payload.get("domain", ""))
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, domain, created_at, updated_at, payload) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "  domain     = excluded.domain, "
                "  updated_at = excluded.updated_at, "
                "  payload    = excluded.payload",
                (run_id, domain, ts, ts, json.dumps(payload, default=str)),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_runs(self, *, domain: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if domain is None:
                rows = self._conn.execute(
                    "SELECT payload FROM runs ORDER BY updated_at DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT payload FROM runs WHERE domain = ? ORDER BY updated_at DESC",
                    (domain,),
                ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    # ── Lifecycle ─────────────────────────────────────────────
    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ──────────────────────────────────────────────────────────────
# Process-wide default singleton
# ──────────────────────────────────────────────────────────────
_default_store: SQLiteStore | None = None
_default_lock = threading.Lock()


def default_store() -> SQLiteStore:
    """Return a process-wide SQLiteStore at the default path. Override
    via ``VALONY_PERSISTENCE_PATH`` env (e.g. for tests / containers
    with a different mount layout)."""
    global _default_store
    if _default_store is None:
        with _default_lock:
            if _default_store is None:
                path = os.environ.get(
                    "VALONY_PERSISTENCE_PATH",
                    "outputs/.persistence/store.db",
                )
                _default_store = SQLiteStore(db_path=path)
                logger.info(f"📦 Persistence store initialized at {path}")
    return _default_store
