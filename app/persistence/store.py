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
# Every method takes ``tenant_id`` as a required positional arg
# (A6b). There is intentionally no escape hatch — tenant isolation
# is enforced at the storage boundary, not by handler discipline.
# Cross-tenant admin views (when needed for ops dashboards) get
# explicit ``list_all_admin``-style methods that scream "you bypassed
# tenant" at the call site. Today: not needed.


class JobStore(Protocol):
    def create(self, job_id: str, tenant_id: str, payload: dict[str, Any]) -> None: ...
    def get(self, job_id: str, tenant_id: str) -> dict[str, Any] | None: ...
    def list(self, tenant_id: str, *, status: str | None = None) -> list[dict[str, Any]]: ...
    def update_fields(self, job_id: str, tenant_id: str, **fields: Any) -> dict[str, Any] | None: ...
    def delete(self, job_id: str, tenant_id: str) -> bool: ...


class RunStore(Protocol):
    def upsert_run(self, run_id: str, tenant_id: str, payload: dict[str, Any]) -> None: ...
    def get_run(self, run_id: str, tenant_id: str) -> dict[str, Any] | None: ...
    def list_runs(self, tenant_id: str, *, domain: str | None = None) -> list[dict[str, Any]]: ...


# ──────────────────────────────────────────────────────────────
# SQLite implementation — used for both JobStore and RunStore
# ──────────────────────────────────────────────────────────────
# Tables FIRST, then migrations bring legacy tables forward, THEN
# indexes — the indexes reference tenant_id which a pre-A6b table
# doesn't have until the migration runs.
_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id     TEXT PRIMARY KEY,
        tenant_id  TEXT NOT NULL DEFAULT 'public',
        status     TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id     TEXT PRIMARY KEY,
        tenant_id  TEXT NOT NULL DEFAULT 'public',
        domain     TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload    TEXT NOT NULL
    )
    """,
]

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_jobs_tenant_status ON jobs(tenant_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_tenant_updated ON jobs(tenant_id, updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_tenant_domain ON runs(tenant_id, domain)",
    "CREATE INDEX IF NOT EXISTS idx_runs_tenant_updated ON runs(tenant_id, updated_at)",
]


# Additive migrations — SQLite has no ALTER TABLE ... ADD COLUMN
# IF NOT EXISTS until very recent versions, so we introspect first.
def _migrate_add_tenant_column(conn, table: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if "tenant_id" not in cols:
        # Existing pre-A6b rows get the synthetic 'public' tenant so
        # auth-disabled workflows keep accessing them.
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'public'"
        )


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
            # 1. Create-if-missing tables (no-op when they already exist)
            for stmt in _TABLES:
                self._conn.execute(stmt)
            # 2. Bring pre-A6b tables forward by adding the tenant_id
            #    column. Safe to re-run; the function is a no-op on
            #    tables that already have the column.
            _migrate_add_tenant_column(self._conn, "jobs")
            _migrate_add_tenant_column(self._conn, "runs")
            # 3. Indexes last — they reference tenant_id which the
            #    migration above just guaranteed exists.
            for stmt in _INDEXES:
                self._conn.execute(stmt)

    @staticmethod
    def _now() -> str:
        # Microsecond precision so two writes in the same second still
        # order deterministically by ``updated_at`` in list queries.
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    # ── JobStore (tenant-scoped) ──────────────────────────────
    def create(self, job_id: str, tenant_id: str, payload: dict[str, Any]) -> None:
        ts = self._now()
        status = str(payload.get("status", "queued"))
        # Stamp tenant into payload so on read we return it without
        # an extra column projection.
        payload = {**payload, "tenant_id": tenant_id}
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (job_id, tenant_id, status, created_at, updated_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, tenant_id, status, ts, ts, json.dumps(payload, default=str)),
            )

    def get(self, job_id: str, tenant_id: str) -> dict[str, Any] | None:
        """Return the job iff it belongs to ``tenant_id``. Cross-tenant
        access returns None — no information leak about whether the
        id exists in some other tenant."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM jobs WHERE job_id = ? AND tenant_id = ?",
                (job_id, tenant_id),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list(self, tenant_id: str, *, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT payload FROM jobs WHERE tenant_id = ? "
                    "ORDER BY updated_at DESC",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT payload FROM jobs WHERE tenant_id = ? AND status = ? "
                    "ORDER BY updated_at DESC",
                    (tenant_id, status),
                ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def update_fields(
        self, job_id: str, tenant_id: str, **fields: Any
    ) -> dict[str, Any] | None:
        """Patch fields on a job iff it belongs to ``tenant_id``.
        Returns None on unknown id OR cross-tenant id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM jobs WHERE job_id = ? AND tenant_id = ?",
                (job_id, tenant_id),
            ).fetchone()
            if row is None:
                return None
            payload = json.loads(row["payload"])
            payload.update(fields)
            payload["tenant_id"] = tenant_id     # keep mirror consistent
            ts = self._now()
            new_status = str(payload.get("status", "queued"))
            self._conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ?, payload = ? "
                "WHERE job_id = ? AND tenant_id = ?",
                (new_status, ts, json.dumps(payload, default=str), job_id, tenant_id),
            )
            return payload

    def delete(self, job_id: str, tenant_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM jobs WHERE job_id = ? AND tenant_id = ?",
                (job_id, tenant_id),
            )
            return cur.rowcount > 0

    # ── RunStore (tenant-scoped) ──────────────────────────────
    def upsert_run(self, run_id: str, tenant_id: str, payload: dict[str, Any]) -> None:
        ts = self._now()
        domain = str(payload.get("domain", ""))
        payload = {**payload, "tenant_id": tenant_id}
        with self._lock:
            # ON CONFLICT: a different tenant trying to upsert the same
            # run_id would silently overwrite. We reject by gating the
            # update on the tenant matching — different-tenant upserts
            # become no-ops at the storage level.
            self._conn.execute(
                "INSERT INTO runs (run_id, tenant_id, domain, created_at, updated_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "  domain     = excluded.domain, "
                "  updated_at = excluded.updated_at, "
                "  payload    = excluded.payload "
                "WHERE runs.tenant_id = excluded.tenant_id",
                (run_id, tenant_id, domain, ts, ts, json.dumps(payload, default=str)),
            )

    def get_run(self, run_id: str, tenant_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM runs WHERE run_id = ? AND tenant_id = ?",
                (run_id, tenant_id),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_runs(
        self, tenant_id: str, *, domain: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            if domain is None:
                rows = self._conn.execute(
                    "SELECT payload FROM runs WHERE tenant_id = ? "
                    "ORDER BY updated_at DESC",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT payload FROM runs WHERE tenant_id = ? AND domain = ? "
                    "ORDER BY updated_at DESC",
                    (tenant_id, domain),
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


def default_store():
    """Return a process-wide store, resolved per env:

      VALONY_PERSISTENCE_BACKEND=postgres → PostgresStore (A6b
        production profile). Requires psycopg-pool + a DSN at
        VALONY_POSTGRES_DSN.
      anything else (default) → SQLiteStore at
        VALONY_PERSISTENCE_PATH (default outputs/.persistence/store.db).

    Both implementations satisfy the JobStore + RunStore Protocols,
    so call sites don't care which is active."""
    global _default_store
    if _default_store is None:
        with _default_lock:
            if _default_store is None:
                backend = os.environ.get("VALONY_PERSISTENCE_BACKEND", "sqlite").lower()
                if backend == "postgres":
                    from .postgres import PostgresStore
                    _default_store = PostgresStore()
                    logger.info(
                        f"📦 Persistence store initialized: postgres "
                        f"(dsn={_default_store.dsn.split('@')[-1] if '@' in _default_store.dsn else 'set'})"
                    )
                else:
                    path = os.environ.get(
                        "VALONY_PERSISTENCE_PATH",
                        "outputs/.persistence/store.db",
                    )
                    _default_store = SQLiteStore(db_path=path)
                    logger.info(f"📦 Persistence store initialized: sqlite ({path})")
    return _default_store
