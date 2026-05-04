"""
app/persistence/postgres.py
───────────────────────────
PostgreSQL backend implementing the JobStore + RunStore Protocols
defined in ``store.py`` (A6b — production profile).

Why Postgres now: A5 shipped the SQLite store with the same
Protocols. A6b makes Postgres the production option behind an env
flag (``VALONY_PERSISTENCE_BACKEND=postgres``). Driver swap, not a
new layer — every call site already speaks the Protocol.

Library: ``psycopg[binary]`` (psycopg3). Connection pooling via
``psycopg_pool.ConnectionPool`` so concurrent uvicorn workers don't
each open and tear down their own raw connections.

Schema mirrors SQLite: ``jobs`` and ``runs`` tables with
``tenant_id`` + JSONB ``payload``. Composite indexes on
``(tenant_id, status)`` and ``(tenant_id, updated_at)``. JSONB lets
us add field-level queries later (e.g. "show me jobs whose
``payload.method = sft``") without a schema migration.

Cross-tenant safety: the same WHERE clauses as SQLite — ``WHERE
job_id = %s AND tenant_id = %s``. Postgres adds row-level security
as a backstop option (RLS policies on the tables); we don't
configure RLS by default since the application-layer guard is
already strong enough and RLS adds operational complexity (you have
to ``SET app.tenant_id`` per session). When that operational story
matures, RLS becomes a one-time policy creation.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id     TEXT PRIMARY KEY,
        tenant_id  TEXT NOT NULL DEFAULT 'public',
        status     TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        payload    JSONB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id     TEXT PRIMARY KEY,
        tenant_id  TEXT NOT NULL DEFAULT 'public',
        domain     TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        payload    JSONB NOT NULL
    )
    """,
]

_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_jobs_tenant_status ON jobs(tenant_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_tenant_updated ON jobs(tenant_id, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_tenant_domain ON runs(tenant_id, domain)",
    "CREATE INDEX IF NOT EXISTS idx_runs_tenant_updated ON runs(tenant_id, updated_at DESC)",
]


def _migrate_add_tenant_column(cur, table: str) -> None:
    """Postgres ``ADD COLUMN IF NOT EXISTS`` is supported natively, so
    this is a one-liner (vs the introspection dance SQLite needs)."""
    cur.execute(
        f"ALTER TABLE {table} "
        f"ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'public'"
    )


class PostgresStore:
    """psycopg-pool backed JobStore + RunStore. Same surface as
    SQLiteStore so it slots in as a drop-in driver swap."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        min_pool: int = 1,
        max_pool: int = 10,
    ):
        # Lazy import so environments without psycopg installed
        # (everyone in the SQLite-default profile) don't pay an import
        # cost on app boot.
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - install issue
            raise RuntimeError(
                "psycopg-pool not installed — pip install 'psycopg[binary]' "
                "psycopg-pool to enable the Postgres backend"
            ) from exc

        self.dsn = dsn or os.environ.get("VALONY_POSTGRES_DSN")
        if not self.dsn:
            raise ValueError(
                "Postgres DSN required — set VALONY_POSTGRES_DSN or pass dsn="
            )
        self._pool = ConnectionPool(
            self.dsn, min_size=min_pool, max_size=max_pool, open=True
        )
        self._init_schema()

    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for stmt in _TABLES_SQL:
                    cur.execute(stmt)
                _migrate_add_tenant_column(cur, "jobs")
                _migrate_add_tenant_column(cur, "runs")
                for stmt in _INDEXES_SQL:
                    cur.execute(stmt)
            conn.commit()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    # ── JobStore ──────────────────────────────────────────────
    def create(self, job_id: str, tenant_id: str, payload: dict[str, Any]) -> None:
        status = str(payload.get("status", "queued"))
        payload = {**payload, "tenant_id": tenant_id}
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO jobs (job_id, tenant_id, status, payload) "
                    "VALUES (%s, %s, %s, %s::jsonb)",
                    (job_id, tenant_id, status, json.dumps(payload, default=str)),
                )
            conn.commit()

    def get(self, job_id: str, tenant_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM jobs WHERE job_id = %s AND tenant_id = %s",
                    (job_id, tenant_id),
                )
                row = cur.fetchone()
        return row[0] if row else None

    def list(self, tenant_id: str, *, status: str | None = None) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if status is None:
                    cur.execute(
                        "SELECT payload FROM jobs WHERE tenant_id = %s "
                        "ORDER BY updated_at DESC",
                        (tenant_id,),
                    )
                else:
                    cur.execute(
                        "SELECT payload FROM jobs WHERE tenant_id = %s "
                        "AND status = %s ORDER BY updated_at DESC",
                        (tenant_id, status),
                    )
                rows = cur.fetchall()
        return [r[0] for r in rows]

    def update_fields(
        self, job_id: str, tenant_id: str, **fields: Any
    ) -> dict[str, Any] | None:
        """Read-modify-write under a row-level lock so concurrent
        updaters don't lose data. Returns None on unknown id OR
        cross-tenant id."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM jobs WHERE job_id = %s AND tenant_id = %s "
                    "FOR UPDATE",
                    (job_id, tenant_id),
                )
                row = cur.fetchone()
                if row is None:
                    conn.rollback()
                    return None
                payload = row[0]
                payload.update(fields)
                payload["tenant_id"] = tenant_id
                new_status = str(payload.get("status", "queued"))
                cur.execute(
                    "UPDATE jobs SET status = %s, updated_at = now(), "
                    "payload = %s::jsonb WHERE job_id = %s AND tenant_id = %s",
                    (new_status, json.dumps(payload, default=str), job_id, tenant_id),
                )
            conn.commit()
        return payload

    def delete(self, job_id: str, tenant_id: str) -> bool:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM jobs WHERE job_id = %s AND tenant_id = %s",
                    (job_id, tenant_id),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    # ── RunStore ──────────────────────────────────────────────
    def upsert_run(self, run_id: str, tenant_id: str, payload: dict[str, Any]) -> None:
        domain = str(payload.get("domain", ""))
        payload = {**payload, "tenant_id": tenant_id}
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # ON CONFLICT path guards against cross-tenant overwrite
                # by gating the UPDATE on tenant_id matching.
                cur.execute(
                    "INSERT INTO runs (run_id, tenant_id, domain, payload) "
                    "VALUES (%s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (run_id) DO UPDATE SET "
                    "  domain     = EXCLUDED.domain, "
                    "  updated_at = now(), "
                    "  payload    = EXCLUDED.payload "
                    "WHERE runs.tenant_id = EXCLUDED.tenant_id",
                    (run_id, tenant_id, domain, json.dumps(payload, default=str)),
                )
            conn.commit()

    def get_run(self, run_id: str, tenant_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM runs WHERE run_id = %s AND tenant_id = %s",
                    (run_id, tenant_id),
                )
                row = cur.fetchone()
        return row[0] if row else None

    def list_runs(
        self, tenant_id: str, *, domain: str | None = None
    ) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if domain is None:
                    cur.execute(
                        "SELECT payload FROM runs WHERE tenant_id = %s "
                        "ORDER BY updated_at DESC",
                        (tenant_id,),
                    )
                else:
                    cur.execute(
                        "SELECT payload FROM runs WHERE tenant_id = %s "
                        "AND domain = %s ORDER BY updated_at DESC",
                        (tenant_id, domain),
                    )
                rows = cur.fetchall()
        return [r[0] for r in rows]

    # ── Lifecycle ─────────────────────────────────────────────
    def close(self) -> None:
        self._pool.close()
