"""
app/persistence — durable state for jobs and pipeline runs.

A5 introduced the SQLite implementation; A6b adds the Postgres
backend behind ``VALONY_PERSISTENCE_BACKEND=postgres`` (driver swap,
same Protocols).

Public surface:
    JobStore        Protocol: create / get / list / update_fields / delete
    RunStore        Protocol: upsert_run / get_run / list_runs
    SQLiteStore     Single-file SQLite impl (WAL mode), default
    PostgresStore   psycopg-pool impl, opt-in via env (A6b)
    default_store   Singleton resolved per env (sqlite | postgres)
"""
from .store import (
    JobStore,
    RunStore,
    SQLiteStore,
    default_store,
)

# PostgresStore is exported but its module is imported lazily inside
# ``default_store`` so environments without psycopg installed don't
# pay the import cost on boot.

__all__ = [
    "JobStore",
    "RunStore",
    "SQLiteStore",
    "default_store",
]
