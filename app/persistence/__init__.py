"""
app/persistence — A5 durable state for jobs and pipeline runs.

Public surface:
    JobStore        Protocol: create / get / list / update_fields / delete
    RunStore        Protocol: upsert_run / get_run / list_runs
    SQLiteStore     Concrete impl backing both (single .db file, WAL mode)
    default_store   Process-wide singleton at outputs/.persistence/store.db
"""
from .store import (
    JobStore,
    RunStore,
    SQLiteStore,
    default_store,
)

__all__ = [
    "JobStore",
    "RunStore",
    "SQLiteStore",
    "default_store",
]
