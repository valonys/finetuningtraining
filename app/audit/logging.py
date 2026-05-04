"""
app/audit/logging.py
────────────────────
Append-only audit log for the API surface (A6b of the Lane A blueprint).

A new line lands in ``outputs/audit/events_<YYYY-MM-DD>.jsonl`` for
every audited action. Each line is a fully self-contained JSON object
(timestamp, tenant_id, user_id, action, target_id, model_version,
metadata) — no normalization, no foreign keys, no opportunity for the
log to drift out of sync with the records it's auditing.

Why JSONL on disk by default:
  * Append + fsync is atomic for line-sized writes on POSIX, so even a
    process crash mid-flush leaves the file readable up to the last
    complete line.
  * Daily rotation keeps individual files searchable with
    ``grep`` / ``jq`` for compliance reviews.
  * Zero infrastructure required — the audit trail exists from day one
    of the deploy, before anyone provisions a centralized log sink.

When real centralization is needed (SIEM, S3 + Athena, CloudWatch
Logs subscription) it ships the JSONL files directly — no schema
translation. A future ``PostgresAuditBackend`` can be added behind
the same ``AuditLogger`` interface as a driver swap.

The action vocabulary is intentionally small. Adding a new action
type means adding a member to ``AuditAction``; freeform strings would
defeat the purpose of having a typed log. Roles / claims-based
authorization decisions are NOT logged here — they belong in a
separate access-decision log (A6c if needed).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────
class AuditAction(str, Enum):
    """Standardized action labels. Add a new member rather than logging
    a freeform string — enums keep the audit log queryable without
    full-text grep."""

    # Training jobs
    JOB_CREATED = "job_created"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"

    # Model registry (A3)
    REGISTRY_REGISTER = "registry_register"
    REGISTRY_PROMOTE = "registry_promote"
    REGISTRY_ROLLBACK = "registry_rollback"

    # Inference
    CHAT = "chat"
    INFERENCE_RELOAD = "inference_reload"

    # Data Forge
    UPLOAD = "upload"
    INGEST = "ingest"
    HARVEST_YOUTUBE = "harvest_youtube"
    HARVEST_ARXIV = "harvest_arxiv"
    HARVEST_CODE = "harvest_code"
    BUILD_DATASET = "build_dataset"

    # Domain configs
    DOMAIN_CONFIG_CREATE = "domain_config_create"


# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────
class AuditEvent(BaseModel):
    """One line in the audit log. All fields except ``metadata`` are
    indexable projections — the lookup story is grep + jq today, but
    keeping these typed makes the future Postgres switch trivial."""

    timestamp: str
    tenant_id: str
    user_id: str | None = None
    action: AuditAction
    target_id: str | None = None        # job_id, model_version, file name
    model_version: str | None = None    # for inference/registry actions
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────
class AuditLogger:
    """Append-only audit logger with daily file rotation."""

    def __init__(
        self,
        audit_dir: Path | str = "outputs/audit",
        *,
        clock: "callable[[], datetime] | None" = None,
    ):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._now = clock or (lambda: datetime.now(timezone.utc))

    def _file_for(self, dt: datetime) -> Path:
        return self.audit_dir / f"events_{dt.date().isoformat()}.jsonl"

    def log(
        self,
        *,
        tenant_id: str,
        action: AuditAction,
        user_id: str | None = None,
        target_id: str | None = None,
        model_version: str | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Append one event. Returns the event so callers can include
        it in the response payload if useful (typically not — audit is
        a side effect)."""
        now = self._now()
        event = AuditEvent(
            timestamp=now.isoformat(timespec="microseconds"),
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            target_id=target_id,
            model_version=model_version,
            success=success,
            metadata=metadata or {},
        )
        path = self._file_for(now)
        line = event.model_dump_json() + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # Sandboxes / network filesystems sometimes refuse
                    # fsync — non-fatal, the line is already in the OS
                    # page cache and will land on disk eventually.
                    pass
        return event

    def query(
        self,
        *,
        tenant_id: str | None = None,
        action: AuditAction | None = None,
        target_id: str | None = None,
        on_date: date | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        """Scan the on-disk log files and return matching events.

        Cheap for compliance spot-checks (one day's worth of traffic
        is typically a few thousand lines). For full-history queries
        ship the files into a real query engine — the per-day layout
        is designed to make that streaming-ingest friendly.
        """
        files: Iterable[Path]
        if on_date is not None:
            f = self.audit_dir / f"events_{on_date.isoformat()}.jsonl"
            files = [f] if f.is_file() else []
        else:
            files = sorted(self.audit_dir.glob("events_*.jsonl"))

        out: list[AuditEvent] = []
        for path in files:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = AuditEvent.model_validate_json(line)
                        except Exception as exc:
                            logger.warning(
                                f"⚠️  malformed audit line in {path.name}: {exc}"
                            )
                            continue
                        if tenant_id is not None and event.tenant_id != tenant_id:
                            continue
                        if action is not None and event.action != action:
                            continue
                        if target_id is not None and event.target_id != target_id:
                            continue
                        out.append(event)
                        if limit is not None and len(out) >= limit:
                            return out
            except OSError as exc:
                logger.warning(f"⚠️  could not read {path}: {exc}")
        return out


# ──────────────────────────────────────────────────────────────
# Process-wide default
# ──────────────────────────────────────────────────────────────
_default: AuditLogger | None = None
_default_lock = threading.Lock()


def default_logger() -> AuditLogger:
    """Return a process-wide AuditLogger rooted at ``outputs/audit``.
    Override the directory via ``VALONY_AUDIT_DIR``."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                root = os.environ.get("VALONY_AUDIT_DIR", "outputs/audit")
                _default = AuditLogger(audit_dir=root)
                logger.info(f"📓 Audit logger initialized at {root}")
    return _default
