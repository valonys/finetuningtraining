"""
app/audit — A6b append-only audit log for compliance traces.

Public surface:
    AuditAction        Enum of standardized action labels
    AuditEvent         Pydantic record persisted per request
    AuditLogger        Append-only writer with daily rotation
    default_logger     Process-wide singleton at outputs/audit/
"""
from .logging import (
    AuditAction,
    AuditEvent,
    AuditLogger,
    default_logger,
)

__all__ = [
    "AuditAction",
    "AuditEvent",
    "AuditLogger",
    "default_logger",
]
