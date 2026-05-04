"""
app/registry/schemas.py
───────────────────────
Pydantic models + lifecycle enum for the A3 model registry.

A ``ModelVersion`` is the unit of release: one exported artifact with
full lineage back to the dataset that produced it and the eval report
that gated it. The ``status`` field walks a small state machine:

    candidate → staging → production
       │           │           │
       │           ▼           ▼
       └───→  rolled_back  ←───┘
                  │
                  ▼
              staging  (re-attempt allowed)

A ``PromotionEvent`` is the audit-log entry emitted on every transition.
A ``RollbackResult`` packages the two records produced by a rollback
(the demoted production + the freshly-promoted replacement, if any).
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"


class ModelVersion(BaseModel):
    """Frozen record of a registered model. Each on-disk JSONL line is
    one snapshot of this record after a state change — the registry
    materializes current state by replaying the file."""

    model_version: str
    domain: str
    base_model_id: str
    adapter_path: str
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    dataset_manifest_path: str | None = None
    eval_report_path: str | None = None
    status: ModelStatus
    promoted_from: str | None = None
    created_at: str
    updated_at: str
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class PromotionEvent(BaseModel):
    """Audit-log entry for a single status transition."""

    model_version: str
    domain: str
    from_status: ModelStatus | None       # None when version is first created
    to_status: ModelStatus
    actor: str | None = None
    reason: str | None = None
    timestamp: str


class RollbackResult(BaseModel):
    """Outcome of ``ModelRegistry.rollback()``: the version moved out of
    production, plus the version that took its place (if any)."""

    rolled_back: ModelVersion
    new_production: ModelVersion | None = None
