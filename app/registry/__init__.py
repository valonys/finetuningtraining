"""
app/registry — A3 model registry with promote / rollback control.

Public surface:
    ModelRegistry      JSONL-backed registry (CRUD + promote + rollback)
    ModelStatus        Lifecycle enum (candidate / staging / production / rolled_back)
    ModelVersion       Frozen record of one trained-and-exported artifact
    PromotionEvent     Audit-log entry for a status transition
    RollbackResult     Return type for ModelRegistry.rollback()
    InvalidTransition  Raised on disallowed state machine transitions
    UnknownVersion     Raised when a model_version id is not found
"""
from .schemas import (
    ModelStatus,
    ModelVersion,
    PromotionEvent,
    RollbackResult,
)
from .model_registry import (
    ModelRegistry,
    InvalidTransition,
    UnknownVersion,
    default_registry,
)

__all__ = [
    "ModelRegistry",
    "ModelStatus",
    "ModelVersion",
    "PromotionEvent",
    "RollbackResult",
    "InvalidTransition",
    "UnknownVersion",
    "default_registry",
]
