"""
app/pipeline — A2 batch pipeline orchestration.

Public surface:
    PipelineRunner   Stage executor with crash-safe state + resume
    RunContext       Per-run state passed to each stage
    Stage            Stage descriptor (name + callable + idempotency key fn)
    StageResult      What a stage returns (status, artifacts, gate verdict)
    StageStatus      Enum: pending, running, completed, failed, skipped
"""
from .runner import (
    PipelineRunner,
    RunContext,
    Stage,
    StageResult,
    StageStatus,
)

__all__ = [
    "PipelineRunner",
    "RunContext",
    "Stage",
    "StageResult",
    "StageStatus",
]
