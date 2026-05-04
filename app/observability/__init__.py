"""
app/observability — A5 cost + SLO surfaces.

Public surface:
    CostTracker     Per-request token + USD accounting (in-memory + JSON sink)
    SLOEvaluator    Latency / error-rate / quality-probe gate evaluator
"""
from .cost import CostTracker
from .slo import SLOEvaluator, SLOResult, SLOThresholds

__all__ = [
    "CostTracker",
    "SLOEvaluator",
    "SLOResult",
    "SLOThresholds",
]
