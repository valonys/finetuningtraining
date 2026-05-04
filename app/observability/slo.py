"""
app/observability/slo.py
────────────────────────
Service Level Objective evaluator (A5).

A5 ships *evaluation*, not *enforcement*: the evaluator computes
pass/fail per check and writes a JSON artifact, but the canary
auto-abort closed loop (which would actually demote a misbehaving
canary) is intentionally deferred. We need real traffic to validate
threshold choices before the runtime acts on them autonomously.

Three checks:
  1. Latency p95 against ``latency_p95_ms_max``
  2. Error rate against ``error_rate_max`` (errors / total)
  3. Quality-probe success rate against ``quality_probe_min``

Caller (typically the inference manager wrapping a canary window)
gathers samples then calls ``evaluate(...)``; the result has a
boolean ``passed`` and per-check verdicts, plus a ``halt_recommended``
flag the runtime can read in a future sprint to actually demote.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class SLOThresholds:
    """Defaults are conservative — a chat endpoint should clear these
    on day one. Tighten per-domain via the canary config when you have
    enough traffic to know what 'normal' looks like."""

    latency_p95_ms_max: float = 5_000.0      # 5s end-to-end
    error_rate_max: float = 0.05             # 5%
    quality_probe_min: float = 0.80          # 80% probe pass rate


@dataclass
class SLOResult:
    passed: bool
    halt_recommended: bool
    checks: dict[str, dict]                  # per-check verdict + observed value
    sample_count: int
    window_started_at: str
    window_ended_at: str
    thresholds: dict[str, float]


@dataclass
class SLOEvaluator:
    """Stateless: each ``evaluate`` call is independent. The metrics_dir
    only matters when the caller asks to ``snapshot`` the result."""

    thresholds: SLOThresholds = field(default_factory=SLOThresholds)
    metrics_dir: Path = field(default_factory=lambda: Path("outputs/metrics"))

    def __post_init__(self):
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        *,
        latency_samples_ms: Iterable[float],
        error_count: int,
        total_count: int,
        quality_probe_outcomes: Iterable[bool] | None = None,
        window_started_at: str | None = None,
    ) -> SLOResult:
        latencies = list(latency_samples_ms)
        n = len(latencies)
        # p95 needs at least one sample; if zero, treat as no-data → pass
        # the latency check trivially (caller's job to gather a sample
        # before evaluating in production).
        p95 = _percentile(latencies, 0.95) if latencies else 0.0
        latency_pass = p95 <= self.thresholds.latency_p95_ms_max

        error_rate = (error_count / total_count) if total_count > 0 else 0.0
        error_pass = error_rate <= self.thresholds.error_rate_max

        probe_outcomes = list(quality_probe_outcomes or [])
        if probe_outcomes:
            probe_success = sum(1 for o in probe_outcomes if o) / len(probe_outcomes)
        else:
            probe_success = 1.0   # no probes → don't fail on this dimension
        probe_pass = probe_success >= self.thresholds.quality_probe_min

        all_passed = latency_pass and error_pass and probe_pass
        ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        return SLOResult(
            passed=all_passed,
            halt_recommended=not all_passed,
            checks={
                "latency_p95": {
                    "observed_ms": round(p95, 2),
                    "threshold_ms": self.thresholds.latency_p95_ms_max,
                    "passed": latency_pass,
                },
                "error_rate": {
                    "observed": round(error_rate, 4),
                    "threshold": self.thresholds.error_rate_max,
                    "errors": error_count,
                    "total": total_count,
                    "passed": error_pass,
                },
                "quality_probe": {
                    "observed_success_rate": round(probe_success, 4),
                    "threshold": self.thresholds.quality_probe_min,
                    "probes": len(probe_outcomes),
                    "passed": probe_pass,
                },
            },
            sample_count=n,
            window_started_at=window_started_at or ended_at,
            window_ended_at=ended_at,
            thresholds={
                "latency_p95_ms_max": self.thresholds.latency_p95_ms_max,
                "error_rate_max": self.thresholds.error_rate_max,
                "quality_probe_min": self.thresholds.quality_probe_min,
            },
        )

    def snapshot(self, result: SLOResult) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = self.metrics_dir / f"slo_{ts}.json"
        out.write_text(
            json.dumps(
                {
                    "passed": result.passed,
                    "halt_recommended": result.halt_recommended,
                    "checks": result.checks,
                    "sample_count": result.sample_count,
                    "window_started_at": result.window_started_at,
                    "window_ended_at": result.window_ended_at,
                    "thresholds": result.thresholds,
                },
                indent=2,
            )
            + "\n"
        )
        return out


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile over an unsorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac
