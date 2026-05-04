"""
Unit tests for app.inference.canary.CanaryRouter.

Behaviour we care about:
  * Routing splits roughly proportional to canary_pct over many trials
  * Domains without a config fall through to the supplied fallback
  * Per-version metrics accumulate independently
  * Reset clears metrics for one or all domains
  * Bad config (canary_pct out of range) raises early
"""
from __future__ import annotations

import random

import pytest

from app.inference.canary import CanaryConfig, CanaryRouter


def test_no_config_returns_fallback():
    router = CanaryRouter()
    assert router.pick_version("ai_llm", fallback="base") == "base"


def test_canary_pct_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        CanaryConfig(domain="x", stable_version="s", canary_version="c", canary_pct=120)
    with pytest.raises(ValueError):
        CanaryConfig(domain="x", stable_version="s", canary_version="c", canary_pct=-5)


def test_routing_splits_proportionally():
    """1000 trials with seeded RNG and 25% canary should land within
    a few percentage points of the configured split."""
    router = CanaryRouter(rng=random.Random(42))
    router.configure(CanaryConfig(
        domain="ai_llm",
        stable_version="v1",
        canary_version="v2",
        canary_pct=25.0,
    ))
    counts = {"v1": 0, "v2": 0}
    for _ in range(1000):
        v = router.pick_version("ai_llm")
        counts[v] += 1
    # Expect ~250 to v2; allow ±50 (5%) wiggle room
    assert 200 <= counts["v2"] <= 300
    assert counts["v1"] + counts["v2"] == 1000


def test_routing_zero_pct_never_picks_canary():
    router = CanaryRouter(rng=random.Random(0))
    router.configure(CanaryConfig(
        domain="ai_llm",
        stable_version="v1",
        canary_version="v2",
        canary_pct=0.0,
    ))
    for _ in range(100):
        assert router.pick_version("ai_llm") == "v1"


def test_routing_hundred_pct_always_picks_canary():
    router = CanaryRouter(rng=random.Random(0))
    router.configure(CanaryConfig(
        domain="ai_llm",
        stable_version="v1",
        canary_version="v2",
        canary_pct=100.0,
    ))
    for _ in range(100):
        assert router.pick_version("ai_llm") == "v2"


def test_metrics_accumulate_per_version():
    router = CanaryRouter()
    router.record(domain="ai_llm", version="v1", latency_ms=100.0)
    router.record(domain="ai_llm", version="v1", latency_ms=200.0, error=True)
    router.record(domain="ai_llm", version="v2", latency_ms=300.0)
    v1 = router.metrics_for("ai_llm", "v1")
    v2 = router.metrics_for("ai_llm", "v2")
    assert v1["requests"] == 2
    assert v1["errors"] == 1
    assert v1["latencies_ms"] == [100.0, 200.0]
    assert v2["requests"] == 1
    assert v2["errors"] == 0


def test_metrics_quality_probes_filter_none():
    router = CanaryRouter()
    router.record(domain="d", version="v1", latency_ms=100.0)
    router.record(domain="d", version="v1", latency_ms=100.0, quality_probe=True)
    router.record(domain="d", version="v1", latency_ms=100.0, quality_probe=False)
    m = router.metrics_for("d", "v1")
    assert m["quality_probes"] == [True, False]


def test_reset_metrics_per_domain():
    router = CanaryRouter()
    router.record(domain="a", version="v1", latency_ms=100.0)
    router.record(domain="b", version="v1", latency_ms=200.0)
    router.reset_metrics(domain="a")
    assert router.metrics_for("a", "v1")["requests"] == 0
    assert router.metrics_for("b", "v1")["requests"] == 1


def test_reset_metrics_all():
    router = CanaryRouter()
    router.record(domain="a", version="v1", latency_ms=100.0)
    router.record(domain="b", version="v1", latency_ms=200.0)
    router.reset_metrics()
    assert router.metrics_for("a", "v1")["requests"] == 0
    assert router.metrics_for("b", "v1")["requests"] == 0


def test_remove_config_falls_back():
    router = CanaryRouter()
    router.configure(CanaryConfig(
        domain="d", stable_version="s", canary_version="c", canary_pct=50,
    ))
    router.remove("d")
    assert router.pick_version("d", fallback="base") == "base"


def test_canary_then_slo_evaluator_integrates(tmp_path):
    """End-to-end: route requests, record latencies, hand the canary's
    metrics to SLOEvaluator, get a verdict back. This is the closed
    loop A5b will eventually use to auto-abort a bad canary."""
    from app.observability import SLOEvaluator, SLOThresholds

    router = CanaryRouter(rng=random.Random(7))
    router.configure(CanaryConfig(
        domain="ai_llm",
        stable_version="v1",
        canary_version="v2",
        canary_pct=20.0,
    ))
    # Simulate 100 requests
    for _ in range(100):
        v = router.pick_version("ai_llm")
        # Canary version is ~3× slower than stable in this simulation
        latency = 1000.0 if v == "v2" else 300.0
        router.record(domain="ai_llm", version=v, latency_ms=latency)

    canary_metrics = router.metrics_for("ai_llm", "v2")
    ev = SLOEvaluator(
        thresholds=SLOThresholds(latency_p95_ms_max=500.0),
        metrics_dir=tmp_path,
    )
    res = ev.evaluate(
        latency_samples_ms=canary_metrics["latencies_ms"],
        error_count=canary_metrics["errors"],
        total_count=canary_metrics["requests"],
    )
    # 1000ms canary latency > 500ms threshold → should fail
    assert res.passed is False
    assert res.halt_recommended is True
