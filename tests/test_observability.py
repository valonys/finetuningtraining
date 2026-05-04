"""
Unit tests for app.observability — CostTracker + SLOEvaluator.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.observability import (
    CostTracker,
    SLOEvaluator,
    SLOResult,
    SLOThresholds,
)


# ──────────────────────────────────────────────────────────────
# CostTracker
# ──────────────────────────────────────────────────────────────
def test_cost_tracker_default_rate_known_pair(tmp_path):
    tracker = CostTracker(metrics_dir=tmp_path)
    out = tracker.record(
        backend="ollama-cloud",
        model="nemotron-3-super",
        tokens_in=1_000_000,
        tokens_out=500_000,
    )
    # 1M in × $0.20 + 500k out × $0.40 = $0.20 + $0.20 = $0.40
    assert out["usd"] == pytest.approx(0.40)


def test_cost_tracker_default_fallback_for_unknown_model(tmp_path):
    tracker = CostTracker(metrics_dir=tmp_path)
    out = tracker.record(
        backend="ollama-cloud",
        model="some-future-model",
        tokens_in=1_000_000,
        tokens_out=0,
    )
    # Falls back to 'ollama-cloud:default' = $0.20 per 1M in
    assert out["usd"] == pytest.approx(0.20)


def test_cost_tracker_unknown_backend_charges_zero(tmp_path, caplog):
    tracker = CostTracker(metrics_dir=tmp_path)
    out = tracker.record(
        backend="never-heard-of-it",
        model="x",
        tokens_in=1_000,
        tokens_out=1_000,
    )
    assert out["usd"] == 0.0


def test_cost_tracker_warns_only_once_per_unknown_pair(tmp_path, caplog):
    tracker = CostTracker(metrics_dir=tmp_path)
    import logging
    caplog.set_level(logging.WARNING, logger="app.observability.cost")
    for _ in range(5):
        tracker.record(backend="weird", model="thing", tokens_in=10, tokens_out=10)
    warns = [r for r in caplog.records if "No cost rate" in r.message]
    assert len(warns) == 1


def test_cost_tracker_totals_split_by_version(tmp_path):
    tracker = CostTracker(metrics_dir=tmp_path)
    tracker.record(
        backend="ollama-cloud", model="nemotron-3-super",
        tokens_in=1_000_000, tokens_out=0, version="stable",
    )
    tracker.record(
        backend="ollama-cloud", model="nemotron-3-super",
        tokens_in=2_000_000, tokens_out=0, version="canary",
    )
    totals = tracker.totals()
    by_pair = {(p["backend"], p["model"], p["version"]): p for p in totals["by_pair"]}
    stable = by_pair[("ollama-cloud", "nemotron-3-super", "stable")]
    canary = by_pair[("ollama-cloud", "nemotron-3-super", "canary")]
    assert stable["requests"] == 1
    assert canary["requests"] == 1
    assert stable["usd"] == pytest.approx(0.20)
    assert canary["usd"] == pytest.approx(0.40)
    assert totals["grand_total_usd"] == pytest.approx(0.60)


def test_cost_tracker_snapshot_writes_json_artifact(tmp_path):
    tracker = CostTracker(metrics_dir=tmp_path)
    tracker.record(
        backend="ollama-cloud", model="nemotron-3-super",
        tokens_in=100, tokens_out=50,
    )
    out = tracker.snapshot()
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["by_pair"][0]["requests"] == 1
    assert "snapshot_at" in payload


def test_cost_tracker_reset_clears_totals(tmp_path):
    tracker = CostTracker(metrics_dir=tmp_path)
    tracker.record(
        backend="ollama-cloud", model="nemotron-3-super",
        tokens_in=100, tokens_out=50,
    )
    assert tracker.totals()["by_pair"]
    tracker.reset()
    assert tracker.totals()["by_pair"] == []
    assert tracker.totals()["grand_total_usd"] == 0.0


# ──────────────────────────────────────────────────────────────
# SLOEvaluator
# ──────────────────────────────────────────────────────────────
def test_slo_all_thresholds_pass(tmp_path):
    ev = SLOEvaluator(metrics_dir=tmp_path)
    res = ev.evaluate(
        latency_samples_ms=[100, 200, 300, 400, 500],
        error_count=0,
        total_count=100,
        quality_probe_outcomes=[True, True, True, True],
    )
    assert res.passed is True
    assert res.halt_recommended is False
    assert res.checks["latency_p95"]["passed"] is True
    assert res.checks["error_rate"]["passed"] is True
    assert res.checks["quality_probe"]["passed"] is True


def test_slo_latency_p95_breach_fails(tmp_path):
    ev = SLOEvaluator(
        thresholds=SLOThresholds(latency_p95_ms_max=200.0),
        metrics_dir=tmp_path,
    )
    res = ev.evaluate(
        latency_samples_ms=[100] * 95 + [10_000] * 5,  # p95 way over 200
        error_count=0,
        total_count=100,
    )
    assert res.passed is False
    assert res.halt_recommended is True
    assert res.checks["latency_p95"]["passed"] is False


def test_slo_error_rate_breach_fails(tmp_path):
    ev = SLOEvaluator(
        thresholds=SLOThresholds(error_rate_max=0.01),
        metrics_dir=tmp_path,
    )
    res = ev.evaluate(
        latency_samples_ms=[100],
        error_count=10,   # 10% > 1%
        total_count=100,
    )
    assert res.passed is False
    assert res.checks["error_rate"]["observed"] == pytest.approx(0.10)


def test_slo_quality_probe_breach_fails(tmp_path):
    ev = SLOEvaluator(
        thresholds=SLOThresholds(quality_probe_min=0.90),
        metrics_dir=tmp_path,
    )
    res = ev.evaluate(
        latency_samples_ms=[100],
        error_count=0,
        total_count=100,
        quality_probe_outcomes=[True, True, False, True, False],   # 60%
    )
    assert res.passed is False
    assert res.checks["quality_probe"]["passed"] is False


def test_slo_no_samples_passes_latency_trivially(tmp_path):
    """Zero samples should not be treated as 'latency 0' = pass and
    also not as a breach. With no data we trivially pass."""
    ev = SLOEvaluator(metrics_dir=tmp_path)
    res = ev.evaluate(
        latency_samples_ms=[],
        error_count=0,
        total_count=0,
    )
    assert res.passed is True
    assert res.checks["latency_p95"]["observed_ms"] == 0.0


def test_slo_snapshot_writes_artifact(tmp_path):
    ev = SLOEvaluator(metrics_dir=tmp_path)
    res = ev.evaluate(
        latency_samples_ms=[100, 200],
        error_count=0,
        total_count=2,
    )
    out = ev.snapshot(res)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["passed"] is True
    assert payload["sample_count"] == 2
