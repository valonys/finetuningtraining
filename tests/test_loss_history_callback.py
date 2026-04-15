"""
Unit tests for LossHistoryCallback — the TrainerCallback that feeds the
live loss chart on the /v1/jobs/{id} endpoint.

We exercise the callback directly with fake `state` / `logs` objects so
the tests run fast and don't require a real HF Trainer.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.trainers.callbacks import LossHistoryCallback, make_loss_callback


def _state(step: int) -> SimpleNamespace:
    """Minimal stand-in for HuggingFace TrainerState."""
    return SimpleNamespace(global_step=step)


def test_appends_per_step_entries():
    sink: list = []
    cb = LossHistoryCallback(sink)

    cb.on_log(args=None, state=_state(1), control=None,
              logs={"loss": 2.34, "learning_rate": 2e-4, "grad_norm": 1.1, "epoch": 0.05})
    cb.on_log(args=None, state=_state(2), control=None,
              logs={"loss": 2.10, "learning_rate": 2e-4, "grad_norm": 0.9, "epoch": 0.10})

    assert len(sink) == 2
    first, second = sink
    assert first["step"] == 1 and first["loss"] == 2.34
    assert second["step"] == 2 and second["loss"] == 2.10
    # All the tracked fields are materialised even when missing in a log
    for entry in sink:
        assert set(entry.keys()) >= {"step", "loss", "learning_rate", "grad_norm", "epoch", "ts"}


def test_ignores_summary_log_without_loss():
    """The Trainer's on_train_end log has `train_loss` but no per-step
    `loss` or `grad_norm`. We don't want that polluting the chart."""
    sink: list = []
    cb = LossHistoryCallback(sink)
    cb.on_log(args=None, state=_state(100), control=None,
              logs={"train_loss": 1.5, "train_runtime": 42.0})
    assert sink == []


def test_ignores_empty_logs():
    sink: list = []
    cb = LossHistoryCallback(sink)
    cb.on_log(args=None, state=_state(0), control=None, logs=None)
    cb.on_log(args=None, state=_state(0), control=None, logs={})
    assert sink == []


def test_respects_max_entries_fifo():
    """When the cap is exceeded, oldest entries drop first."""
    sink: list = []
    cb = LossHistoryCallback(sink, max_entries=5)
    for i in range(12):
        cb.on_log(args=None, state=_state(i), control=None,
                  logs={"loss": float(i), "grad_norm": 0.1})
    assert len(sink) == 5
    # Oldest (steps 0-6) were dropped; newest (7-11) remain
    assert [e["step"] for e in sink] == [7, 8, 9, 10, 11]


def test_make_loss_callback_returns_none_when_transformers_missing(monkeypatch):
    import app.trainers.callbacks as mod
    monkeypatch.setattr(mod, "_TRANSFORMERS_AVAILABLE", False)
    assert make_loss_callback([]) is None


def test_missing_optional_fields_become_none():
    sink: list = []
    cb = LossHistoryCallback(sink)
    cb.on_log(args=None, state=_state(3), control=None, logs={"loss": 1.5})
    assert len(sink) == 1
    entry = sink[0]
    assert entry["loss"] == 1.5
    assert entry["learning_rate"] is None
    assert entry["grad_norm"] is None
    assert entry["epoch"] is None
