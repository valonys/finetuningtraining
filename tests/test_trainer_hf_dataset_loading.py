"""
Regression tests for BaseAgnosticTrainer._load_dataset — specifically the
HF-dataset path, which had two bugs that kept breaking the 02_sft notebook:

  1. `max_samples` in `hf_dataset_config` was documented but silently
     ignored, so asking for 500 rows got all 52k.
  2. `input_column` / `output_column` weren't applied, so Alpaca's
     `{"instruction","input","output"}` never got renamed to the
     `{"instruction","response"}` shape SFT expects.

We don't import `datasets` (too heavy for CI); instead we hand-roll a
`_FakeDataset` that implements just the surface _load_dataset touches:
`__len__`, `select`, `rename_columns`, `column_names`, and `__getitem__`.
"""
from __future__ import annotations

import sys
import types
from typing import Any, Iterable

import pytest

from app.trainers.base import BaseAgnosticTrainer, TrainRequest


class _FakeDataset:
    """Minimal stand-in for `datasets.Dataset` covering the calls in
    BaseAgnosticTrainer._load_dataset."""
    def __init__(self, rows: list[dict]):
        self._rows = list(rows)

    def __len__(self): return len(self._rows)
    def __getitem__(self, i): return self._rows[i]
    def __iter__(self): return iter(self._rows)

    @property
    def column_names(self) -> list[str]:
        return list(self._rows[0].keys()) if self._rows else []

    def select(self, indices: Iterable[int]) -> "_FakeDataset":
        idxs = list(indices)
        return _FakeDataset([self._rows[i] for i in idxs])

    def rename_columns(self, mapping: dict) -> "_FakeDataset":
        return _FakeDataset([
            {mapping.get(k, k): v for k, v in row.items()}
            for row in self._rows
        ])


class _StubTrainer(BaseAgnosticTrainer):
    """BaseAgnosticTrainer is abstract; fill in abstractmethods so we
    can construct it and call `_load_dataset` in isolation."""
    method = "test"
    def _format_dataset(self, dataset, tokenizer): return dataset
    def _run(self, model, tokenizer, dataset) -> float: return 0.0


@pytest.fixture
def alpaca_like_ds():
    return _FakeDataset([
        {"instruction": f"Q{i}", "input": "", "output": f"A{i}"}
        for i in range(50)
    ])


def _make_trainer(monkeypatch, stub_dataset, hf_cfg):
    """Wire up `_StubTrainer` so `datasets.load_dataset` returns our stub
    and the hardware/template machinery is mocked."""
    import app.trainers.base as base_mod

    monkeypatch.setattr(base_mod, "detect_hardware",
        lambda: types.SimpleNamespace(tier="cpu", effective_memory_gb=8))
    monkeypatch.setattr(base_mod, "resolve_profile", lambda hw: types.SimpleNamespace(
        training_backend="trl", max_seq_length=512, per_device_batch_size=1,
        gradient_accumulation_steps=1, gradient_checkpointing=False, load_in_4bit=False,
    ))
    monkeypatch.setattr(base_mod, "get_template_for", lambda mid: types.SimpleNamespace(
        name="qwen", response_template=None,
    ))

    # Install a fake `datasets` module so `from datasets import ...` works.
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *a, **kw: stub_dataset
    fake_datasets.load_from_disk = lambda *a, **kw: stub_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    req = TrainRequest(
        config={"domain_name": "test"},
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hf_dataset_config=hf_cfg,
    )
    return _StubTrainer(req)


def test_max_samples_actually_caps_rows(monkeypatch, alpaca_like_ds):
    trainer = _make_trainer(monkeypatch, alpaca_like_ds, {
        "repo_id": "tatsu-lab/alpaca",
        "split": "train",
        "max_samples": 10,
    })
    ds = trainer._load_dataset()
    assert len(ds) == 10, "max_samples=10 should cap the dataset to 10 rows"


def test_max_samples_none_means_full_dataset(monkeypatch, alpaca_like_ds):
    trainer = _make_trainer(monkeypatch, alpaca_like_ds, {
        "repo_id": "tatsu-lab/alpaca",
        "split": "train",
    })
    ds = trainer._load_dataset()
    assert len(ds) == 50


def test_max_samples_larger_than_dataset_is_noop(monkeypatch, alpaca_like_ds):
    trainer = _make_trainer(monkeypatch, alpaca_like_ds, {
        "repo_id": "tatsu-lab/alpaca",
        "split": "train",
        "max_samples": 99999,
    })
    ds = trainer._load_dataset()
    assert len(ds) == 50


def test_output_column_gets_renamed_to_response(monkeypatch, alpaca_like_ds):
    """Alpaca-shaped rows should come out with a `response` column so the
    SFT formatter's `_pick('response', ...)` matches."""
    trainer = _make_trainer(monkeypatch, alpaca_like_ds, {
        "repo_id": "tatsu-lab/alpaca",
        "split": "train",
        "input_column": "instruction",
        "output_column": "output",
        "max_samples": 5,
    })
    ds = trainer._load_dataset()
    assert "response" in ds.column_names
    assert "output" not in ds.column_names
    assert "instruction" in ds.column_names        # no accidental rename
    assert ds[0]["response"] == "A0"


def test_loss_history_sink_auto_initialised(monkeypatch, alpaca_like_ds):
    """Notebook callers don't pass a loss_history_sink, but they still
    need result['loss_history'] to exist for plotting. Verify the
    trainer auto-allocates an empty list when no sink is supplied."""
    trainer = _make_trainer(monkeypatch, alpaca_like_ds, {
        "repo_id": "tatsu-lab/alpaca",
        "split": "train",
    })
    assert trainer.loss_history_sink == []
    assert trainer.loss_history_sink is not None


def test_loss_history_sink_caller_reference_preserved(monkeypatch, alpaca_like_ds):
    """The FastAPI background task passes its own list (the same one
    /v1/jobs/{id} reads). Make sure we keep the *same* object so the
    HTTP endpoint sees live updates -- not a fresh allocation."""
    import app.trainers.base as base_mod
    monkeypatch.setattr(base_mod, "detect_hardware",
        lambda: __import__("types").SimpleNamespace(tier="cpu", effective_memory_gb=8))
    monkeypatch.setattr(base_mod, "resolve_profile", lambda hw: __import__("types").SimpleNamespace(
        training_backend="trl", max_seq_length=512, per_device_batch_size=1,
        gradient_accumulation_steps=1, gradient_checkpointing=False, load_in_4bit=False,
    ))
    monkeypatch.setattr(base_mod, "get_template_for", lambda mid: __import__("types").SimpleNamespace(
        name="qwen", response_template=None,
    ))

    caller_list: list = []
    req = TrainRequest(
        config={"domain_name": "test"},
        base_model_id="x",
        loss_history_sink=caller_list,
    )
    t = _StubTrainer(req)
    assert t.loss_history_sink is caller_list   # same object, not a copy


def test_input_column_gets_renamed_to_instruction(monkeypatch):
    """When upstream names the prompt `prompt`, the caller passes
    input_column='prompt' and the trainer renames it to `instruction`."""
    ds_in = _FakeDataset([{"prompt": "Hi", "answer": "Hello"}])
    trainer = _make_trainer(monkeypatch, ds_in, {
        "repo_id": "x/y",
        "split": "train",
        "input_column": "prompt",
        "output_column": "answer",
    })
    ds = trainer._load_dataset()
    assert "instruction" in ds.column_names
    assert "response"    in ds.column_names
    assert ds[0]["instruction"] == "Hi"
    assert ds[0]["response"]    == "Hello"
