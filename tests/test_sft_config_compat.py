"""
Regression tests for SFTConfig kwarg translation in AgnosticSFTTrainer.

Reason: TRL 0.16 renamed `max_seq_length` -> `max_length`. Pinning trl
pre-0.16 was the original workaround; the durable fix is to translate
the kwarg at runtime by inspecting the installed SFTConfig signature.
We exercise that translation logic with a fake SFTConfig that exposes
either the old or the new kwarg name.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest


# ── Fake SFTConfig flavors ────────────────────────────────────────
class _OldSFTConfig:
    """TRL <= 0.15 -- accepts max_seq_length."""
    def __init__(self, *, output_dir, max_seq_length=None, **rest):
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.rest = rest


class _NewSFTConfig:
    """TRL >= 0.16 -- accepts max_length, refuses max_seq_length."""
    def __init__(self, *, output_dir, max_length=None, **rest):
        self.output_dir = output_dir
        self.max_length = max_length
        self.rest = rest


def _build_kwargs_for(SFTConfig, max_seq) -> dict:
    """Replicates the relevant slice of AgnosticSFTTrainer._run_trl
    so we can unit-test the translation in isolation, without spinning
    up a real model + dataset."""
    import inspect
    wanted = {
        "output_dir": "/tmp/x",
        "max_seq_length": max_seq,
        "logging_steps": 5,
    }
    cfg_params = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_seq_length" not in cfg_params and "max_length" in cfg_params:
        wanted["max_length"] = wanted.pop("max_seq_length")
    return {k: v for k, v in wanted.items() if k in cfg_params}


def test_old_trl_keeps_max_seq_length():
    kwargs = _build_kwargs_for(_OldSFTConfig, 2048)
    assert kwargs == {"output_dir": "/tmp/x", "max_seq_length": 2048}
    cfg = _OldSFTConfig(**kwargs)
    assert cfg.max_seq_length == 2048


def test_new_trl_translates_to_max_length():
    kwargs = _build_kwargs_for(_NewSFTConfig, 2048)
    assert "max_seq_length" not in kwargs
    assert kwargs.get("max_length") == 2048
    cfg = _NewSFTConfig(**kwargs)
    assert cfg.max_length == 2048


def test_unknown_kwargs_dropped_silently():
    """`logging_steps` isn't on either fake SFTConfig -- should be filtered
    out, not raise TypeError."""
    kwargs = _build_kwargs_for(_OldSFTConfig, 1024)
    # The real translation logic also drops unknown kwargs; verify by
    # running through it with a minimal SFTConfig that only takes one.
    class _MinimalCfg:
        def __init__(self, *, output_dir): self.output_dir = output_dir
    out = _build_kwargs_for(_MinimalCfg, 1024)
    assert out == {"output_dir": "/tmp/x"}
    _MinimalCfg(**out)   # would TypeError if we passed extras
