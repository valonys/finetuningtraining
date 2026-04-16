"""
Regression tests for _resolve_attn_impl.

Reason this exists: the cuda_consumer profile flips use_flash_attn=True
for every CUDA GPU, but Colab's free T4 is Turing (compute 7.5) and
flash_attn refuses to load. We must downgrade to SDPA in that case
instead of letting transformers raise.
"""
from __future__ import annotations

import sys
import types

import pytest

from app.trainers.backends import _resolve_attn_impl


def test_returns_none_when_profile_flag_off():
    assert _resolve_attn_impl(False, force_cpu=False) is None


def test_returns_none_when_force_cpu():
    assert _resolve_attn_impl(True, force_cpu=True) is None


def test_falls_back_to_sdpa_when_flash_attn_missing(monkeypatch):
    # Make sure flash_attn appears uninstalled
    import importlib.util as ilu
    real_find_spec = ilu.find_spec
    monkeypatch.setattr(
        ilu, "find_spec",
        lambda name, *a, **kw: None if name == "flash_attn" else real_find_spec(name, *a, **kw),
    )
    assert _resolve_attn_impl(True, force_cpu=False) == "sdpa"


def test_falls_back_to_sdpa_on_pre_ampere_gpu(monkeypatch):
    """T4 / V100 / RTX-2080 (compute 7.x) must NOT request FA2."""
    # Pretend flash_attn IS installed
    import importlib.util as ilu
    real_find_spec = ilu.find_spec
    monkeypatch.setattr(
        ilu, "find_spec",
        lambda name, *a, **kw: types.SimpleNamespace(name="flash_attn")
            if name == "flash_attn" else real_find_spec(name, *a, **kw),
    )
    # Stub torch to look like a Turing GPU is present
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda i: (7, 5),     # Turing
        get_device_name=lambda i: "Tesla T4",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _resolve_attn_impl(True, force_cpu=False) == "sdpa"


def test_picks_flash_attention_2_on_ampere_or_newer(monkeypatch):
    """A100 / L4 / 4090 (compute 8.x or 9.x) should opt in to FA2."""
    import importlib.util as ilu
    real_find_spec = ilu.find_spec
    monkeypatch.setattr(
        ilu, "find_spec",
        lambda name, *a, **kw: types.SimpleNamespace(name="flash_attn")
            if name == "flash_attn" else real_find_spec(name, *a, **kw),
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda i: (8, 0),     # Ampere (A100)
        get_device_name=lambda i: "A100",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _resolve_attn_impl(True, force_cpu=False) == "flash_attention_2"
