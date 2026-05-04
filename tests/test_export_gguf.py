"""
Unit tests for app.trainers.export.merge_and_export.

We stub peft + transformers + the llama.cpp subprocess invocations so the
test runs offline on CPU in milliseconds. The fake quantize step writes a
small payload to the expected output path, which lets us exercise the
real metadata, hashing, and rollback-pointer logic without llama.cpp
actually being installed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def fake_adapter(tmp_path: Path) -> Path:
    adapter = tmp_path / "outputs" / "ai_llm"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r": 16, "lora_alpha": 32}')
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00fake-lora-weights")
    (adapter / "tokenizer.json").write_text("{}")
    return adapter


@pytest.fixture
def fake_llama_cpp(tmp_path: Path) -> Path:
    """Fake a vendored llama.cpp install with a present convert script and
    quantize binary. Both files just have to exist + be executable."""
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    convert = root / "convert_hf_to_gguf.py"
    quantize = root / "build" / "bin" / "llama-quantize"
    convert.write_text("#!/usr/bin/env python3\n")
    quantize.write_text("#!/bin/sh\n")
    quantize.chmod(0o755)
    return root


@pytest.fixture
def stub_peft_and_transformers(monkeypatch):
    """Replace the heavy ML deps with no-op stand-ins."""
    captures: dict = {}

    class _FakeMerged:
        def save_pretrained(self, path, safe_serialization=True):
            captures["merged_save_path"] = path
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text('{"model_type": "fake"}')
            (Path(path) / "model.safetensors").write_bytes(b"merged-weights")

    class _FakePeftModel:
        def merge_and_unload(self):
            captures["merge_called"] = True
            return _FakeMerged()

    peft_mod = types.ModuleType("peft")
    peft_mod.PeftModel = type("PeftModel", (), {
        "from_pretrained": staticmethod(lambda base, path: _FakePeftModel())
    })
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    class _FakeBase:
        pass

    class _FakeTokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}")

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {
        "from_pretrained": staticmethod(lambda mid, **_: _FakeBase())
    })
    transformers_mod.AutoTokenizer = type("AutoTokenizer", (), {
        "from_pretrained": staticmethod(lambda mid, **_: _FakeTokenizer())
    })
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    return captures


@pytest.fixture
def stub_subprocess(monkeypatch, fake_llama_cpp: Path):
    """Replace subprocess.run so the convert step is a no-op and the
    quantize step writes a fake GGUF blob to its output path."""
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        # Convert step: ``python convert.py merged_dir --outfile X --outtype f16``
        if "--outfile" in cmd:
            outfile = Path(cmd[cmd.index("--outfile") + 1])
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_bytes(b"FAKE_GGUF_F16_PAYLOAD" * 64)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        # Quantize step: ``llama-quantize input output Q4_K_M``
        # cmd[0] is the binary, cmd[1] is input, cmd[2] is output, cmd[3] is quant
        if str(cmd[0]).endswith("llama-quantize"):
            outfile = Path(cmd[2])
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_bytes(b"FAKE_GGUF_QUANTIZED_PAYLOAD" * 32)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected subprocess call: {cmd}")

    monkeypatch.setattr("app.trainers.export.subprocess.run", fake_run)
    return calls


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────
def test_merge_and_export_happy_path(
    tmp_path, fake_adapter, fake_llama_cpp,
    stub_peft_and_transformers, stub_subprocess,
):
    from app.trainers.export import merge_and_export

    out = tmp_path / "artifacts"
    result = merge_and_export(
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path=str(fake_adapter),
        output_dir=str(out),
        quant="Q4_K_M",
        llama_cpp_path=str(fake_llama_cpp),
    )

    # ── return contract ────────────────────────────────────────
    assert set(result.keys()) >= {
        "gguf_path", "metadata_path", "sha256", "latest_pointer",
        "quant", "base_model_id", "adapter_sha256", "exported_at",
    }
    assert result["quant"] == "Q4_K_M"
    assert result["base_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert len(result["sha256"]) == 64       # hex sha256
    assert len(result["adapter_sha256"]) == 64

    # ── files on disk ──────────────────────────────────────────
    gguf = Path(result["gguf_path"])
    meta = Path(result["metadata_path"])
    latest = Path(result["latest_pointer"])
    assert gguf.is_file() and gguf.stat().st_size > 0
    assert meta.is_file()
    assert latest.exists()
    assert gguf.name == "ai_llm-q4_k_m.gguf"
    assert meta.name == "ai_llm-q4_k_m.metadata.json"
    assert latest.name == "latest.gguf"

    # ── metadata sidecar contents ──────────────────────────────
    payload = json.loads(meta.read_text())
    assert payload["base_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert payload["quant"] == "Q4_K_M"
    assert payload["gguf_filename"] == gguf.name
    assert payload["file_sha256"] == result["sha256"]
    assert payload["file_bytes"] == gguf.stat().st_size
    assert payload["adapter_sha256"] == result["adapter_sha256"]
    assert "exported_at" in payload

    # ── peft merge actually invoked ────────────────────────────
    assert stub_peft_and_transformers["merge_called"] is True
    assert "merged_save_path" in stub_peft_and_transformers

    # ── two subprocess calls: convert + quantize ───────────────
    assert len(stub_subprocess) == 2
    convert_call, quantize_call = stub_subprocess
    assert "--outfile" in convert_call
    assert "Q4_K_M" in quantize_call
    assert str(fake_llama_cpp) in convert_call[1]  # convert script path


def test_merge_and_export_raises_on_missing_adapter(tmp_path, fake_llama_cpp):
    from app.trainers.export import merge_and_export
    with pytest.raises(FileNotFoundError, match="Adapter dir invalid"):
        merge_and_export(
            base_model_id="x/y",
            adapter_path=str(tmp_path / "nope"),
            output_dir=str(tmp_path / "out"),
            llama_cpp_path=str(fake_llama_cpp),
        )


def test_merge_and_export_raises_on_missing_llama_cpp(tmp_path, fake_adapter):
    from app.trainers.export import merge_and_export
    with pytest.raises(FileNotFoundError, match="llama.cpp converter missing"):
        merge_and_export(
            base_model_id="x/y",
            adapter_path=str(fake_adapter),
            output_dir=str(tmp_path / "out"),
            llama_cpp_path=str(tmp_path / "no_llama"),
        )


def test_merge_and_export_raises_on_missing_quantize_binary(
    tmp_path, fake_adapter, fake_llama_cpp,
):
    """Convert script present but quantize binary absent."""
    quantize = fake_llama_cpp / "build" / "bin" / "llama-quantize"
    quantize.unlink()

    from app.trainers.export import merge_and_export
    with pytest.raises(FileNotFoundError, match="llama-quantize binary missing"):
        merge_and_export(
            base_model_id="x/y",
            adapter_path=str(fake_adapter),
            output_dir=str(tmp_path / "out"),
            llama_cpp_path=str(fake_llama_cpp),
        )


def test_f16_pass_through_skips_quantize(
    tmp_path, fake_adapter, fake_llama_cpp,
    stub_peft_and_transformers, stub_subprocess,
):
    from app.trainers.export import merge_and_export
    result = merge_and_export(
        base_model_id="x/y",
        adapter_path=str(fake_adapter),
        output_dir=str(tmp_path / "out"),
        quant="F16",
        llama_cpp_path=str(fake_llama_cpp),
    )
    # Only the convert step should run -- quantize is bypassed for f16
    assert len(stub_subprocess) == 1
    assert "--outfile" in stub_subprocess[0]
    assert Path(result["gguf_path"]).name == "ai_llm-f16.gguf"


def test_rollback_pointer_updates_across_exports(
    tmp_path, fake_adapter, fake_llama_cpp,
    stub_peft_and_transformers, stub_subprocess,
):
    """Two successive exports: the prior artifact survives, the symlink
    moves to the new one."""
    from app.trainers.export import merge_and_export
    out = tmp_path / "artifacts"

    first = merge_and_export(
        base_model_id="x/y",
        adapter_path=str(fake_adapter),
        output_dir=str(out),
        quant="Q4_K_M",
        llama_cpp_path=str(fake_llama_cpp),
    )
    second = merge_and_export(
        base_model_id="x/y",
        adapter_path=str(fake_adapter),
        output_dir=str(out),
        quant="Q5_K_M",
        llama_cpp_path=str(fake_llama_cpp),
    )

    # both gguf artifacts persist
    assert Path(first["gguf_path"]).is_file()
    assert Path(second["gguf_path"]).is_file()

    # latest.gguf now points at the second export
    latest = Path(second["latest_pointer"])
    assert latest.exists()
    target = latest.resolve() if latest.is_symlink() else latest
    if latest.is_symlink():
        assert latest.readlink().name == Path(second["gguf_path"]).name
    else:
        # copy fallback (non-symlink platforms)
        assert target.read_bytes() == Path(second["gguf_path"]).read_bytes()
