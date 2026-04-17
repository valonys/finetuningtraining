"""
Unit tests for app.trainers.hub.push_adapter_to_hub.

We mock huggingface_hub's create_repo and upload_folder so the test
doesn't touch the network. We do create real files in a tmp dir so the
adapter-folder existence check + README staging path are exercised.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _stub_hf_hub(monkeypatch, *, captures: dict):
    """Install a fake `huggingface_hub` module that records call args."""
    mod = types.ModuleType("huggingface_hub")

    def create_repo(**kwargs):
        captures["create_repo"] = kwargs

    def upload_folder(**kwargs):
        captures["upload_folder"] = kwargs
        # The function uploads from a tempdir that gets cleaned up the
        # moment push_adapter_to_hub returns -- so we snapshot file list
        # AND the README contents *now*, before the tempdir vanishes.
        folder = Path(kwargs["folder_path"])
        captures["uploaded_files"] = sorted(p.name for p in folder.iterdir())
        readme = folder / "README.md"
        captures["readme"] = readme.read_text() if readme.exists() else None

    mod.create_repo = create_repo
    mod.upload_folder = upload_folder
    mod.HfApi = type("HfApi", (), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


def test_push_adapter_writes_readme_and_uploads(tmp_path, monkeypatch):
    # ── Create a fake adapter folder (peft layout) ──
    adapter = tmp_path / "outputs" / "ai_llm"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"fake-weights")
    (adapter / "tokenizer.json").write_text("{}")

    captures: dict = {}
    _stub_hf_hub(monkeypatch, captures=captures)
    monkeypatch.setenv("HF_TOKEN", "hf_test_xxxxxx")

    from app.trainers.hub import push_adapter_to_hub
    url = push_adapter_to_hub(
        adapter_dir=str(adapter),
        repo_id="me/qwen-0.5b-test-sft",
        private=True,
        metadata={
            "method": "sft",
            "backend": "trl",
            "template": "qwen",
            "samples": 500,
            "final_loss": 1.95,
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        },
    )

    # ── Repo creation ──
    assert captures["create_repo"]["repo_id"] == "me/qwen-0.5b-test-sft"
    assert captures["create_repo"]["private"] is True
    assert captures["create_repo"]["exist_ok"] is True
    assert captures["create_repo"]["token"] == "hf_test_xxxxxx"

    # ── Upload contents include the original adapter files + README ──
    assert "adapter_config.json"      in captures["uploaded_files"]
    assert "adapter_model.safetensors" in captures["uploaded_files"]
    assert "tokenizer.json"           in captures["uploaded_files"]
    assert "README.md"                in captures["uploaded_files"]

    # ── README has the YAML front-matter with the base model ──
    readme = captures["readme"]
    assert readme is not None
    assert "base_model: Qwen/Qwen2.5-0.5B-Instruct" in readme
    assert "library_name: peft" in readme
    assert "1.9500" in readme   # final_loss formatted

    # ── Returned URL ──
    assert url == "https://huggingface.co/me/qwen-0.5b-test-sft"


def test_push_raises_when_no_token(tmp_path, monkeypatch):
    adapter = tmp_path / "out"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    from app.trainers.hub import push_adapter_to_hub
    with pytest.raises(RuntimeError, match="No HF token found"):
        push_adapter_to_hub(adapter_dir=str(adapter), repo_id="me/x")


def test_push_raises_when_adapter_dir_missing(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    from app.trainers.hub import push_adapter_to_hub
    with pytest.raises(FileNotFoundError):
        push_adapter_to_hub(adapter_dir="/tmp/nope/does/not/exist", repo_id="me/x")


def test_push_uses_explicit_token_argument(tmp_path, monkeypatch):
    adapter = tmp_path / "out"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")

    captures: dict = {}
    _stub_hf_hub(monkeypatch, captures=captures)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    from app.trainers.hub import push_adapter_to_hub
    push_adapter_to_hub(
        adapter_dir=str(adapter),
        repo_id="me/x",
        token="hf_explicit_arg_xxx",
    )
    assert captures["create_repo"]["token"] == "hf_explicit_arg_xxx"
    assert captures["upload_folder"]["token"] == "hf_explicit_arg_xxx"


def test_default_commit_message_includes_metrics(monkeypatch):
    from app.trainers.hub import _default_commit
    msg = _default_commit({"method": "sft", "samples": 500, "final_loss": 1.95})
    assert "SFT" in msg
    assert "500 samples" in msg
    assert "loss=1.950" in msg
    assert "ValonyLabs Studio" in msg
