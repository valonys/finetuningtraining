"""
Unit tests for app.inference.ollama_backend — HF→Ollama tag resolution,
domain routing semantics, and a mocked streaming generate() path.

No network. `requests.post` is patched with a fake SSE stream.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────
# HF → Ollama tag resolver
# ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("hf, expected", [
    ("Qwen/Qwen2.5-7B-Instruct",           "qwen2.5:7b"),
    ("Qwen/Qwen2.5-72B-Instruct",          "qwen2.5:72b"),
    ("meta-llama/Llama-3.2-1B-Instruct",   "llama3.2:1b"),
    ("meta-llama/Llama-3.3-70B-Instruct",  "llama3.3:70b"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral:7b"),
    ("google/gemma-2-9b-it",               "gemma2:9b"),
    ("microsoft/Phi-4",                    "phi4"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-r1:7b"),
    ("nvidia/Llama-3.1-Nemotron-70B-Instruct",  "nemotron"),
    # already a tag — pass through
    ("llama3.1",  "llama3.1"),
    ("nemotron",  "nemotron"),
    ("qwen2.5:7b", "qwen2.5:7b"),
    # unknown HF id — strip org + -instruct suffix
    ("acme/mystery-13B-instruct", "mystery-13b"),
])
def test_resolve_ollama_model(hf, expected):
    from app.inference.ollama_backend import resolve_ollama_model
    assert resolve_ollama_model(hf) == expected


# ──────────────────────────────────────────────────────────────
# Helpers: fake SSE stream
# ──────────────────────────────────────────────────────────────
def _sse_lines(deltas):
    """Build an SSE response body from a list of text deltas."""
    lines = []
    for d in deltas:
        chunk = {"choices": [{"delta": {"content": d}}]}
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")
    return lines


def _mock_stream_response(deltas, status=200):
    mock = MagicMock()
    mock.status_code = status
    mock.text = "" if status < 400 else "error"
    mock.iter_lines.return_value = iter(_sse_lines(deltas))
    # Support `with requests.post(...) as resp:`
    mock.__enter__ = lambda self: self
    mock.__exit__ = lambda self, *args: None
    return mock


def _make_backend(monkeypatch, base_model_id="llama3.1"):
    monkeypatch.setenv("OLLAMA_API_KEY", "fake")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    from app.inference.ollama_backend import OllamaInferenceBackend
    backend = OllamaInferenceBackend(
        base_model_id,
        profile=None,
        lora_registry={},
    )
    return backend


# ──────────────────────────────────────────────────────────────
# Backend behaviour
# ──────────────────────────────────────────────────────────────
def test_hf_id_passes_through_with_warning(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("OLLAMA_API_KEY", "fake")
    from app.inference.ollama_backend import OllamaInferenceBackend

    with caplog.at_level(logging.WARNING):
        b = OllamaInferenceBackend(
            "Qwen/Qwen2.5-7B-Instruct",
            profile=None,
            lora_registry={},
        )
    # The backend stored the mapped tag, not the HF id
    assert b.model_id == "qwen2.5:7b"
    # And warned about the mapping
    assert any("Ollama backend received HF-style id" in r.message for r in caplog.records)


def test_register_adapter_maps_domain_to_model_tag(monkeypatch):
    b = _make_backend(monkeypatch)
    b.register_adapter("customer_grasps", "qwen2.5:7b")
    b.register_adapter("ai_llm", "nemotron")
    assert b.lora_registry["customer_grasps"] == "qwen2.5:7b"
    assert b.lora_registry["ai_llm"] == "nemotron"


def test_register_adapter_resolves_hf_id_to_tag(monkeypatch):
    b = _make_backend(monkeypatch)
    b.register_adapter("customer_grasps", "Qwen/Qwen2.5-7B-Instruct")
    assert b.lora_registry["customer_grasps"] == "qwen2.5:7b"


def test_generate_uses_base_model_by_default(monkeypatch):
    from app.inference.manager import GenerationRequest
    b = _make_backend(monkeypatch, base_model_id="llama3.1")

    captured = {}

    def _capture_post(url, json=None, headers=None, stream=None, timeout=None):
        captured["body"] = json
        captured["url"] = url
        return _mock_stream_response(["Hello ", "world", "!"])

    with patch("requests.post", side_effect=_capture_post):
        req = GenerationRequest(prompt="hi", domain_name="base", max_new_tokens=16)
        resp = b.generate(req)

    assert resp.text == "Hello world!"
    assert resp.backend == "ollama"
    assert resp.model == "llama3.1"          # base default
    assert resp.tokens_generated == 3
    assert resp.ttft_ms > 0
    assert captured["body"]["model"] == "llama3.1"
    assert captured["body"]["stream"] is True


def test_generate_routes_registered_domain_to_mapped_model(monkeypatch):
    from app.inference.manager import GenerationRequest
    b = _make_backend(monkeypatch, base_model_id="llama3.1")
    b.register_adapter("ai_llm", "nemotron")

    captured = {}

    def _capture_post(url, json=None, **kw):
        captured["model"] = json["model"]
        return _mock_stream_response(["OK"])

    with patch("requests.post", side_effect=_capture_post):
        req = GenerationRequest(prompt="hi", domain_name="ai_llm")
        resp = b.generate(req)

    assert captured["model"] == "nemotron"
    assert resp.model == "nemotron"
    # The backend should have restored the provider model after the call
    assert b._provider.model == "llama3.1"


def test_generate_unregistered_domain_falls_back_to_base(monkeypatch):
    from app.inference.manager import GenerationRequest
    b = _make_backend(monkeypatch, base_model_id="llama3.1")

    captured = {}
    def _cp(url, json=None, **kw):
        captured["model"] = json["model"]
        return _mock_stream_response(["X"])

    with patch("requests.post", side_effect=_cp):
        req = GenerationRequest(prompt="hi", domain_name="never_registered")
        resp = b.generate(req)

    assert captured["model"] == "llama3.1"
    assert resp.model == "llama3.1"


def test_generate_measures_ttft(monkeypatch):
    from app.inference.manager import GenerationRequest
    b = _make_backend(monkeypatch)

    with patch("requests.post", return_value=_mock_stream_response(["one", "two", "three"])):
        req = GenerationRequest(prompt="hi", domain_name="base")
        resp = b.generate(req)

    assert resp.tokens_generated == 3
    assert resp.ttft_ms > 0
    assert resp.latency_ms >= resp.ttft_ms
    assert resp.tokens_per_second >= 0
