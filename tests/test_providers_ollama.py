"""
Unit tests for app.providers.ollama — URL/auth resolution and a mocked chat call.
No network activity: requests.post is patched to return a canned response.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────
# URL and mode resolution
# ──────────────────────────────────────────────────────────────
def test_cloud_defaults_when_api_key_set(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-fake")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("VALONY_SYNTH_MODEL", raising=False)

    from app.providers.ollama import OllamaProvider
    p = OllamaProvider()
    assert p.is_cloud
    assert "ollama.com" in p.base_url
    assert p.base_url.endswith("/v1")
    assert p.model == "nemotron"       # cloud default
    assert p.api_key == "sk-fake"


def test_local_defaults_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("VALONY_SYNTH_MODEL", raising=False)

    from app.providers.ollama import OllamaProvider
    p = OllamaProvider(api_key=None)
    assert not p.is_cloud
    assert "localhost:11434" in p.base_url
    assert p.base_url.endswith("/v1")
    assert p.model == "llama3.1"       # local default
    assert p.api_key == "ollama"       # dummy accepted by local daemon


def test_explicit_host_override(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://my-gpu-box:11434")

    from app.providers.ollama import OllamaProvider
    p = OllamaProvider()
    assert p.base_url == "http://my-gpu-box:11434/v1"
    assert not p.is_cloud


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:72b")
    from app.providers.ollama import OllamaProvider
    p = OllamaProvider()
    assert p.model == "qwen2.5:72b"


def test_constructor_args_override_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
    monkeypatch.setenv("OLLAMA_MODEL", "env-model")
    from app.providers.ollama import OllamaProvider
    p = OllamaProvider(api_key="ctor-key", model="ctor-model", base_url="http://x:1234")
    assert p.api_key == "ctor-key"
    assert p.model == "ctor-model"
    assert p.base_url == "http://x:1234/v1"


# ──────────────────────────────────────────────────────────────
# chat() — mocked HTTP
# ──────────────────────────────────────────────────────────────
def _mock_response(status: int, payload: dict | str):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = payload if isinstance(payload, dict) else {}
    mock.text = payload if isinstance(payload, str) else ""
    return mock


def test_chat_returns_content_on_200(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    from app.providers.ollama import OllamaProvider

    payload = {
        "choices": [
            {"message": {"content": "hello from nemotron", "role": "assistant"}}
        ]
    }
    with patch("requests.post", return_value=_mock_response(200, payload)) as mocked:
        p = OllamaProvider()
        out = p.chat([{"role": "user", "content": "hi"}])
        assert out == "hello from nemotron"
        mocked.assert_called_once()
        call_kwargs = mocked.call_args.kwargs
        assert call_kwargs["json"]["model"] == "nemotron"
        assert call_kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]
        assert call_kwargs["headers"]["Authorization"] == "Bearer k"
        assert call_kwargs["json"]["stream"] is False


def test_chat_raises_provider_error_on_401(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "bad")
    from app.providers.base import ProviderError
    from app.providers.ollama import OllamaProvider

    with patch("requests.post", return_value=_mock_response(401, "Unauthorized")):
        p = OllamaProvider()
        with pytest.raises(ProviderError) as excinfo:
            p.chat([{"role": "user", "content": "hi"}])
        assert excinfo.value.status_code == 401


def test_stream_chat_parses_sse_deltas(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    from app.providers.ollama import OllamaProvider
    import json as _json

    lines = [
        f'data: {_json.dumps({"choices":[{"delta":{"content":"Hello "}}]})}',
        f'data: {_json.dumps({"choices":[{"delta":{"content":"world"}}]})}',
        f'data: {_json.dumps({"choices":[{"delta":{"content":"!"}}]})}',
        "data: [DONE]",
    ]

    stream_mock = MagicMock()
    stream_mock.status_code = 200
    stream_mock.iter_lines.return_value = iter(lines)
    stream_mock.__enter__ = lambda self: self
    stream_mock.__exit__ = lambda self, *a: None

    with patch("requests.post", return_value=stream_mock):
        p = OllamaProvider()
        deltas = list(p.stream_chat([{"role": "user", "content": "hi"}]))

    assert deltas == ["Hello ", "world", "!"]


def test_stream_chat_skips_non_data_lines(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    from app.providers.ollama import OllamaProvider
    import json as _json

    lines = [
        "",                                    # empty line
        ": heartbeat comment",                 # SSE comment
        f'data: {_json.dumps({"choices":[{"delta":{"content":"x"}}]})}',
        "data: not-valid-json",                # malformed — should be skipped
        f'data: {_json.dumps({"choices":[{"delta":{"content":"y"}}]})}',
        "data: [DONE]",
    ]
    stream_mock = MagicMock()
    stream_mock.status_code = 200
    stream_mock.iter_lines.return_value = iter(lines)
    stream_mock.__enter__ = lambda self: self
    stream_mock.__exit__ = lambda self, *a: None

    with patch("requests.post", return_value=stream_mock):
        p = OllamaProvider()
        deltas = list(p.stream_chat([{"role": "user", "content": "hi"}]))

    assert deltas == ["x", "y"]


def test_stream_chat_raises_on_4xx(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    from app.providers.base import ProviderError
    from app.providers.ollama import OllamaProvider

    stream_mock = MagicMock()
    stream_mock.status_code = 401
    stream_mock.text = "Unauthorized"
    stream_mock.iter_lines.return_value = iter([])
    stream_mock.__enter__ = lambda self: self
    stream_mock.__exit__ = lambda self, *a: None

    with patch("requests.post", return_value=stream_mock):
        p = OllamaProvider()
        with pytest.raises(ProviderError) as excinfo:
            list(p.stream_chat([{"role": "user", "content": "hi"}]))
        assert excinfo.value.status_code == 401


def test_chat_passes_response_format(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    from app.providers.ollama import OllamaProvider

    payload = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    with patch("requests.post", return_value=_mock_response(200, payload)) as mocked:
        p = OllamaProvider()
        p.chat(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=42,
        )
        body = mocked.call_args.kwargs["json"]
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == 0.1
        assert body["max_tokens"] == 42
