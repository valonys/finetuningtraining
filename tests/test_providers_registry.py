"""
Unit tests for app.providers.get_synth_provider — env-based routing.
"""
from __future__ import annotations


def _clear_all(monkeypatch):
    for k in (
        "VALONY_SYNTH_PROVIDER",
        "OLLAMA_API_KEY", "OLLAMA_HOST", "OLLAMA_MODEL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "VALONY_SYNTH_BASE_URL", "VALONY_SYNTH_MODEL", "VALONY_SYNTH_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


def test_none_when_nothing_configured(monkeypatch):
    _clear_all(monkeypatch)
    from app.providers import get_synth_provider
    assert get_synth_provider() is None


def test_ollama_cloud_auto_detect(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-x")
    from app.providers import get_synth_provider
    p = get_synth_provider()
    assert p is not None
    assert p.name == "ollama"
    assert getattr(p, "is_cloud", False)
    assert "ollama.com" in p.base_url


def test_ollama_local_auto_detect(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    from app.providers import get_synth_provider
    p = get_synth_provider()
    assert p is not None
    assert p.name == "ollama"
    assert not getattr(p, "is_cloud", False)


def test_openai_auto_detect(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    from app.providers import get_synth_provider
    p = get_synth_provider()
    assert p is not None
    assert p.name == "openai"
    assert "api.openai.com" in p.base_url
    assert p.model == "gpt-4o-mini"


def test_generic_openai_compat_auto_detect(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("VALONY_SYNTH_BASE_URL", "https://api.together.xyz")
    monkeypatch.setenv("VALONY_SYNTH_MODEL", "meta-llama/Llama-3.1-70B")
    monkeypatch.setenv("VALONY_SYNTH_API_KEY", "tgp_fake")
    from app.providers import get_synth_provider
    p = get_synth_provider()
    assert p is not None
    assert p.name == "generic"
    assert "together.xyz" in p.base_url
    assert p.model == "meta-llama/Llama-3.1-70B"


def test_explicit_override_rule_based(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-x")
    monkeypatch.setenv("VALONY_SYNTH_PROVIDER", "rule_based")
    from app.providers import get_synth_provider
    assert get_synth_provider() is None


def test_explicit_override_ollama(monkeypatch):
    _clear_all(monkeypatch)
    # No env credentials, but explicit override should still yield an
    # Ollama provider (local mode, since no API key is set).
    monkeypatch.setenv("VALONY_SYNTH_PROVIDER", "ollama")
    from app.providers import get_synth_provider
    p = get_synth_provider()
    assert p is not None
    assert p.name == "ollama"


def test_ollama_cloud_takes_priority_over_openai(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-ol")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
    from app.providers import get_synth_provider
    p = get_synth_provider()
    assert p.name == "ollama"


def test_describe_active_provider_when_none(monkeypatch):
    _clear_all(monkeypatch)
    from app.providers import describe_active_provider
    info = describe_active_provider()
    assert info["provider"] == "rule_based"
    assert info["model"] is None


def test_describe_active_provider_when_ollama_cloud(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    from app.providers import describe_active_provider
    info = describe_active_provider()
    assert info["provider"] == "ollama"
    assert info["model"] == "nemotron"
    assert info["is_cloud"] is True
