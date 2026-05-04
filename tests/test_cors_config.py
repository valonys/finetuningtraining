"""
Unit tests for app.security.cors.resolve_cors_origins.

The resolver replaces the wildcard ``allow_origins=["*"]`` from
pre-S06. Tests cover the env-var parsing, the dev-default fallback,
and the whitespace / empty-entry handling.
"""
from __future__ import annotations

import pytest

from app.security import resolve_cors_origins


_DEV_DEFAULT = ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_unset_env_returns_dev_defaults(monkeypatch):
    monkeypatch.delenv("VALONY_CORS_ORIGINS", raising=False)
    assert resolve_cors_origins() == _DEV_DEFAULT


def test_empty_env_returns_dev_defaults(monkeypatch):
    monkeypatch.setenv("VALONY_CORS_ORIGINS", "")
    assert resolve_cors_origins() == _DEV_DEFAULT


def test_whitespace_only_env_returns_dev_defaults(monkeypatch):
    monkeypatch.setenv("VALONY_CORS_ORIGINS", "   ")
    assert resolve_cors_origins() == _DEV_DEFAULT


def test_single_origin(monkeypatch):
    monkeypatch.setenv("VALONY_CORS_ORIGINS", "https://studio.example.com")
    assert resolve_cors_origins() == ["https://studio.example.com"]


def test_multiple_origins(monkeypatch):
    monkeypatch.setenv(
        "VALONY_CORS_ORIGINS",
        "https://studio.example.com,https://staging.example.com",
    )
    assert resolve_cors_origins() == [
        "https://studio.example.com",
        "https://staging.example.com",
    ]


def test_whitespace_around_entries_trimmed(monkeypatch):
    monkeypatch.setenv(
        "VALONY_CORS_ORIGINS",
        " https://a.example.com , https://b.example.com ",
    )
    assert resolve_cors_origins() == [
        "https://a.example.com",
        "https://b.example.com",
    ]


def test_empty_entries_dropped(monkeypatch):
    monkeypatch.setenv(
        "VALONY_CORS_ORIGINS",
        "https://a.example.com,,https://b.example.com,",
    )
    assert resolve_cors_origins() == [
        "https://a.example.com",
        "https://b.example.com",
    ]


def test_all_empty_after_trim_falls_back_to_default(monkeypatch):
    """If every entry is empty/whitespace, fall back instead of
    returning an empty list (which would block ALL origins, breaking
    dev silently)."""
    monkeypatch.setenv("VALONY_CORS_ORIGINS", " , , ,")
    assert resolve_cors_origins() == _DEV_DEFAULT


def test_dev_defaults_are_a_fresh_list(monkeypatch):
    """Mutating the returned list must not poison the next call."""
    monkeypatch.delenv("VALONY_CORS_ORIGINS", raising=False)
    first = resolve_cors_origins()
    first.append("https://injected.example.com")
    second = resolve_cors_origins()
    assert "https://injected.example.com" not in second
