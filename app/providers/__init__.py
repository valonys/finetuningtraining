"""
app/providers/
──────────────
LLM provider clients for synthetic data generation.

This is deliberately separate from `app/inference/`:

  * `app/inference/`  — serves **trained** adapters at low latency (vLLM,
                         SGLang, MLX, llama.cpp, HF). Called on every chat
                         turn. Local to the machine running the Studio.
  * `app/providers/`  — calls **remote** LLMs as contractors to help build
                         datasets (Q/A pair synth, DPO contrastive pairs,
                         LLM-as-judge eval). Called once per dataset build,
                         network-bound, cost-sensitive. Ollama Cloud +
                         Nemotron is the recommended default for cheap
                         high-quality synth.

Every provider implements a single method:

    chat(messages, *, temperature, max_tokens, response_format) -> str

and returns the assistant message content as a plain string.

Env-var driven selection (in priority order):

    1. VALONY_SYNTH_PROVIDER=ollama | openai | anthropic | openrouter | rule_based
       (explicit override)
    2. OLLAMA_API_KEY set                      → Ollama Cloud (default: nemotron)
    3. OLLAMA_HOST set (no API key)            → local Ollama daemon
    4. OPENAI_API_KEY set                      → OpenAI (default: gpt-4o-mini)
    5. VALONY_SYNTH_BASE_URL + VALONY_SYNTH_MODEL set → generic OpenAI-compat
    6. Nothing set                             → None (callers fall back to rule-based)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .base import SynthProvider, ChatMessage
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)

__all__ = [
    "SynthProvider",
    "ChatMessage",
    "OllamaProvider",
    "OpenAICompatProvider",
    "get_synth_provider",
    "describe_active_provider",
]


def get_synth_provider() -> Optional[SynthProvider]:
    """
    Resolve the active synth provider from environment variables.

    Returns None if no credentials are configured — the caller should fall
    back to rule-based synthesis.
    """
    override = (os.environ.get("VALONY_SYNTH_PROVIDER") or "").lower().strip()

    if override == "rule_based" or override == "none":
        return None

    if override == "ollama":
        return OllamaProvider()

    if override == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            logger.warning("VALONY_SYNTH_PROVIDER=openai but OPENAI_API_KEY is unset")
            return None
        return OpenAICompatProvider(
            api_key=key,
            base_url="https://api.openai.com/v1",
            model=os.environ.get("VALONY_SYNTH_MODEL", "gpt-4o-mini"),
            name="openai",
        )

    if override == "anthropic":
        # Anthropic's API is NOT OpenAI-compatible — we'd need a separate
        # client. For now, fall through to the generic path which assumes
        # an OpenAI-compatible endpoint.
        logger.warning("Anthropic provider isn't directly supported — "
                       "point VALONY_SYNTH_BASE_URL at an OpenAI-compat gateway "
                       "(e.g., LiteLLM / OpenRouter)")
        return None

    if override == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("VALONY_SYNTH_API_KEY")
        if not key:
            logger.warning("VALONY_SYNTH_PROVIDER=openrouter but OPENROUTER_API_KEY is unset")
            return None
        return OpenAICompatProvider(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            model=os.environ.get("VALONY_SYNTH_MODEL", "meta-llama/llama-3.1-70b-instruct"),
            name="openrouter",
        )

    # ── Auto-detect in priority order ─────────────────────────
    if os.environ.get("OLLAMA_API_KEY"):
        return OllamaProvider()                  # cloud mode

    if os.environ.get("OLLAMA_HOST"):
        return OllamaProvider(api_key=None)      # local mode

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
            model=os.environ.get("VALONY_SYNTH_MODEL", "gpt-4o-mini"),
            name="openai",
        )

    if os.environ.get("VALONY_SYNTH_BASE_URL") and os.environ.get("VALONY_SYNTH_MODEL"):
        return OpenAICompatProvider(
            api_key=os.environ.get("VALONY_SYNTH_API_KEY", "sk-local"),
            base_url=os.environ["VALONY_SYNTH_BASE_URL"],
            model=os.environ["VALONY_SYNTH_MODEL"],
            name="generic",
        )

    return None


def describe_active_provider() -> dict:
    """Return a dict describing the currently-selected provider (for /healthz)."""
    p = get_synth_provider()
    if p is None:
        return {"provider": "rule_based", "model": None, "base_url": None}
    return {
        "provider": p.name,
        "model": p.model,
        "base_url": p.base_url,
        "is_cloud": getattr(p, "is_cloud", False),
    }
