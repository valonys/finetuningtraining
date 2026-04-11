"""
Generic OpenAI-compatible provider.

Works with anything that speaks the OpenAI `/v1/chat/completions` API:
  * OpenAI directly (api.openai.com)
  * OpenRouter (openrouter.ai)
  * Together AI (api.together.xyz)
  * Groq (api.groq.com)
  * Fireworks (api.fireworks.ai)
  * LiteLLM gateway
  * Any vLLM / SGLang server running `--served-model-name ...`

For Ollama specifically, use `OllamaProvider` — it has smarter defaults
for the local vs cloud split and knows about Nemotron.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .base import ChatMessage, ProviderError, SynthProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(SynthProvider):
    """Thin OpenAI-compatible chat-completion client."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        name: str = "openai_compat",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.model = model
        self.name = name
        logger.info(f"🔌 {name} provider → {self.base_url} | model={self.model}")

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
    ) -> str:
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("`requests` required — pip install requests") from e

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, json=body, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            raise ProviderError(status_code=resp.status_code, body=resp.text)

        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, json.JSONDecodeError, TypeError) as e:
            raise ProviderError(
                status_code=resp.status_code,
                body=f"Malformed response from {self.name}: {e}",
            )
