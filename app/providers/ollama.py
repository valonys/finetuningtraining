"""
Ollama provider — local daemon AND Ollama Cloud (Turbo).

Ollama exposes an OpenAI-compatible endpoint at `{base}/v1/chat/completions`
on both the local daemon and the hosted cloud service, so one client works
for both. The only differences are:

  * Local:  base = http://localhost:11434    (auth is a no-op dummy token)
  * Cloud:  base = https://ollama.com         (auth is the API key from your
                                               ollama.com account)

Recommended models for dataset synthesis:

    nemotron:latest               Llama-3.1-Nemotron-70B — SOTA for synth,
                                   default when running on Cloud
    llama3.3:70b                  General purpose strong reasoning
    qwen2.5:72b                   Very strong reasoning, multilingual
    deepseek-v2.5                 Strong for code and technical content
    mixtral:8x22b                 Good cost/quality for bulk generation

For *cheap high-volume* synthesis on Ollama Cloud, Nemotron is the sweet
spot — dramatically better than any 7B local model and cheaper than
frontier APIs.

Configuration:

    OLLAMA_API_KEY=...                     # set → cloud mode (default URL)
    OLLAMA_HOST=http://my-ollama:11434     # optional base-URL override
    OLLAMA_MODEL=nemotron                  # optional model override

Env vars are resolved in the order: constructor args → env vars → defaults.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import ChatMessage, ProviderError, SynthProvider

logger = logging.getLogger(__name__)


class OllamaProvider(SynthProvider):
    """Client for Ollama (local daemon or Ollama Cloud)."""

    name = "ollama"

    DEFAULT_LOCAL_BASE = "http://localhost:11434"
    DEFAULT_CLOUD_BASE = "https://ollama.com"
    DEFAULT_CLOUD_MODEL = "nemotron"     # Llama-3.1-Nemotron-70B
    DEFAULT_LOCAL_MODEL = "llama3.1"     # reasonable local default

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # Resolve auth: explicit arg > env > None
        if api_key is None:
            api_key = os.environ.get("OLLAMA_API_KEY")

        # Resolve base URL: explicit arg > env > cloud-if-keyed > local
        if base_url is None:
            base_url = os.environ.get("OLLAMA_HOST")
        if base_url is None:
            base_url = self.DEFAULT_CLOUD_BASE if api_key else self.DEFAULT_LOCAL_BASE

        # Normalise — we always hit /v1/chat/completions
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        self.base_url = base_url

        # Local Ollama accepts any non-empty auth token; cloud requires a real one
        self.api_key = api_key or "ollama"
        self.is_cloud = "ollama.com" in self.base_url or "ollama.ai" in self.base_url

        # Resolve model: explicit arg > env > default (cloud vs local)
        if model is None:
            model = (
                os.environ.get("OLLAMA_MODEL")
                or os.environ.get("VALONY_SYNTH_MODEL")
                or (self.DEFAULT_CLOUD_MODEL if self.is_cloud else self.DEFAULT_LOCAL_MODEL)
            )
        self.model = model

        logger.info(
            f"🦙 Ollama provider → {self.base_url} | model={self.model} | "
            f"mode={'cloud' if self.is_cloud else 'local'}"
        )

    # ──────────────────────────────────────────────────────────
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
            "stream": False,
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
                body=f"Malformed Ollama response: {e} | raw: {resp.text[:500]}",
            )

    def stream_chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        timeout: int = 300,
    ):
        """
        Stream tokens from a chat completion as they arrive.

        Yields plain string deltas (not SSE frames). The inference backend
        wraps this in a generator that also measures TTFT on the first
        non-empty delta.

        Uses the OpenAI-compatible `/v1/chat/completions?stream=true` SSE
        endpoint, which Ollama supports on both local (`/v1`) and cloud.
        """
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("`requests` required — pip install requests") from e

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if stop:
            body["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        url = f"{self.base_url}/chat/completions"
        with requests.post(url, json=body, headers=headers, stream=True, timeout=timeout) as resp:
            if resp.status_code >= 400:
                raise ProviderError(status_code=resp.status_code, body=resp.text)

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data:"):
                    continue
                chunk = raw_line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content

    def health(self) -> dict:
        """Light-touch health check. Returns {'ok': bool, ...}."""
        try:
            import requests
            # Use the native Ollama /api/tags endpoint for a zero-token probe
            base_root = self.base_url.rsplit("/v1", 1)[0]
            r = requests.get(
                f"{base_root}/api/tags",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            return {
                "ok": r.status_code == 200,
                "status": r.status_code,
                "base_url": self.base_url,
                "model": self.model,
                "is_cloud": self.is_cloud,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "base_url": self.base_url}
