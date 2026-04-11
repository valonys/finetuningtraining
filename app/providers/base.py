"""
Base interface every LLM synthesis provider implements.

One method: `chat(messages, ...)`. Returns a plain string (the assistant's
message content). All providers are OpenAI-compatible at the wire level;
the only differences are base URL, authentication, and default model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ProviderError(Exception):
    status_code: int
    body: str

    def __str__(self) -> str:
        return f"Provider error {self.status_code}: {self.body[:200]}"


class SynthProvider(ABC):
    """Abstract base class for synth providers."""

    name: str = "base"
    base_url: str = ""
    model: str = ""

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
    ) -> str:
        """
        Send an OpenAI-compatible chat completion request.

        Args:
            messages: list of `{"role": ..., "content": ...}` dicts.
            temperature: sampling temperature.
            max_tokens: completion length cap.
            response_format: optional OpenAI-style structured output hint,
                e.g. `{"type": "json_object"}`. Providers that don't
                support it silently ignore.
            timeout: HTTP timeout in seconds.

        Returns:
            The assistant message content as a plain string.

        Raises:
            ProviderError: on non-2xx HTTP or malformed response body.
        """
