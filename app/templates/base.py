"""
ChatTemplate — the interface every concrete template implements.

Design:
  - `format_sft(system, instruction, response)` returns the **full** training
    string, suitable for the `text` field of a TRL SFT dataset.
  - `format_prompt_only(system, instruction)` returns just the prompt, used
    for DPO/GRPO where the response is separate.
  - `as_messages(...)` returns the canonical `[{"role": ..., "content": ...}]`
    list, which lets downstream code call `tokenizer.apply_chat_template`.
  - `response_template` — the exact string that marks the boundary between
    prompt and response. Used by `DataCollatorForCompletionOnlyLM` so that
    the loss is computed **only** on response tokens.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Pattern
import re


class ChatTemplate(ABC):
    """Abstract base for chat templates. Not a dataclass — all attributes are ClassVars."""
    name: ClassVar[str] = "base"
    # Regex patterns that match model IDs this template applies to.
    # Subclasses override this list.
    match_patterns: ClassVar[list] = []

    @abstractmethod
    def format_sft(self, *, system: str, instruction: str, response: str) -> str: ...

    @abstractmethod
    def format_prompt_only(self, *, system: str, instruction: str) -> str: ...

    def as_messages(self, *, system: str, instruction: str, response: str | None = None) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": instruction})
        if response is not None:
            msgs.append({"role": "assistant", "content": response})
        return msgs

    # Response template: used by DataCollatorForCompletionOnlyLM.
    # If a subclass sets this, the trainer will mask prompt tokens from the loss.
    response_template: ClassVar[str] = ""

    # Optional: the canonical HF `chat_template` Jinja string for this template.
    # When present, we can call `tokenizer.apply_chat_template(..., chat_template=...)`
    # on base models that ship with a broken or missing template.
    hf_chat_template: ClassVar[str] = ""

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return any(p.search(model_id) for p in cls.match_patterns)


def _re(*patterns: str) -> list[Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]
