"""
Phi chat template (Phi-3, Phi-4).

Phi-3 uses `<|system|>`, `<|user|>`, `<|assistant|>` markers each followed by a
newline, with `<|end|>` terminating each turn. Phi-4 introduced updated role
markers with similar structure — this template matches both.
"""
from __future__ import annotations

from .base import ChatTemplate, _re


class PhiChatTemplate(ChatTemplate):
    name = "phi"
    match_patterns = _re(
        r"microsoft/Phi-3",
        r"microsoft/Phi-4",
        r"unsloth/phi-3",
        r"unsloth/phi-4",
        r"Phi-3\.\d-",
        r"Phi-4",
    )
    response_template = "<|assistant|>\n"

    def format_sft(self, *, system: str, instruction: str, response: str) -> str:
        sys_block = f"<|system|>\n{system}<|end|>\n" if system else ""
        return (
            f"{sys_block}"
            f"<|user|>\n{instruction}<|end|>\n"
            f"<|assistant|>\n{response}<|end|>"
        )

    def format_prompt_only(self, *, system: str, instruction: str) -> str:
        sys_block = f"<|system|>\n{system}<|end|>\n" if system else ""
        return (
            f"{sys_block}"
            f"<|user|>\n{instruction}<|end|>\n"
            f"<|assistant|>\n"
        )
