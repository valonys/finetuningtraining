"""
Gemma 1 / 2 / 3 chat template.

Gemma uses `<start_of_turn>` / `<end_of_turn>` delimiters with role markers.
It does **not** have a dedicated system slot — system prompts are prepended to
the first user turn.
"""
from __future__ import annotations

from .base import ChatTemplate, _re


class GemmaChatTemplate(ChatTemplate):
    name = "gemma"
    match_patterns = _re(
        r"google/gemma",
        r"unsloth/gemma",
        r"gemma-[123]",
        r"gemma-\d+b",
    )
    response_template = "<start_of_turn>model\n"

    def format_sft(self, *, system: str, instruction: str, response: str) -> str:
        user_content = f"{system}\n\n{instruction}" if system else instruction
        return (
            f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
            f"<start_of_turn>model\n{response}<end_of_turn>"
        )

    def format_prompt_only(self, *, system: str, instruction: str) -> str:
        user_content = f"{system}\n\n{instruction}" if system else instruction
        return (
            f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
