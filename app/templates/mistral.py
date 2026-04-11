"""
Mistral / Mixtral chat template.

Mistral Instruct v0.1 / v0.2 / v0.3 and Mixtral use `[INST] ... [/INST]` but
**do not** support a `<<SYS>>` block — the system prompt is prepended to the
first user turn.

Newer Mistral models (NeMo 12B, Small 22B, Large) use Mistral's "v7 Tekken"
template which is structurally the same for single-turn.
"""
from __future__ import annotations

from .base import ChatTemplate, _re


class MistralChatTemplate(ChatTemplate):
    name = "mistral"
    match_patterns = _re(
        r"mistralai/Mistral-",
        r"mistralai/Mixtral-",
        r"unsloth/mistral",
        r"unsloth/Mixtral",
        r"[Mm]istral-7B-Instruct",
        r"[Mm]ixtral-8x",
        r"Mistral-Nemo",
        r"Mistral-Small",
        r"Mistral-Large",
    )
    response_template = "[/INST]"

    def format_sft(self, *, system: str, instruction: str, response: str) -> str:
        # Mistral does not use <<SYS>>; prepend the system prompt inside [INST].
        user_content = f"{system}\n\n{instruction}" if system else instruction
        return f"<s>[INST] {user_content} [/INST] {response}</s>"

    def format_prompt_only(self, *, system: str, instruction: str) -> str:
        user_content = f"{system}\n\n{instruction}" if system else instruction
        return f"<s>[INST] {user_content} [/INST]"
