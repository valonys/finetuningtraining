"""
Qwen chat templates (Qwen2, Qwen2.5, Qwen3).

Qwen uses ChatML-style delimiters (`<|im_start|>`, `<|im_end|>`) but recent
Qwen3 Instruct models also support "thinking" blocks. For SFT we stay on
the non-thinking path (matching `enable_thinking=False`).
"""
from __future__ import annotations

from .base import ChatTemplate, _re


class QwenChatTemplate(ChatTemplate):
    name = "qwen"
    match_patterns = _re(
        r"Qwen/?Qwen[23](?:\.\d)?",
        r"unsloth/[Qq]wen[23]",
        r"Qwen[23][A-Za-z\d.\-]*-(?:Instruct|Chat)",
        r"Qwen[23]\.\d[-\w]*",
        # Catches Qwen2.5-7B, Qwen3-8B-Instruct, etc.
        r"Qwen",
    )
    response_template = "<|im_start|>assistant\n"

    def format_sft(self, *, system: str, instruction: str, response: str) -> str:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )

    def format_prompt_only(self, *, system: str, instruction: str) -> str:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
