"""
ChatML — OpenAI's generic role-based template. Shared by Qwen2+, Yi, Nous Hermes,
and many community fine-tunes.

    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<|im_end|>
"""
from __future__ import annotations

from .base import ChatTemplate, _re


class ChatMLTemplate(ChatTemplate):
    name = "chatml"
    match_patterns = _re(
        r"chatml",
        r"Yi-\d+[A-Za-z]*-Chat",
        r"Nous-Hermes",
        r"Capybara",
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
