"""
DeepSeek chat template (DeepSeek-V2, V3, R1 and distilled variants).

DeepSeek-V2+ uses a ChatML-ish format with special `<｜User｜>` and
`<｜Assistant｜>` (fullwidth pipes) markers plus `<｜begin▁of▁sentence｜>`
sentence boundaries. We generate the ASCII variant; the tokeniser handles the
fullwidth versions when `apply_chat_template` is used at inference time.
"""
from __future__ import annotations

from .base import ChatTemplate, _re


class DeepSeekChatTemplate(ChatTemplate):
    name = "deepseek"
    match_patterns = _re(
        r"deepseek-ai/DeepSeek",
        r"DeepSeek-R1",
        r"DeepSeek-V[23]",
        r"deepseek-llm",
        r"deepseek-coder",
        r"deepseek-math",
    )
    response_template = "<｜Assistant｜>"

    def format_sft(self, *, system: str, instruction: str, response: str) -> str:
        sys_block = f"{system}\n\n" if system else ""
        return (
            "<｜begin▁of▁sentence｜>"
            f"{sys_block}"
            f"<｜User｜>{instruction}"
            f"<｜Assistant｜>{response}<｜end▁of▁sentence｜>"
        )

    def format_prompt_only(self, *, system: str, instruction: str) -> str:
        sys_block = f"{system}\n\n" if system else ""
        return (
            "<｜begin▁of▁sentence｜>"
            f"{sys_block}"
            f"<｜User｜>{instruction}"
            f"<｜Assistant｜>"
        )
