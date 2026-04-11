"""
ShareGPT — a conversation format used by many community chat datasets.
Typical row layout:

    {"conversations": [{"from": "human", "value": "..."},
                       {"from": "gpt",   "value": "..."}]}

For SFT we serialise into ChatML (same special tokens as ChatML/Qwen) to keep
training compatible with the widest set of base models. This template is only
returned when the user explicitly asks for it via `sharegpt`.
"""
from __future__ import annotations

from .base import ChatTemplate, _re


class ShareGPTTemplate(ChatTemplate):
    name = "sharegpt"
    match_patterns = _re(
        r"sharegpt",
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
