"""
Alpaca — the original Stanford Alpaca instruction template. Still widely used
for community SFT datasets with `{instruction, input, output}` columns.

    Below is an instruction...
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:
    {response}
"""
from __future__ import annotations

from .base import ChatTemplate, _re


_ALPACA_HEADER = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)


class AlpacaChatTemplate(ChatTemplate):
    name = "alpaca"
    match_patterns = _re(
        r"alpaca",
        r"vicuna",                    # Vicuna v1.1 uses a similar instruction-response structure
    )
    response_template = "### Response:\n"

    def format_sft(self, *, system: str, instruction: str, response: str) -> str:
        header = system or _ALPACA_HEADER
        return (
            f"{header}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )

    def format_prompt_only(self, *, system: str, instruction: str) -> str:
        header = system or _ALPACA_HEADER
        return (
            f"{header}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
