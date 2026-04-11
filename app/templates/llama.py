"""
Llama chat templates (Llama 2, Llama 3, Llama 3.1/3.2/3.3).

Llama 2 uses the `<s>[INST] ... [/INST]` format with `<<SYS>>` blocks.
Llama 3+ uses `<|begin_of_text|><|start_header_id|>...<|end_header_id|>` with
`<|eot_id|>` end-of-turn markers.
"""
from __future__ import annotations

from .base import ChatTemplate, _re


class Llama2ChatTemplate(ChatTemplate):
    name = "llama2"
    match_patterns = _re(
        r"meta-llama/Llama-2-",
        r"unsloth/llama-2-",
        r"Llama-2-\d+b",
    )
    response_template = "[/INST]"

    def format_sft(self, *, system: str, instruction: str, response: str) -> str:
        sys_block = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
        return f"<s>[INST] {sys_block}{instruction} [/INST] {response} </s>"

    def format_prompt_only(self, *, system: str, instruction: str) -> str:
        sys_block = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
        return f"<s>[INST] {sys_block}{instruction} [/INST]"


class Llama3ChatTemplate(ChatTemplate):
    name = "llama3"
    match_patterns = _re(
        r"meta-llama/Llama-3",
        r"meta-llama/Meta-Llama-3",
        r"unsloth/llama-3",
        r"unsloth/Meta-Llama-3",
        r"Llama-3(?:\.\d)?-\d+b",
        r"Llama-3\.2",
        r"Llama-3\.3",
    )
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def format_sft(self, *, system: str, instruction: str, response: str) -> str:
        sys_block = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            if system else ""
        )
        return (
            "<|begin_of_text|>"
            f"{sys_block}"
            f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        )

    def format_prompt_only(self, *, system: str, instruction: str) -> str:
        sys_block = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            if system else ""
        )
        return (
            "<|begin_of_text|>"
            f"{sys_block}"
            f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
