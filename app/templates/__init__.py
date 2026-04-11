"""
Chat-template registry.

Usage:
    from app.templates import get_template_for

    template = get_template_for("Qwen/Qwen2.5-7B-Instruct")
    text = template.format_sft(system="...", instruction="...", response="...")
    prompt = template.format_prompt_only(system="...", instruction="...")
    messages = template.as_messages(system="...", instruction="...", response="...")

Every concrete template (Qwen, Llama, Mistral, Gemma, Phi, DeepSeek, Alpaca,
ChatML, ShareGPT) implements `ChatTemplate` in `base.py`.
"""
from .base import ChatTemplate
from .registry import get_template_for, list_templates, register_template

__all__ = ["ChatTemplate", "get_template_for", "list_templates", "register_template"]
