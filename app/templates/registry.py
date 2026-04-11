"""
Template registry — the single source of truth for "given a model ID, which
chat template should I use?".

Resolution order:
  1. Explicit `template=...` kwarg
  2. Model family pattern match (Qwen → QwenChatTemplate, etc.)
  3. Try to read the tokenizer's built-in `chat_template` attribute
     and wrap it in a HFAutoTemplate (fallback)
  4. Default: ChatML

Registering a new template is a one-liner:

    from app.templates import register_template
    register_template(MyCustomTemplate)
"""
from __future__ import annotations

import logging
from typing import Optional

from .alpaca import AlpacaChatTemplate
from .base import ChatTemplate
from .chatml import ChatMLTemplate
from .deepseek import DeepSeekChatTemplate
from .gemma import GemmaChatTemplate
from .llama import Llama2ChatTemplate, Llama3ChatTemplate
from .mistral import MistralChatTemplate
from .phi import PhiChatTemplate
from .qwen import QwenChatTemplate
from .sharegpt import ShareGPTTemplate

logger = logging.getLogger(__name__)

# ── Ordered registry — specificity matters! ────────────────────
# DeepSeek distilled variants include "Qwen"/"Llama" in their IDs, so the
# DeepSeek matcher must run before Qwen/Llama. Likewise, Llama 3 before Llama 2.
_TEMPLATES: list[type[ChatTemplate]] = [
    DeepSeekChatTemplate,     # first — catches DeepSeek-R1-Distill-Qwen/Llama
    QwenChatTemplate,
    Llama3ChatTemplate,
    Llama2ChatTemplate,
    MistralChatTemplate,
    GemmaChatTemplate,
    PhiChatTemplate,
    AlpacaChatTemplate,
    ShareGPTTemplate,
    ChatMLTemplate,           # fallback for anything using ChatML
]


def register_template(cls: type[ChatTemplate]) -> None:
    """Insert a new template class at the **front** of the resolver list."""
    _TEMPLATES.insert(0, cls)


def list_templates() -> list[str]:
    return [cls.name for cls in _TEMPLATES]


def get_template_for(model_id: str, template: Optional[str] = None) -> ChatTemplate:
    """
    Resolve a ChatTemplate for a model.

    Args:
        model_id: e.g. "Qwen/Qwen2.5-7B-Instruct"
        template: override — one of "qwen" | "llama3" | "mistral" | ...
    """
    if template:
        for cls in _TEMPLATES:
            if cls.name == template:
                return cls()
        logger.warning(f"⚠️  Requested template '{template}' not registered — falling back to model match")

    for cls in _TEMPLATES:
        if cls.matches(model_id):
            logger.debug(f"📎 Template match: {model_id} → {cls.name}")
            return cls()

    # Last-chance: try to read the tokenizer's own chat_template
    auto = _try_hf_tokenizer_template(model_id)
    if auto is not None:
        return auto

    logger.warning(f"⚠️  No template matched {model_id} — defaulting to ChatML")
    return ChatMLTemplate()


def _try_hf_tokenizer_template(model_id: str) -> Optional[ChatTemplate]:
    """Wrap a HuggingFace tokenizer's native chat_template in our interface."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if not getattr(tok, "chat_template", None):
            return None
    except Exception:
        return None
    logger.info(f"🪄 Using HF tokenizer's native chat_template for {model_id}")
    return HFAutoTemplate(tok)


class HFAutoTemplate(ChatTemplate):
    """Wrap `tokenizer.apply_chat_template` into the ChatTemplate interface."""
    name = "hf_auto"

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.response_template = ""  # unknown — loss masking disabled

    def format_sft(self, *, system: str, instruction: str, response: str) -> str:
        messages = [
            *( [{"role": "system", "content": system}] if system else [] ),
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def format_prompt_only(self, *, system: str, instruction: str) -> str:
        messages = [
            *( [{"role": "system", "content": system}] if system else [] ),
            {"role": "user", "content": instruction},
        ]
        return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
