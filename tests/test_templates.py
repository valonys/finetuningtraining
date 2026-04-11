"""
Template registry sanity tests — run with `pytest tests/`.

No GPU, no model download — these validate the model-ID → template mapping
and the format_* methods produce non-empty strings with the right markers.
"""
from __future__ import annotations

import pytest

from app.templates import get_template_for, list_templates


@pytest.mark.parametrize("model_id, expected", [
    ("Qwen/Qwen2.5-7B-Instruct",              "qwen"),
    ("Qwen/Qwen3-8B-Instruct",                "qwen"),
    ("unsloth/Qwen2.5-1.5B",                  "qwen"),
    ("meta-llama/Llama-3.2-1B-Instruct",      "llama3"),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama3"),
    ("meta-llama/Llama-2-7b-chat-hf",         "llama2"),
    ("mistralai/Mistral-7B-Instruct-v0.3",    "mistral"),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1",  "mistral"),
    ("google/gemma-2-9b-it",                  "gemma"),
    ("microsoft/Phi-3.5-mini-instruct",       "phi"),
    ("microsoft/Phi-4",                       "phi"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek"),
    ("some/unknown-model",                    "chatml"),  # fallback
])
def test_template_routing(model_id, expected):
    tpl = get_template_for(model_id)
    assert tpl.name == expected, f"{model_id} routed to {tpl.name}, expected {expected}"


def test_format_sft_produces_response():
    tpl = get_template_for("Qwen/Qwen2.5-7B-Instruct")
    txt = tpl.format_sft(system="s", instruction="q", response="a")
    assert "<|im_start|>assistant" in txt
    assert txt.endswith("<|im_end|>")


def test_format_prompt_only_no_response_token():
    tpl = get_template_for("meta-llama/Llama-3.2-1B-Instruct")
    p = tpl.format_prompt_only(system="s", instruction="q")
    assert p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_registry_nonempty():
    assert "qwen" in list_templates()
    assert "llama3" in list_templates()
    assert "chatml" in list_templates()


def test_as_messages_round_trip():
    tpl = get_template_for("Qwen/Qwen2.5-7B-Instruct")
    msgs = tpl.as_messages(system="S", instruction="I", response="R")
    assert msgs == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "I"},
        {"role": "assistant", "content": "R"},
    ]
