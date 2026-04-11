"""
Q/A synthesis — converts doc chunks into SFT (instruction, response) pairs.

Two modes:
  1. `rule_based`  — zero-LLM, deterministic extraction (heuristic Q from chunk,
                     chunk itself as A). Fast, no deps. Works on CPU.
  2. `llm`         — uses any OpenAI-compatible chat endpoint (Ollama, vLLM
                     OpenAI server, OpenRouter, etc.) to synthesise high-quality
                     Q/A pairs. Endpoint & model are taken from env vars:
                         VALONY_SYNTH_BASE_URL
                         VALONY_SYNTH_MODEL
                         VALONY_SYNTH_API_KEY
                     Falls back to rule-based if any are missing.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Iterable

logger = logging.getLogger(__name__)


def synthesize_qa(
    chunks: Iterable[dict],
    *,
    mode: str = "auto",              # "auto" | "rule_based" | "llm"
    per_chunk: int = 2,
    system_prompt: str = "You are a helpful domain expert.",
) -> list[dict]:
    """
    Args:
        chunks: iterable of {"chunk": str, "source": str, ...}
        mode:   "auto" tries LLM then falls back to rule-based
    Returns:
        list of {"instruction": str, "response": str, "source": str}
    """
    chunks = list(chunks)
    if not chunks:
        return []

    use_llm = False
    if mode in ("llm", "auto"):
        use_llm = _llm_ready()
        if mode == "llm" and not use_llm:
            logger.warning("⚠️  LLM synth requested but env vars missing — falling back to rule-based")

    if use_llm:
        try:
            return _llm_synth(chunks, per_chunk=per_chunk, system_prompt=system_prompt)
        except Exception as e:
            logger.warning(f"⚠️  LLM synth failed ({e}) — falling back to rule-based")

    return _rule_based(chunks, per_chunk=per_chunk)


# ──────────────────────────────────────────────────────────────
# Rule-based synthesis
# ──────────────────────────────────────────────────────────────
def _rule_based(chunks: list[dict], *, per_chunk: int) -> list[dict]:
    pairs: list[dict] = []
    for ch in chunks:
        txt = ch["chunk"]
        # Heuristic: take the first heading (if any) as a topic hint
        topic = _first_heading(txt) or _first_sentence(txt) or ""
        # Produce up to `per_chunk` variants
        variants = _question_variants(topic or "this section")[:per_chunk]
        for q in variants:
            pairs.append({
                "instruction": q,
                "response": _clean_response(txt),
                "source": ch.get("source", ""),
                "synth": "rule_based",
            })
    return pairs


def _first_heading(txt: str) -> str:
    m = re.search(r"^\s*#+\s+(.*)$", txt, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _first_sentence(txt: str) -> str:
    m = re.match(r"\s*(.+?[\.\?\!])", txt, re.DOTALL)
    return m.group(1).strip() if m else ""


def _question_variants(topic: str) -> list[str]:
    topic = topic.strip(". ?!")
    return [
        f"Explain {topic}.",
        f"What should I know about {topic}?",
        f"Summarise the key points of {topic}.",
        f"Given the context, describe {topic} in detail.",
    ]


def _clean_response(txt: str) -> str:
    # Strip leading headings so the answer reads naturally
    return re.sub(r"^\s*#+\s+.*\n", "", txt, count=1).strip()


# ──────────────────────────────────────────────────────────────
# LLM-based synthesis (OpenAI-compatible chat API)
# ──────────────────────────────────────────────────────────────
def _llm_ready() -> bool:
    return bool(os.environ.get("VALONY_SYNTH_BASE_URL") and os.environ.get("VALONY_SYNTH_MODEL"))


def _llm_synth(chunks: list[dict], *, per_chunk: int, system_prompt: str) -> list[dict]:
    try:
        import requests
    except ImportError:
        raise RuntimeError("`requests` required for LLM synthesis")

    base_url = os.environ["VALONY_SYNTH_BASE_URL"].rstrip("/")
    model = os.environ["VALONY_SYNTH_MODEL"]
    api_key = os.environ.get("VALONY_SYNTH_API_KEY", "sk-local")

    pairs: list[dict] = []
    for ch in chunks:
        prompt = (
            f"From the following passage, produce {per_chunk} diverse instruction/answer "
            f"pairs suitable for supervised fine-tuning. Return JSON only in the form:\n"
            f'[{{"instruction": "...", "response": "..."}}]\n\n'
            f"Passage:\n\"\"\"\n{ch['chunk']}\n\"\"\""
        )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.6,
            "response_format": {"type": "json_object"},
        }
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        try:
            obj = json.loads(content)
            items = obj if isinstance(obj, list) else obj.get("pairs") or obj.get("data") or []
        except Exception:
            items = []
        for it in items:
            pairs.append({
                "instruction": it.get("instruction", "").strip(),
                "response": it.get("response", "").strip(),
                "source": ch.get("source", ""),
                "synth": "llm",
            })
    return pairs
