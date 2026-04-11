"""
Q/A synthesis — converts doc chunks into SFT (instruction, response) pairs.

Two modes:
  1. `rule_based`  — zero-LLM, deterministic extraction (heuristic Q from
                     chunk, chunk itself as A). Fast, no deps. Works on CPU.
  2. `llm`         — uses the provider registry (`app.providers`) to call an
                     OpenAI-compatible LLM. Priority order:
                         OLLAMA_API_KEY  → Ollama Cloud (default: nemotron)
                         OLLAMA_HOST     → local Ollama daemon
                         OPENAI_API_KEY  → OpenAI
                         VALONY_SYNTH_*  → generic endpoint
                     Set `VALONY_SYNTH_PROVIDER` to force a specific one.
                     Falls back to rule-based if no provider is available.

Recommended for dataset build: Ollama Cloud + Nemotron-70B. It's dramatically
cheaper than frontier APIs and dramatically better than local 7B models for
high-volume synthetic data generation.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Iterable, List, Optional

from app.providers import SynthProvider, get_synth_provider

logger = logging.getLogger(__name__)


def synthesize_qa(
    chunks: Iterable[dict],
    *,
    mode: str = "auto",              # "auto" | "rule_based" | "llm"
    per_chunk: int = 2,
    system_prompt: str = "You are a helpful domain expert.",
    provider: Optional[SynthProvider] = None,
) -> list[dict]:
    """
    Args:
        chunks: iterable of {"chunk": str, "source": str, ...}
        mode: "auto" tries the configured LLM then falls back to rule-based
        per_chunk: how many Q/A variants to produce per chunk
        system_prompt: system message fed to the LLM
        provider: explicit provider override (for tests / programmatic use)

    Returns:
        list of {"instruction": str, "response": str, "source": str, "synth": str}
    """
    chunks = list(chunks)
    if not chunks:
        return []

    if mode == "rule_based":
        return _rule_based(chunks, per_chunk=per_chunk)

    # Resolve provider for "llm" / "auto" modes
    if provider is None and mode in ("llm", "auto"):
        provider = get_synth_provider()

    if provider is None:
        if mode == "llm":
            logger.warning("⚠️  LLM synth requested but no provider configured "
                           "(set OLLAMA_API_KEY for Ollama Cloud + Nemotron, or "
                           "OPENAI_API_KEY, or VALONY_SYNTH_BASE_URL). "
                           "Falling back to rule-based.")
        return _rule_based(chunks, per_chunk=per_chunk)

    try:
        return _llm_synth(chunks, provider, per_chunk=per_chunk, system_prompt=system_prompt)
    except Exception as e:
        logger.warning(f"⚠️  LLM synth via {provider.name} failed ({e}) — "
                       f"falling back to rule-based")
        return _rule_based(chunks, per_chunk=per_chunk)


# ──────────────────────────────────────────────────────────────
# Rule-based synthesis
# ──────────────────────────────────────────────────────────────
def _rule_based(chunks: list[dict], *, per_chunk: int) -> list[dict]:
    pairs: list[dict] = []
    for ch in chunks:
        txt = ch["chunk"]
        topic = _first_heading(txt) or _first_sentence(txt) or ""
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
    return re.sub(r"^\s*#+\s+.*\n", "", txt, count=1).strip()


# ──────────────────────────────────────────────────────────────
# LLM-based synthesis (via SynthProvider)
# ──────────────────────────────────────────────────────────────
_SYNTH_INSTRUCTIONS = (
    "From the following passage, produce {n} diverse instruction/answer pairs "
    "suitable for supervised fine-tuning. Each pair must be self-contained — "
    "a reader should be able to answer the instruction from the passage alone. "
    "Vary the phrasing, difficulty, and focus across the {n} pairs. Return "
    'strictly valid JSON in the form: {{"pairs": [{{"instruction": "...", '
    '"response": "..."}}, ...]}}\n\nPassage:\n"""\n{chunk}\n"""'
)


def _llm_synth(
    chunks: list[dict],
    provider: SynthProvider,
    *,
    per_chunk: int,
    system_prompt: str,
) -> list[dict]:
    logger.info(f"🤖 LLM synth via {provider.name} ({provider.model}) — "
                f"{len(chunks)} chunks × {per_chunk} pairs each")

    pairs: list[dict] = []
    for idx, ch in enumerate(chunks):
        user_msg = _SYNTH_INSTRUCTIONS.format(n=per_chunk, chunk=ch["chunk"])
        try:
            content = provider.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.6,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"  [{idx + 1}/{len(chunks)}] provider error: {e}")
            continue

        items = _parse_pairs(content)
        for it in items[:per_chunk]:
            if not it.get("instruction") or not it.get("response"):
                continue
            pairs.append({
                "instruction": it["instruction"].strip(),
                "response": it["response"].strip(),
                "source": ch.get("source", ""),
                "synth": provider.name,
            })

    logger.info(f"✅ LLM synth produced {len(pairs)} pairs")
    return pairs


def _parse_pairs(content: str) -> list[dict]:
    """Be generous — try strict JSON first, then extract an embedded object."""
    content = content.strip()
    if not content:
        return []
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        # Try to find a JSON object inside the response (some models wrap it
        # in markdown fences or prose)
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("pairs", "data", "items", "results"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        # Single pair returned at the top level
        if "instruction" in obj and "response" in obj:
            return [obj]
    return []
