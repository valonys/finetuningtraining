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

# Regexes that mark a generated question as trivial metadata and
# cause us to discard the pair post-generation. The prompt forbids
# these too, but models will occasionally slip.
_TRIVIAL_Q_PATTERNS = [
    # "Who is the author / who wrote / who is the illustrator"
    re.compile(r"\bwho\s+(?:is|was)\s+the\s+(?:author|illustrator|editor|publisher|translator)\b", re.IGNORECASE),
    re.compile(r"\bwho\s+(?:wrote|authored|published|edited|illustrated)\b", re.IGNORECASE),
    # "What is the title / what is the name of this book"
    re.compile(r"\bwhat\s+is\s+the\s+(?:title|name)\s+of\s+(?:this|the)\s+(?:book|article|paper|chapter|document)\b", re.IGNORECASE),
    # "How many chapters / sections / pages"
    re.compile(r"\bhow\s+many\s+(?:chapters?|sections?|pages?|words?)\b", re.IGNORECASE),
    # "When was this published / what year"
    re.compile(r"\b(?:when\s+was|what\s+year\s+was)\s+(?:this|the)\s+(?:book|article|paper|document)\s+(?:published|written|released)\b", re.IGNORECASE),
    # "What is the ISBN / DOI / page number"
    re.compile(r"\bwhat\s+is\s+the\s+(?:isbn|doi|url|page\s+number)\b", re.IGNORECASE),
    # "What does the table of contents say"
    re.compile(r"\btable\s+of\s+contents\b", re.IGNORECASE),
    # "What is the copyright year / who owns the copyright"
    re.compile(r"\bcopyright\b", re.IGNORECASE),
]

# Minimum answer length (in chars) to be considered substantive.
# Short answers ("The answer is 12.", "Yes.", "Chapter 5.") are
# almost always trivial one-fact lookups rather than knowledge.
_MIN_ANSWER_CHARS = 80


_SYNTH_INSTRUCTIONS = """\
You are generating high-quality instruction/response pairs for supervised fine-tuning.

Goal: pairs that teach UNDERSTANDING, not trivia lookup. Each pair should be
answerable from the passage alone and should read as if a domain expert were
explaining the concept to a learner.

Produce {n} diverse pairs covering DIFFERENT cognitive angles:

  - **Conceptual** : "Why does X happen?"  "What is the purpose of Y?"
  - **Procedural** : "How do you perform Z?"  "What are the steps to..."
  - **Comparative**: "How does X differ from Y?"  "When would you prefer A over B?"
  - **Causal**    : "What causes X?"  "What are the downstream effects of Y?"
  - **Applied**   : "In scenario X, how would Y apply?"
  - **Analytical* : "What does X imply about Y?"  "What does the data in X tell us?"

DO NOT generate trivial metadata questions. These are EXPLICITLY FORBIDDEN:

  - "Who is the author?" / "Who wrote this?"
  - "What is the title of this book/article?"
  - "How many chapters/sections/pages are there?"
  - "When was this published?"
  - "What is the ISBN / DOI / page number?"
  - "What does the table of contents say?"
  - "What is the copyright year?"

Any pair matching the above will be REJECTED and the run will be considered
a failure.

Quality rules for each pair:

  1. The RESPONSE must be at least 2-3 sentences long (roughly 80+ characters).
     One-word or one-sentence answers are trivia; reject them.
  2. The RESPONSE must use terminology from the passage and expand on it.
     Do not just echo a phrase -- explain what it means and why it matters.
  3. The INSTRUCTION must be self-contained. A reader should be able to
     understand the question without seeing the passage.
  4. The INSTRUCTION must not contain phrases like "in this passage",
     "according to the text", or "as stated above" -- the pairs will be
     used standalone in training.
  5. Do NOT invent facts not present in the passage. Stay grounded.

Passage:
\"\"\"
{chunk}
\"\"\"

Return strictly valid JSON only:

{{"pairs": [{{"instruction": "...", "response": "..."}}, ...]}}
"""


def _is_trivial_question(question: str) -> bool:
    """Return True if the question matches a known metadata-trivia pattern."""
    return any(p.search(question) for p in _TRIVIAL_Q_PATTERNS)


def _is_substantive_answer(answer: str) -> bool:
    """Reject one-liners and yes/no responses as non-teaching."""
    a = answer.strip()
    if len(a) < _MIN_ANSWER_CHARS:
        return False
    # Multiple sentences OR at least one long one
    sents = [s for s in re.split(r"[.!?]+", a) if s.strip()]
    return len(sents) >= 2 or len(a) >= 160


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
            instruction = (it.get("instruction") or "").strip()
            response = (it.get("response") or "").strip()
            if not instruction or not response:
                continue
            # Post-filter trivia the model slipped past the prompt guardrails
            if _is_trivial_question(instruction):
                logger.debug(f"    rejected (trivial Q): {instruction[:60]}...")
                continue
            if not _is_substantive_answer(response):
                logger.debug(f"    rejected (thin A):   {response[:60]}...")
                continue
            pairs.append({
                "instruction": instruction,
                "response": response,
                "source": ch.get("source", ""),
                "synth": provider.name,
            })

    logger.info(f"✅ LLM synth produced {len(pairs)} pairs (post-filter)")
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
