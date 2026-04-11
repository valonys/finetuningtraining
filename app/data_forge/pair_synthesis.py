"""
Contrastive pair synthesis for DPO / ORPO / KTO training.

Given a seed (instruction, optional ground truth), produce a plausible
`(chosen, rejected)` pair where:

  * `chosen`   — high-quality, accurate, well-structured answer
  * `rejected` — plausible but subtly wrong answer (factually off, too
                 brief, missing the key step, wrong tone, or breaking a
                 stated rule). NOT gibberish — the whole point of DPO is
                 that the model learns to distinguish realistic mistakes
                 from correct behaviour.

The naive "truncate the good answer" approach (which the previous
`dataset_builder._to_dpo_row` used as a placeholder) produces almost no
useful training signal — the model just learns "longer = better", which
causes runaway length and degenerate preferences.

With a strong teacher (Nemotron-70B on Ollama Cloud), a single structured
API call gives us both responses in JSON. This module exposes:

    synthesize_pairs(pairs, *, provider=None) → list[dict]
        where each output row is
        {"instruction": ..., "chosen": ..., "rejected": ..., "source": ...}

Falls back to a deterministic rule-based rejector when no provider is
configured, so offline runs still produce valid (if weak) training data.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterable, List, Optional

from app.providers import SynthProvider, get_synth_provider

logger = logging.getLogger(__name__)


_PAIR_PROMPT = (
    "Given this instruction, produce TWO responses:\n"
    '  1. "chosen":   a high-quality, accurate, well-structured answer.\n'
    '  2. "rejected": a PLAUSIBLE but subtly wrong answer — factually off,\n'
    "                 missing a key step, too brief, wrong tone, or breaking\n"
    "                 a stated rule. It must NOT be gibberish, nonsense, or\n"
    "                 an outright refusal — the model needs to learn from\n"
    "                 realistic mistakes, not junk.\n\n"
    "Instruction:\n\"\"\"\n{instruction}\n\"\"\"\n"
    "{gt_block}"
    '\nReturn strictly valid JSON: {{"chosen": "...", "rejected": "..."}}'
)


def synthesize_pairs(
    seeds: Iterable[dict],
    *,
    provider: Optional[SynthProvider] = None,
    system_prompt: str = "You are generating contrastive response pairs for DPO training.",
) -> List[dict]:
    """
    Args:
        seeds: iterable of dicts with at least {"instruction": str}. May also
               contain {"response": str} (used as reference ground truth) and
               {"source": str}.
        provider: explicit provider override (else env-based auto-detect).
        system_prompt: system message for the LLM.

    Returns:
        list of {"instruction", "chosen", "rejected", "source", "synth"}
    """
    seeds = list(seeds)
    if not seeds:
        return []

    if provider is None:
        provider = get_synth_provider()

    if provider is None:
        logger.warning("⚠️  No provider configured for pair synth — "
                       "using rule-based rejector (low training signal; "
                       "set OLLAMA_API_KEY for real pairs via Nemotron)")
        return [_rule_based_pair(s) for s in seeds]

    logger.info(f"🎯 Pair synth via {provider.name} ({provider.model}) — "
                f"{len(seeds)} seeds")

    out: List[dict] = []
    for idx, seed in enumerate(seeds):
        instruction = seed["instruction"]
        reference = seed.get("response")
        gt_block = (
            f'Reference answer (use as the quality bar for "chosen"):\n"""\n{reference}\n"""\n'
            if reference else ""
        )
        user_msg = _PAIR_PROMPT.format(instruction=instruction, gt_block=gt_block)

        try:
            content = provider.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"  [{idx + 1}/{len(seeds)}] provider error: {e}")
            out.append(_rule_based_pair(seed))
            continue

        parsed = _parse_pair(content)
        if parsed is None:
            logger.warning(f"  [{idx + 1}/{len(seeds)}] unparsable LLM output — rule-based fallback")
            out.append(_rule_based_pair(seed))
            continue

        chosen, rejected = parsed
        # Safety: drop rows where chosen and rejected are identical (useless)
        if chosen.strip() == rejected.strip():
            logger.debug(f"  [{idx + 1}] chosen == rejected, skipping")
            continue

        out.append({
            "instruction": instruction,
            "chosen": chosen,
            "rejected": rejected,
            "source": seed.get("source", ""),
            "synth": provider.name,
        })

    logger.info(f"✅ Pair synth produced {len(out)} (chosen, rejected) pairs")
    return out


# ──────────────────────────────────────────────────────────────
def _parse_pair(content: str) -> Optional[tuple[str, str]]:
    content = content.strip()
    if not content:
        return None
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None
    chosen = obj.get("chosen") or obj.get("good") or obj.get("accepted")
    rejected = obj.get("rejected") or obj.get("bad") or obj.get("wrong")
    if not chosen or not rejected:
        return None
    return str(chosen).strip(), str(rejected).strip()


def _rule_based_pair(seed: dict) -> dict:
    """
    Offline fallback — produces a weak but valid DPO row.

    Strategy: take the reference response (if any) as chosen, and synthesize
    a rejected response by (a) truncating to ~25% of length and (b) replacing
    the last sentence with a generic hedge. Low signal, but it's **labeled**
    low-signal so downstream filtering can drop it.
    """
    instruction = seed["instruction"]
    response = seed.get("response") or ""
    chosen = response.strip() or "I can help with that."
    if len(chosen) > 40:
        rejected = chosen[: max(20, len(chosen) // 4)].strip() + "... (I'm not sure, consider checking elsewhere.)"
    else:
        rejected = "I'm not sure about this — you may want to consult another source."
    return {
        "instruction": instruction,
        "chosen": chosen,
        "rejected": rejected,
        "source": seed.get("source", ""),
        "synth": "rule_based_weak",
    }
