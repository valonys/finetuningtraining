"""
app/eval/judge.py
─────────────────
LLM-as-judge comparative evaluation.

Sends the same prompt to two models (base vs adapted), then asks a
strong judge model (Ollama Cloud Nemotron, or any OpenAI-compatible
endpoint) to pick the better response and rate them on a 1-5 scale.

Returns a structured verdict for each prompt and an aggregate win rate
that the runner uses as the quality gate: deploy the new adapter only
if win_rate > threshold (default 0.6).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# The judge prompt is deliberately terse — long instructions cause
# the judge model to parrot them back instead of rating.
_JUDGE_SYSTEM = (
    "You are a fair evaluator. Given a QUESTION and two ANSWERS (A and B), "
    "decide which answer is better. Reply with ONLY a JSON object:\n"
    '{"winner": "A" or "B" or "tie", "score_a": 1-5, "score_b": 1-5, '
    '"reason": "one sentence"}'
)

_JUDGE_USER = (
    "QUESTION:\n{question}\n\n"
    "ANSWER A:\n{answer_a}\n\n"
    "ANSWER B:\n{answer_b}\n\n"
    "Which is better? Respond ONLY with the JSON."
)


def llm_judge_compare(
    prompts: List[str],
    generate_base: Callable[[str], str],
    generate_adapted: Callable[[str], str],
    *,
    judge_fn: Optional[Callable[[str, str], str]] = None,
    max_prompts: int = 50,
) -> Dict[str, Any]:
    """Run a head-to-head evaluation.

    Args:
        prompts: Questions to evaluate on.
        generate_base: Inference function for the base model.
        generate_adapted: Inference function for the adapted model.
        judge_fn: Optional ``(system, user) -> response`` callable. If
            None, uses the Ollama Cloud provider (reads OLLAMA_API_KEY
            from env).
        max_prompts: Cap on how many prompts to evaluate (cost control).

    Returns::

        {
            "win_rate": float,        # fraction of prompts where adapted > base
            "adapted_wins": int,
            "base_wins": int,
            "ties": int,
            "avg_score_base": float,
            "avg_score_adapted": float,
            "verdicts": [...],        # per-prompt details (capped at 20)
        }
    """
    if judge_fn is None:
        judge_fn = _default_judge()

    evaluated = min(len(prompts), max_prompts)
    adapted_wins = 0
    base_wins = 0
    ties = 0
    scores_base: list[float] = []
    scores_adapted: list[float] = []
    verdicts: list[dict] = []

    for i, q in enumerate(prompts[:evaluated]):
        logger.info(f"  eval {i+1}/{evaluated}: {q[:60]!r}")
        ans_base = generate_base(q).strip()
        ans_adapted = generate_adapted(q).strip()

        # Randomise position to remove order bias
        import random
        if random.random() < 0.5:
            a_text, b_text = ans_base, ans_adapted
            mapping = {"A": "base", "B": "adapted"}
        else:
            a_text, b_text = ans_adapted, ans_base
            mapping = {"A": "adapted", "B": "base"}

        user_msg = _JUDGE_USER.format(question=q, answer_a=a_text, answer_b=b_text)
        try:
            raw = judge_fn(_JUDGE_SYSTEM, user_msg)
            verdict = _parse_verdict(raw)
        except Exception as e:
            logger.warning(f"  judge failed on prompt {i}: {e}")
            verdict = {"winner": "tie", "score_a": 3, "score_b": 3, "reason": f"judge error: {e}"}

        actual_winner = mapping.get(verdict.get("winner", "tie"), "tie")
        if actual_winner == "adapted":
            adapted_wins += 1
        elif actual_winner == "base":
            base_wins += 1
        else:
            ties += 1

        # Map scores back to base/adapted regardless of position
        if mapping["A"] == "base":
            scores_base.append(verdict.get("score_a", 3))
            scores_adapted.append(verdict.get("score_b", 3))
        else:
            scores_adapted.append(verdict.get("score_a", 3))
            scores_base.append(verdict.get("score_b", 3))

        if len(verdicts) < 20:
            verdicts.append({
                "question": q[:200],
                "base_response": ans_base[:300],
                "adapted_response": ans_adapted[:300],
                "winner": actual_winner,
                "reason": verdict.get("reason", ""),
            })

    total = adapted_wins + base_wins + ties
    win_rate = adapted_wins / max(total, 1)
    return {
        "win_rate": round(win_rate, 4),
        "adapted_wins": adapted_wins,
        "base_wins": base_wins,
        "ties": ties,
        "avg_score_base": round(sum(scores_base) / max(len(scores_base), 1), 2),
        "avg_score_adapted": round(sum(scores_adapted) / max(len(scores_adapted), 1), 2),
        "verdicts": verdicts,
    }


def _default_judge():
    """Build a judge callable from the Ollama Cloud provider."""
    from app.providers import get_provider
    provider = get_provider()

    def _call(system: str, user: str) -> str:
        return provider.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=256,
        )
    return _call


def _parse_verdict(raw: str) -> dict:
    """Extract the JSON object from the judge's response."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Find the JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {"winner": "tie", "score_a": 3, "score_b": 3, "reason": "could not parse judge response"}

    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError:
        return {"winner": "tie", "score_a": 3, "score_b": 3, "reason": "invalid JSON from judge"}

    # Normalise
    winner = str(obj.get("winner", "tie")).upper()
    if winner not in ("A", "B"):
        winner = "tie"
    return {
        "winner": winner,
        "score_a": int(obj.get("score_a", 3)),
        "score_b": int(obj.get("score_b", 3)),
        "reason": str(obj.get("reason", "")),
    }
