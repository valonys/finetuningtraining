"""
app/eval/scorer.py
──────────────────
Quantitative metrics on a held-out test set:

  * eval_loss         — average cross-entropy on a JSONL test split
  * eval_qa_accuracy  — exact-match (or substring-match) on a Q/A bank

Both return a plain dict so they compose cleanly into the runner's
JSON report.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def eval_loss(
    model,
    tokenizer,
    test_path: str,
    *,
    max_seq_length: int = 2048,
    max_samples: int = 200,
) -> Dict[str, Any]:
    """Compute average per-token cross-entropy loss on a held-out JSONL.

    Expects rows with a `text` field (already template-formatted) or
    `instruction` + `response` fields. Returns::

        {"test_loss": float, "perplexity": float, "samples": int}
    """
    import torch
    from torch.nn import CrossEntropyLoss

    rows = _load_jsonl(test_path, max_samples=max_samples)
    if not rows:
        logger.warning("eval_loss: no rows in %s", test_path)
        return {"test_loss": None, "perplexity": None, "samples": 0}

    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for row in rows:
            text = row.get("text") or _format_row(row)
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_seq_length).to(device)
            input_ids = enc["input_ids"]
            labels = input_ids.clone()

            outputs = model(**enc, labels=labels)
            n_tokens = input_ids.numel()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return {"test_loss": round(avg_loss, 4), "perplexity": round(ppl, 2), "samples": len(rows)}


def eval_qa_accuracy(
    generate_fn,
    test_path: str,
    *,
    max_samples: int = 50,
    match_mode: str = "substring",
) -> Dict[str, Any]:
    """Run a Q/A bank through the model and score exact/substring match.

    Each row needs `instruction` (the question) and `response` (the
    gold answer). We generate a model response and check whether the
    gold answer appears.

    `generate_fn(prompt: str) -> str` is the callable that drives
    inference (so the scorer is backend-agnostic).

    Returns::

        {"accuracy": float 0-1, "correct": int, "total": int,
         "wrong": [{"q": str, "expected": str, "got": str}, ...]}
    """
    rows = _load_jsonl(test_path, max_samples=max_samples)
    if not rows:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "wrong": []}

    correct = 0
    wrong: List[Dict[str, str]] = []
    for row in rows:
        q = row.get("instruction") or row.get("input") or row.get("question", "")
        gold = row.get("response") or row.get("output") or row.get("answer", "")
        if not q or not gold:
            continue

        pred = generate_fn(q).strip()
        hit = _check_match(pred, gold, match_mode)
        if hit:
            correct += 1
        else:
            wrong.append({"q": q[:200], "expected": gold[:200], "got": pred[:200]})

    total = correct + len(wrong)
    acc = correct / max(total, 1)
    return {"accuracy": round(acc, 4), "correct": correct, "total": total, "wrong": wrong[:10]}


# ── Helpers ──────────────────────────────────────────────────────
def _load_jsonl(path: str, max_samples: int = 200) -> List[dict]:
    p = Path(path)
    if not p.exists():
        logger.warning("eval: file not found: %s", path)
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    return rows


def _format_row(row: dict) -> str:
    inst = row.get("instruction") or row.get("input") or row.get("question", "")
    resp = row.get("response") or row.get("output") or row.get("answer", "")
    return f"### Instruction:\n{inst}\n\n### Response:\n{resp}"


def _check_match(prediction: str, gold: str, mode: str) -> bool:
    pred_lower = prediction.lower().strip()
    gold_lower = gold.lower().strip()
    if mode == "exact":
        return pred_lower == gold_lower
    # substring: gold answer appears somewhere in the model output
    return gold_lower in pred_lower
