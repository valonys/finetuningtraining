"""
app/eval/runner.py
──────────────────
Orchestrate a full eval run and produce a timestamped JSON report.

The runner is the quality gate in the continuous pipeline:

    Collect → Forge → Train → **Eval** → Deploy (if pass)

A run produces `outputs/<domain>/eval_<timestamp>.json` with:
  * test_loss + perplexity
  * qa_accuracy on a held-out bank
  * llm_judge win_rate (adapted vs base)
  * pass / fail verdict against a configurable threshold

The `/v1/eval/run` endpoint (wired in main.py) calls this so the
Studio UI can trigger evals from the Train tab, and the twice-weekly
cron can gate adapter deployment.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def run_eval(
    *,
    domain_name: str,
    adapter_path: str,
    base_model_id: str,
    test_path: str,
    qa_bank_path: Optional[str] = None,
    judge_prompts: Optional[List[str]] = None,
    generate_base: Optional[Callable[[str], str]] = None,
    generate_adapted: Optional[Callable[[str], str]] = None,
    win_rate_threshold: float = 0.55,
    max_judge_prompts: int = 30,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full eval battery and return a structured report.

    Args:
        domain_name: For file naming and the report header.
        adapter_path: Path to the adapter under evaluation.
        base_model_id: HF model id used as the base.
        test_path: JSONL held-out test split (for loss + perplexity).
        qa_bank_path: Optional JSONL with {instruction, response} rows
            for exact-match accuracy.
        judge_prompts: List of prompts for the LLM-as-judge head-to-head.
            If None and qa_bank_path is set, the first N instructions
            from the QA bank are used.
        generate_base: Callable for base model inference. Needed for
            judge eval. If None, judge eval is skipped.
        generate_adapted: Callable for adapted model inference.
        win_rate_threshold: Minimum win_rate to pass the quality gate.
        max_judge_prompts: Cap on judge evaluations (cost control).
        output_dir: Where to write the report JSON. Defaults to
            ``outputs/<domain_name>/``.

    Returns:
        The full report dict (also written to disk as JSON).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: Dict[str, Any] = {
        "domain": domain_name,
        "adapter_path": adapter_path,
        "base_model": base_model_id,
        "timestamp": ts,
        "sections": {},
    }

    # ── 1. Test-set loss ─────────────────────────────────────────
    if test_path and Path(test_path).exists():
        logger.info("📊 Eval: computing test-set loss on %s", test_path)
        try:
            model, tokenizer = _load_adapted_model(base_model_id, adapter_path)
            from .scorer import eval_loss
            report["sections"]["test_loss"] = eval_loss(
                model, tokenizer, test_path, max_samples=200,
            )
            del model, tokenizer
        except Exception as e:
            logger.exception("eval_loss failed")
            report["sections"]["test_loss"] = {"error": str(e)}
    else:
        report["sections"]["test_loss"] = {"skipped": "no test_path or file not found"}

    # ── 2. QA accuracy ───────────────────────────────────────────
    qa_path = qa_bank_path or test_path
    if qa_path and Path(qa_path).exists() and generate_adapted:
        logger.info("📊 Eval: QA accuracy on %s", qa_path)
        try:
            from .scorer import eval_qa_accuracy
            report["sections"]["qa_accuracy"] = eval_qa_accuracy(
                generate_adapted, qa_path, max_samples=50,
            )
        except Exception as e:
            logger.exception("eval_qa_accuracy failed")
            report["sections"]["qa_accuracy"] = {"error": str(e)}
    else:
        report["sections"]["qa_accuracy"] = {"skipped": "no qa_bank or generate_adapted"}

    # ── 3. LLM-as-judge ─────────────────────────────────────────
    if generate_base and generate_adapted:
        prompts = judge_prompts
        if not prompts and qa_path and Path(qa_path).exists():
            prompts = _extract_prompts(qa_path, max_judge_prompts)
        if prompts:
            logger.info("📊 Eval: LLM-as-judge on %d prompts", len(prompts))
            try:
                from .judge import llm_judge_compare
                report["sections"]["judge"] = llm_judge_compare(
                    prompts, generate_base, generate_adapted,
                    max_prompts=max_judge_prompts,
                )
            except Exception as e:
                logger.exception("llm_judge failed")
                report["sections"]["judge"] = {"error": str(e)}
        else:
            report["sections"]["judge"] = {"skipped": "no prompts available"}
    else:
        report["sections"]["judge"] = {"skipped": "need both generate_base and generate_adapted"}

    # ── Quality gate ─────────────────────────────────────────────
    judge = report["sections"].get("judge", {})
    win_rate = judge.get("win_rate")
    test_loss = (report["sections"].get("test_loss") or {}).get("test_loss")

    if win_rate is not None:
        passed = win_rate >= win_rate_threshold
    elif test_loss is not None:
        # No judge available — fall back to loss threshold
        passed = test_loss < 2.5
    else:
        passed = None  # can't determine

    report["quality_gate"] = {
        "passed": passed,
        "win_rate": win_rate,
        "threshold": win_rate_threshold,
        "test_loss": test_loss,
        "verdict": (
            "PASS — adapter is better than base" if passed is True
            else "FAIL — adapter did not improve over base" if passed is False
            else "INCONCLUSIVE — insufficient eval data"
        ),
    }

    # ── Write report to disk ─────────────────────────────────────
    out = Path(output_dir or f"outputs/{domain_name}")
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"eval_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(report_path)
    logger.info("📊 Eval report → %s | gate=%s", report_path, report["quality_gate"]["verdict"])
    return report


# ── Helpers ──────────────────────────────────────────────────────
def _load_adapted_model(base_model_id: str, adapter_path: str):
    """Load base model + PEFT adapter for eval_loss."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float32,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def _extract_prompts(path: str, n: int) -> List[str]:
    """Pull the first N instruction fields from a JSONL."""
    out = []
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            q = row.get("instruction") or row.get("input") or row.get("question")
            if q:
                out.append(q)
            if len(out) >= n:
                break
    return out
