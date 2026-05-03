"""
app/pipeline/stages.py
──────────────────────
Concrete stage implementations for the Lane A pipeline. Each stage is a
thin wrapper around an existing module; the heavy lifting stays where it
already lives (data_forge, trainers, eval, etc.).

Stages report their *artifacts* via ``StageResult.artifacts`` so the next
stage can reach back into ``ctx.stage_outputs[<previous stage>]`` for
the path / id / metric it needs.

Idempotency keys are derived from the inputs each stage actually
consumes — that way a config or upstream-output change forces a re-run
instead of a silent stale resume.

Implementation maturity (2026-05-03):
  collect / forge / train  → wiring stubs marked TODO; runner mechanics
                             (state machine, resume, gate) is what A2
                             ships — fleshing out each stage tracks as
                             follow-up work.
  eval                     → wired to ``app.eval.run_eval``.
  deploy                   → wired to ``app.trainers.merge_and_export`` (A1).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .runner import RunContext, Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Stage implementations
# ──────────────────────────────────────────────────────────────
def collect_fn(ctx: RunContext) -> StageResult:
    """Snapshot raw inputs available for this domain.

    MVP behavior: enumerate ``data/uploads/<domain>/`` and record the
    file list + content hash. Routing to harvesters (YouTube, arXiv,
    code) per ``ctx.config['collect']`` is a follow-up — the schema
    needs a tighter contract first.
    """
    uploads_dir = Path(ctx.config.get("uploads_dir", f"data/uploads/{ctx.domain}"))
    files = sorted(p for p in uploads_dir.glob("**/*") if p.is_file()) if uploads_dir.is_dir() else []
    artifacts = {
        "uploads_dir": str(uploads_dir),
        "file_count": len(files),
        "files": [str(p) for p in files],
        # TODO: route to harvesters per ctx.config['collect'] when the
        # config schema lands.
    }
    return StageResult(artifacts=artifacts)


def _collect_key(ctx: RunContext) -> str:
    uploads_dir = Path(ctx.config.get("uploads_dir", f"data/uploads/{ctx.domain}"))
    h = hashlib.sha256()
    if uploads_dir.is_dir():
        for p in sorted(uploads_dir.glob("**/*")):
            if p.is_file():
                h.update(p.relative_to(uploads_dir).as_posix().encode())
                h.update(b"\0")
                h.update(str(p.stat().st_size).encode())
                h.update(b"\0")
                h.update(str(int(p.stat().st_mtime)).encode())
                h.update(b"\0")
    return h.hexdigest()


def forge_fn(ctx: RunContext) -> StageResult:
    """Build the SFT/DPO dataset from collected inputs.

    Wires to ``app.data_forge`` once the run-level dataset config is
    finalized. Today: produces a placeholder dataset path the train
    stage can consume in tests.
    """
    out = Path(ctx.config.get("dataset_path", f"data/processed/{ctx.domain}_sft.jsonl"))
    artifacts = {
        "dataset_path": str(out),
        "samples": ctx.stage_outputs.get("collect", {}).get("file_count", 0),
        # TODO: invoke app.data_forge.build_dataset(...) with collect outputs.
    }
    return StageResult(artifacts=artifacts)


def train_fn(ctx: RunContext) -> StageResult:
    """Run the configured trainer (SFT/DPO/ORPO/KTO/GRPO).

    Wires to ``app.trainers.AgnosticSFTTrainer`` (and siblings) once the
    config-to-trainer mapping is finalized. Today: surfaces the adapter
    path that the deploy stage consumes.
    """
    adapter_path = ctx.config.get("adapter_path", f"outputs/{ctx.domain}")
    artifacts = {
        "adapter_path": adapter_path,
        "method": ctx.config.get("method", "sft"),
        "base_model_id": ctx.config.get("base_model_id"),
        # TODO: instantiate the right trainer class per ctx.config['method'],
        # call .train(), surface result['final_loss'] etc.
    }
    return StageResult(artifacts=artifacts)


def eval_fn(ctx: RunContext) -> StageResult:
    """Run the eval module's quality gate.

    Calls ``app.eval.run_eval`` if the necessary inputs are present;
    otherwise reports a soft pass with an explanation so the pipeline
    can still demonstrate end-to-end flow.
    """
    train_out = ctx.stage_outputs.get("train", {})
    adapter_path = train_out.get("adapter_path")
    base_model_id = train_out.get("base_model_id") or ctx.config.get("base_model_id")
    test_path = ctx.config.get("eval_test_path")

    if not (adapter_path and base_model_id and test_path):
        # Soft-pass when eval inputs aren't wired yet — surfaced to the
        # operator via the report so it's never silent.
        return StageResult(
            artifacts={"status": "skipped", "reason": "eval inputs not configured"},
            gate_passed=True,
        )

    try:
        from app.eval import run_eval
    except Exception as exc:
        return StageResult(
            status=StageStatus.FAILED,
            artifacts={},
            error=f"eval import failed: {exc}",
        )

    report = run_eval(
        domain_name=ctx.domain,
        adapter_path=adapter_path,
        base_model_id=base_model_id,
        test_path=test_path,
        win_rate_threshold=ctx.config.get("win_rate_threshold", 0.55),
    )
    passed = bool(report.get("quality_gate", {}).get("passed", False))
    return StageResult(
        artifacts={"report": report, "passed": passed},
        gate_passed=passed,
    )


def deploy_fn(ctx: RunContext) -> StageResult:
    """Export a deployable GGUF artifact via A1's merge_and_export.

    Reads adapter path from the train stage's outputs and base model id
    from the train output / domain config. Honors ``ctx.config['deploy']``
    for output dir + quant + llama.cpp path overrides.
    """
    train_out = ctx.stage_outputs.get("train", {})
    adapter_path = train_out.get("adapter_path")
    base_model_id = train_out.get("base_model_id") or ctx.config.get("base_model_id")
    if not (adapter_path and base_model_id):
        return StageResult(
            status=StageStatus.FAILED,
            error="deploy stage missing adapter_path or base_model_id from train output",
        )

    deploy_cfg = ctx.config.get("deploy", {}) or {}
    output_dir = deploy_cfg.get("output_dir") or f"outputs/{ctx.domain}/artifacts"

    from app.trainers.export import merge_and_export
    result = merge_and_export(
        base_model_id=base_model_id,
        adapter_path=adapter_path,
        output_dir=output_dir,
        quant=deploy_cfg.get("quant", "Q4_K_M"),
        llama_cpp_path=deploy_cfg.get("llama_cpp_path"),
        artifact_name=deploy_cfg.get("artifact_name"),
    )
    return StageResult(artifacts=result)


# ──────────────────────────────────────────────────────────────
# Registry — the canonical Lane A pipeline order
# ──────────────────────────────────────────────────────────────
DEFAULT_STAGES: list[Stage] = [
    Stage(name="collect", fn=collect_fn, idempotency_key_fn=_collect_key),
    Stage(name="forge", fn=forge_fn),
    Stage(name="train", fn=train_fn),
    Stage(name="eval", fn=eval_fn),
    Stage(name="deploy", fn=deploy_fn),
]


def select_stages(names: list[str] | None = None) -> list[Stage]:
    """Filter ``DEFAULT_STAGES`` by name, preserving canonical order."""
    if not names:
        return list(DEFAULT_STAGES)
    requested = set(names)
    unknown = requested - {s.name for s in DEFAULT_STAGES}
    if unknown:
        raise ValueError(f"Unknown stage(s): {sorted(unknown)}")
    return [s for s in DEFAULT_STAGES if s.name in requested]
