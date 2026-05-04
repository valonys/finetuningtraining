"""
Unit tests for app.pipeline.runner.PipelineRunner.

Every test wires fake stages into the runner — the goal is to verify the
*orchestration* mechanics (state machine, resume, idempotency keys, hard
gates) independent of any real collect/forge/train code path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline import (
    PipelineRunner,
    RunContext,
    Stage,
    StageResult,
    StageStatus,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _make_runner(tmp_path: Path, *, config: dict | None = None) -> PipelineRunner:
    return PipelineRunner(
        domain="testdom",
        config=config or {"k": "v"},
        runs_root=tmp_path / "runs",
    )


def _read_status(run_dir: Path) -> dict:
    return json.loads((run_dir / "stage_status.json").read_text())


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────
def test_happy_path_executes_all_stages_in_order(tmp_path):
    calls: list[str] = []

    def make(name):
        def fn(ctx: RunContext) -> StageResult:
            calls.append(name)
            return StageResult(artifacts={"who": name})
        return Stage(name=name, fn=fn)

    runner = _make_runner(tmp_path)
    ctx = runner.start_run(stages_requested=["a", "b", "c"])
    report = runner.execute([make("a"), make("b"), make("c")], ctx)

    assert calls == ["a", "b", "c"]
    assert report["status"] == "completed"
    assert report["halt_reason"] is None
    assert [s["name"] for s in report["stages"]] == ["a", "b", "c"]
    assert all(s["status"] == "completed" for s in report["stages"])

    # Each stage's artifacts available to the next via stage_outputs
    status = _read_status(ctx.run_dir)
    assert status["stages"]["a"]["artifacts"] == {"who": "a"}
    assert status["stages"]["c"]["status"] == "completed"

    # report.json + manifest.json are persisted
    assert (ctx.run_dir / "report.json").is_file()
    assert (ctx.run_dir / "manifest.json").is_file()


def test_manifest_freezes_inputs(tmp_path):
    runner = _make_runner(tmp_path, config={"base_model_id": "Qwen/x", "method": "sft"})
    ctx = runner.start_run(stages_requested=["forge"])
    manifest = json.loads((ctx.run_dir / "manifest.json").read_text())
    assert manifest["domain"] == "testdom"
    assert manifest["config"]["base_model_id"] == "Qwen/x"
    assert "config_sha256" in manifest
    assert manifest["stages_requested"] == ["forge"]


def test_stage_failure_halts_pipeline_and_records_error(tmp_path):
    calls: list[str] = []

    def good(name):
        def fn(_):
            calls.append(name)
            return StageResult(artifacts={"who": name})
        return Stage(name=name, fn=fn)

    def boom(_):
        calls.append("boom")
        raise RuntimeError("kaboom")

    runner = _make_runner(tmp_path)
    ctx = runner.start_run()
    report = runner.execute([good("a"), Stage("b", boom), good("c")], ctx)

    assert calls == ["a", "boom"]      # stage c never ran
    assert report["status"] == "failed"
    assert "RuntimeError" in (report["halt_reason"] or "")
    status = _read_status(ctx.run_dir)
    assert status["stages"]["a"]["status"] == "completed"
    assert status["stages"]["b"]["status"] == "failed"
    assert "kaboom" in status["stages"]["b"]["error"]
    assert "traceback" in status["stages"]["b"]
    assert "c" not in status["stages"]


def test_resume_skips_completed_stages_when_key_matches(tmp_path):
    """First run completes 'a', crashes in 'b'. Second run resumes 'a',
    re-runs 'b' (now succeeding), runs 'c'."""
    a_calls = []
    b_calls = []
    c_calls = []

    def stage_a(ctx):
        a_calls.append(1)
        return StageResult(artifacts={"phase": "alpha"})

    def stage_c(_):
        c_calls.append(1)
        return StageResult(artifacts={"phase": "gamma"})

    runner = _make_runner(tmp_path)
    ctx1 = runner.start_run()

    # First run: 'b' raises
    def b_fail(_):
        b_calls.append(1)
        raise RuntimeError("first attempt fails")

    runner.execute(
        [Stage("a", stage_a), Stage("b", b_fail), Stage("c", stage_c)],
        ctx1,
    )
    assert a_calls == [1]
    assert b_calls == [1]
    assert c_calls == []

    # Resume: same runner instance simulating new process
    runner2 = _make_runner(tmp_path)
    ctx2 = runner2.resume_run(ctx1.run_id)

    def b_ok(_):
        b_calls.append(2)
        return StageResult(artifacts={"phase": "beta"})

    report = runner2.execute(
        [Stage("a", stage_a), Stage("b", b_ok), Stage("c", stage_c)],
        ctx2,
    )

    assert a_calls == [1]                      # NOT re-run
    assert b_calls == [1, 2]                   # re-run because status was failed
    assert c_calls == [1]
    assert report["status"] == "completed"
    # 'a' was skipped via resume; 'b' and 'c' completed fresh
    by_name = {s["name"]: s for s in report["stages"]}
    assert by_name["a"]["status"] == "skipped"
    assert by_name["b"]["status"] == "completed"
    assert by_name["c"]["status"] == "completed"

    # Resume restored 'a' artifacts into ctx
    assert ctx2.stage_outputs["a"] == {"phase": "alpha"}


def test_idempotency_key_change_forces_rerun(tmp_path):
    """If a stage's idempotency key changes between invocations (e.g.
    upstream input drifted), the previously COMPLETED stage must re-run
    instead of being silently skipped."""
    calls = []

    def fn(_):
        calls.append(1)
        return StageResult(artifacts={"n": len(calls)})

    counter = {"v": "first"}

    def key_fn(_):
        return counter["v"]

    stage = Stage("s", fn, idempotency_key_fn=key_fn)

    runner = _make_runner(tmp_path)
    ctx = runner.start_run()
    runner.execute([stage], ctx)
    assert calls == [1]

    # Same key → resume skips
    runner.execute([stage], ctx)
    assert calls == [1]                  # skipped

    # Drift the key → must re-run
    counter["v"] = "second"
    runner.execute([stage], ctx)
    assert calls == [1, 1]               # re-ran


def test_hard_gate_failure_halts_before_downstream(tmp_path):
    """A stage that completes but reports gate_passed=False must block
    every downstream stage — this is how eval blocks deploy."""
    deploy_calls = []

    def evalfn(_):
        return StageResult(artifacts={"score": 0.42}, gate_passed=False)

    def deployfn(_):
        deploy_calls.append(1)
        return StageResult()

    runner = _make_runner(tmp_path)
    ctx = runner.start_run()
    report = runner.execute(
        [Stage("eval", evalfn), Stage("deploy", deployfn)],
        ctx,
    )

    assert deploy_calls == []
    assert report["status"] == "failed"
    assert "hard gate" in (report["halt_reason"] or "")
    status = _read_status(ctx.run_dir)
    assert status["stages"]["eval"]["status"] == "completed"
    assert status["stages"]["eval"]["gate_passed"] is False
    assert "deploy" not in status["stages"]


def test_atomic_write_no_tmp_files_left_behind(tmp_path):
    runner = _make_runner(tmp_path)
    ctx = runner.start_run()

    def stage_fn(_):
        return StageResult(artifacts={"x": 1})

    runner.execute([Stage("a", stage_fn)], ctx)

    # No leftover .tmp files
    leftovers = list(ctx.run_dir.glob("*.tmp"))
    assert leftovers == [], f"atomic write leaked tmp files: {leftovers}"


def test_resume_raises_when_run_dir_missing(tmp_path):
    runner = _make_runner(tmp_path)
    with pytest.raises(FileNotFoundError, match="cannot resume"):
        runner.resume_run("nonexistent-run-id")


def test_select_stages_filters_in_canonical_order(tmp_path):
    from app.pipeline.stages import DEFAULT_STAGES, select_stages

    # Default returns all in pipeline order
    assert [s.name for s in select_stages()] == [s.name for s in DEFAULT_STAGES]

    # Filtered subset preserves canonical order even if requested out of order
    picked = select_stages(["deploy", "collect", "train"])
    assert [s.name for s in picked] == ["collect", "train", "deploy"]

    # Unknown stage name raises
    with pytest.raises(ValueError, match="Unknown stage"):
        select_stages(["forge", "definitely_not_a_stage"])


def test_stage_outputs_threaded_to_downstream(tmp_path):
    """A downstream stage should see upstream artifacts via ctx.stage_outputs."""
    seen: dict = {}

    def upstream(ctx):
        return StageResult(artifacts={"path": "/tmp/dataset.jsonl", "samples": 500})

    def downstream(ctx):
        seen.update(ctx.stage_outputs.get("forge", {}))
        return StageResult()

    runner = _make_runner(tmp_path)
    ctx = runner.start_run()
    runner.execute(
        [Stage("forge", upstream), Stage("train", downstream)],
        ctx,
    )
    assert seen == {"path": "/tmp/dataset.jsonl", "samples": 500}
