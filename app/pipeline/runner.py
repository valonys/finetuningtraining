"""
app/pipeline/runner.py
──────────────────────
Stage orchestration with crash-safe state, idempotency-keyed resume,
and hard-gate enforcement. Closes A2 of the Lane A blueprint
(see ``docs/SPRINTS.md``).

Pipeline contract:
    collect → forge → train → eval → deploy

Key invariants:
  * State is persisted with atomic write-then-rename so a crash mid-stage
    leaves the previous committed state intact.
  * A stage is *resumed* (skipped, artifacts restored) only when both
    (a) its previous status was COMPLETED and
    (b) its current idempotency key matches the recorded one.
    Any input drift forces a re-run, and downstream stages are
    invalidated transparently.
  * A FAILED stage or a passed stage that flips ``gate_passed=False``
    halts execution before any later stage runs. Promotion is gated on
    every preceding hard gate having passed.

The runner is intentionally generic about *what* each stage does — the
concrete stage implementations live in ``app/pipeline/stages.py``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────
class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"        # resumed from prior completed state


@dataclass
class StageResult:
    status: StageStatus = StageStatus.COMPLETED
    artifacts: dict[str, Any] = field(default_factory=dict)
    gate_passed: bool = True
    error: str | None = None


@dataclass
class RunContext:
    run_id: str
    domain: str
    run_dir: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    stage_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Stage:
    """A pipeline step. ``fn`` does the work; ``idempotency_key_fn``
    derives a string fingerprint from the run context. The runner skips
    a stage whose recorded status is COMPLETED *and* whose current key
    matches the recorded key."""
    name: str
    fn: Callable[[RunContext], StageResult]
    idempotency_key_fn: Callable[[RunContext], str] | None = None

    def key_for(self, ctx: RunContext) -> str:
        if self.idempotency_key_fn is None:
            # Default: hash of (manifest snapshot + stage name). Stages
            # that depend on upstream outputs should provide a custom fn.
            payload = json.dumps(
                {"manifest": ctx.manifest, "stage": self.name},
                sort_keys=True,
                default=str,
            )
            return hashlib.sha256(payload.encode()).hexdigest()
        return self.idempotency_key_fn(ctx)


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────
class PipelineRunner:
    """Orchestrate a sequence of stages with resume + gate semantics.

    Typical usage::

        runner = PipelineRunner(
            domain="ai_llm",
            config=load_domain_config("ai_llm"),
            runs_root=Path("outputs/runs"),
        )
        ctx = runner.start_run(stages_requested=["collect", "forge", ...])
        report = runner.execute([collect_stage, forge_stage, ...], ctx)

    Resume an existing run::

        ctx = runner.resume_run(run_id="20260503-093000-ai_llm")
        report = runner.execute([...same stages...], ctx)
    """

    MANIFEST_FILE = "manifest.json"
    STATUS_FILE = "stage_status.json"

    def __init__(
        self,
        *,
        domain: str,
        config: dict[str, Any] | None = None,
        runs_root: Path | str = "outputs/runs",
        clock: Callable[[], datetime] | None = None,
    ):
        self.domain = domain
        self.config = config or {}
        self.runs_root = Path(runs_root)
        self._now = clock or (lambda: datetime.now(timezone.utc))

    # ── Run lifecycle ─────────────────────────────────────────
    def start_run(
        self,
        *,
        stages_requested: list[str] | None = None,
        run_id: str | None = None,
        extra_manifest: dict[str, Any] | None = None,
    ) -> RunContext:
        """Begin a fresh run, writing the immutable manifest snapshot."""
        rid = run_id or self._generate_run_id()
        run_dir = self.runs_root / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "run_id": rid,
            "domain": self.domain,
            "config": self.config,
            "config_sha256": _hash_obj(self.config),
            "stages_requested": stages_requested or [],
            "created_at": self._now().isoformat(timespec="seconds"),
        }
        if extra_manifest:
            manifest.update(extra_manifest)

        _atomic_write_json(run_dir / self.MANIFEST_FILE, manifest)
        _atomic_write_json(
            run_dir / self.STATUS_FILE,
            {
                "run_id": rid,
                "domain": self.domain,
                "started_at": manifest["created_at"],
                "updated_at": manifest["created_at"],
                "stages": {},
            },
        )
        logger.info(f"📋 Started run {rid} in {run_dir}")
        return RunContext(
            run_id=rid,
            domain=self.domain,
            run_dir=run_dir,
            manifest=manifest,
            config=self.config,
        )

    def resume_run(self, run_id: str) -> RunContext:
        """Reconstruct a RunContext from an existing on-disk run."""
        run_dir = self.runs_root / run_id
        manifest_path = run_dir / self.MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No manifest at {manifest_path} — cannot resume")
        manifest = json.loads(manifest_path.read_text())
        cfg = manifest.get("config", {})

        ctx = RunContext(
            run_id=run_id,
            domain=manifest.get("domain", self.domain),
            run_dir=run_dir,
            manifest=manifest,
            config=cfg or self.config,
        )
        # Restore artifacts of any previously completed stage.
        status = self._read_status(run_dir)
        for sname, sdata in status.get("stages", {}).items():
            if sdata.get("status") == StageStatus.COMPLETED.value:
                ctx.stage_outputs[sname] = sdata.get("artifacts", {})
        logger.info(f"♻️  Resuming run {run_id} (completed: {sorted(ctx.stage_outputs)})")
        return ctx

    # ── Execution ─────────────────────────────────────────────
    def execute(self, stages: Iterable[Stage], ctx: RunContext) -> dict[str, Any]:
        """Run stages in order, persisting state after each transition.

        Returns a report dict with the final pipeline status, list of
        stages executed (or skipped/failed), and a halt reason if any
        gate or stage failed.
        """
        executed: list[dict[str, Any]] = []
        halt_reason: str | None = None

        for stage in stages:
            current_key = stage.key_for(ctx)
            recorded = self._read_stage_record(ctx.run_dir, stage.name)

            # ── Resume: previously completed AND inputs unchanged ──
            if (
                recorded
                and recorded.get("status") == StageStatus.COMPLETED.value
                and recorded.get("key") == current_key
            ):
                logger.info(f"⏭️  {stage.name}: resumed (key matches)")
                ctx.stage_outputs[stage.name] = recorded.get("artifacts", {})
                executed.append({
                    "name": stage.name,
                    "status": StageStatus.SKIPPED.value,
                    "key": current_key,
                })
                continue

            # ── Run fresh ──
            self._mark_stage(ctx.run_dir, stage.name, {
                "status": StageStatus.RUNNING.value,
                "key": current_key,
                "started_at": self._now().isoformat(timespec="seconds"),
            })
            started = time.monotonic()
            try:
                result = stage.fn(ctx) or StageResult()
            except Exception as exc:
                tb = traceback.format_exc(limit=8)
                logger.error(f"❌ {stage.name}: raised {type(exc).__name__}: {exc}")
                self._mark_stage(ctx.run_dir, stage.name, {
                    "status": StageStatus.FAILED.value,
                    "key": current_key,
                    "started_at": self._read_stage_record(ctx.run_dir, stage.name).get("started_at"),
                    "ended_at": self._now().isoformat(timespec="seconds"),
                    "duration_s": round(time.monotonic() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": tb,
                })
                executed.append({
                    "name": stage.name,
                    "status": StageStatus.FAILED.value,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                halt_reason = f"{stage.name} raised {type(exc).__name__}"
                break

            duration = round(time.monotonic() - started, 3)
            ended_at = self._now().isoformat(timespec="seconds")

            # Stage explicitly reports failure via result.status.
            if result.status == StageStatus.FAILED:
                logger.error(f"❌ {stage.name}: returned FAILED ({result.error})")
                self._mark_stage(ctx.run_dir, stage.name, {
                    "status": StageStatus.FAILED.value,
                    "key": current_key,
                    "started_at": self._read_stage_record(ctx.run_dir, stage.name).get("started_at"),
                    "ended_at": ended_at,
                    "duration_s": duration,
                    "artifacts": result.artifacts,
                    "error": result.error,
                })
                executed.append({
                    "name": stage.name,
                    "status": StageStatus.FAILED.value,
                    "error": result.error,
                })
                halt_reason = f"{stage.name} returned FAILED"
                break

            # Successful execution — record artifacts + key.
            self._mark_stage(ctx.run_dir, stage.name, {
                "status": StageStatus.COMPLETED.value,
                "key": current_key,
                "started_at": self._read_stage_record(ctx.run_dir, stage.name).get("started_at"),
                "ended_at": ended_at,
                "duration_s": duration,
                "artifacts": result.artifacts,
                "gate_passed": result.gate_passed,
            })
            ctx.stage_outputs[stage.name] = result.artifacts
            executed.append({
                "name": stage.name,
                "status": StageStatus.COMPLETED.value,
                "duration_s": duration,
                "gate_passed": result.gate_passed,
            })
            logger.info(f"✅ {stage.name}: completed in {duration}s (gate={'✓' if result.gate_passed else '✗'})")

            # Hard gate: a passed stage that fails its gate halts the rest.
            if not result.gate_passed:
                halt_reason = f"{stage.name} failed its hard gate"
                logger.warning(f"🛑 Halting pipeline: {halt_reason}")
                break

        report = {
            "run_id": ctx.run_id,
            "domain": ctx.domain,
            "stages": executed,
            "halt_reason": halt_reason,
            "status": StageStatus.FAILED.value if halt_reason else StageStatus.COMPLETED.value,
            "finished_at": self._now().isoformat(timespec="seconds"),
        }
        _atomic_write_json(ctx.run_dir / "report.json", report)
        # A5: mirror the run report into the SQLite RunStore so
        # observability surfaces (cost / SLO / canary) can query
        # relationally without parsing the per-run JSONL trees.
        # Best-effort — a persistence hiccup must not fail the run.
        try:
            from app.persistence import default_store
            default_store().upsert_run(ctx.run_id, {
                "run_id": ctx.run_id,
                "domain": ctx.domain,
                "manifest": ctx.manifest,
                "report": report,
            })
        except Exception as exc:
            logger.warning(f"⚠️  RunStore mirror failed: {exc}")
        return report

    # ── Internals ─────────────────────────────────────────────
    def _generate_run_id(self) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", self.domain.lower()).strip("-") or "run"
        return f"{self._now().strftime('%Y%m%d-%H%M%S')}-{slug}"

    def _read_status(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / self.STATUS_FILE
        if not path.is_file():
            return {"stages": {}}
        return json.loads(path.read_text())

    def _read_stage_record(self, run_dir: Path, name: str) -> dict[str, Any]:
        return self._read_status(run_dir).get("stages", {}).get(name, {})

    def _mark_stage(self, run_dir: Path, name: str, record: dict[str, Any]) -> None:
        status = self._read_status(run_dir)
        stages = status.setdefault("stages", {})
        existing = stages.get(name, {})
        existing.update(record)
        stages[name] = existing
        status["updated_at"] = self._now().isoformat(timespec="seconds")
        _atomic_write_json(run_dir / self.STATUS_FILE, status)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write-then-rename so partial writes never become visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, default=str) + "\n"
    with tmp.open("w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some filesystems / sandboxes refuse fsync — non-fatal.
            pass
    os.replace(tmp, path)


def _hash_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()
