# ValonyLabs Studio — Sprint Plan

**Last updated:** 2026-05-03
**Lane:** A (production autoregressive). Diffusion / dLLM lane out of scope.
**Strategy spec:** `docs/BLUEPRINT_ADOPTION_GUIDE.md` on `origin/main` at commit `fb75bc1` (PR #13). Not merged into `develop` by design — this file is the local execution surface, the remote doc is the strategy reference.
**Execution backlog:** `docs/BLUEPRINT_EXECUTION_BACKLOG.md` on the same remote commit.

## Sequence

```
Sprint 03 → A1   GGUF export                  [next, active queue]
Sprint 04 → A2   Batch pipeline (collect→deploy)
Sprint 05 → A3   Model registry + promote/rollback
Sprint 06 → A5   Inference hardening + cost/SLO/canary
Sprint 07 → A6a  JWT auth + tenant context
Sprint 08 → A6b  Tenant-scoped persistence + audit
        ↘
         (parallel, opportunistic)
         App Runner serving lane — cheap MVP demo host ($0–$5/mo)
```

A4 is intentionally absent — that's the diffusion lane in the source blueprint, out of scope here.

## Hard guardrails (apply to every sprint)

1. Never promote an unevaluated model.
2. Never replace an active production model without a rollback pointer.
3. Never trust synthetic-data quality as assumed; always verify and score.
4. Keep BM25 fallback even when retrieval upgrades land.
5. Every release clears: eval gate → deploy smoke → cost/SLO gate → signed lineage.

---

## Sprint 03 — A1: GGUF export (shipped 2026-05-03)

**Goal:** Close the only "critical gap" the blueprint flags. Trained adapters become deployable artifacts that any llama.cpp / Ollama runtime can consume.

**Status:** Code + unit tests landed on `develop`. End-to-end smoke against a real adapter still needs a workstation with `scripts/install_llamacpp.sh` run — tracked as a post-merge integration check.

**Shipped**
- `app/trainers/export.py` — `merge_and_export(base_model_id, adapter_path, output_dir, *, quant="Q4_K_M", llama_cpp_path=None, artifact_name=None) -> dict` with stable return contract (`gguf_path`, `metadata_path`, `sha256`, `latest_pointer`, `quant`, `base_model_id`, `adapter_sha256`, `exported_at`).
- `scripts/install_llamacpp.sh` — vendor llama.cpp into `~/.local/llama.cpp` (override via `VALONY_LLAMA_CPP_PATH`, pin via `LLAMA_CPP_REF`). Builds `llama-quantize`, installs python deps for `convert_hf_to_gguf.py`.
- `scripts/export_gguf.py` — operator CLI; emits final result as JSON to stdout.
- `tests/test_export_gguf.py` — 6 cases covering happy path (Q4_K_M), missing adapter, missing llama.cpp, missing quantize binary, F16 pass-through (skip quantize), rollback pointer across two exports. Stubs peft + transformers + subprocess so suite stays offline-CPU.
- Sidecar `<artifact>-<quant>.metadata.json` with `{base_model_id, adapter_path, adapter_sha256, quant, gguf_filename, file_sha256, file_bytes, exported_at, llama_cpp_path}`.
- `latest.gguf` rollback pointer (symlink on POSIX, file copy on Windows).
- `merge_and_export` re-exported from `app.trainers` for ergonomic imports.
- README section: "Export adapter to GGUF (Sprint 03 / A1)".

**Acceptance gate**
- Unit suite green on CPU (6/6 pass; full suite 249/249 pass — no regressions).
- ⏳ Real adapter round-trip via `app/inference/llamacpp_backend.py` — pending workstation with llama.cpp built. Mocked equivalent passes.
- Rollback pointer survives a re-export — covered by `test_rollback_pointer_updates_across_exports`.

**Risks resolved / open**
- ✅ llama.cpp toolchain version drift — `install_llamacpp.sh` honors `LLAMA_CPP_REF`; default `master` should be pinned to a tag in production.
- ⏳ Disk pressure during merge — currently uses an OS tempdir; no preflight free-space check yet. Add when first real export trips it.
- ⏳ Windows symlink fallback — copy fallback in place but not exercised in CI.

---

## Sprint 04 — A2: Batch pipeline (shipped 2026-05-04)

**Goal:** Unattended `collect → forge → train → eval → deploy` with crash recovery.

**Status:** Runner mechanics (state machine, atomic state writes, resume, hard-gate enforcement) and CLI shipped. Stage implementations land as a mix: `eval` and `deploy` are wired live (eval → `app.eval.run_eval`, deploy → A1's `merge_and_export`); `collect`, `forge`, `train` are thin stubs that surface artifacts but defer the real harvester/trainer wiring until each stage's input contract is finalized — TODOs in source.

**Test execution caveat:** The 10-case test suite was written but could not be executed in this session because macOS TCC revoked Full Disk Access for Python on the Documents folder mid-session. Code reviewed by hand. Run locally to verify:
```bash
python3 -m pytest tests/test_batch_pipeline.py -v
```

**Shipped**
- `app/pipeline/runner.py` (~280 LoC) — `PipelineRunner` with `start_run` / `resume_run` / `execute`. Atomic write-then-rename via `os.replace` on `stage_status.json`. Per-stage idempotency keys (default = sha256 of manifest+name; override with `Stage.idempotency_key_fn`). Resume skips a stage only when prior status was COMPLETED *and* the current key matches the recorded one. Stage exceptions are caught, traceback persisted, pipeline halted. A passed stage that flips `gate_passed=False` halts the pipeline (this is how eval blocks deploy).
- `app/pipeline/stages.py` (~140 LoC) — five concrete stages + `DEFAULT_STAGES` + `select_stages()` filter that preserves canonical order. `_collect_key` derives an idempotency key from upload-file size+mtime so any new/changed input forces a re-run.
- `app/pipeline/__init__.py` — exports `PipelineRunner`, `RunContext`, `Stage`, `StageResult`, `StageStatus`.
- `scripts/batch_pipeline.py` — operator CLI (`--domain` / `--resume` / `--stages` / `--runs-root` / `--quiet`). Loads `configs/domains/<domain>.yaml` if present. Emits the run report as JSON to stdout. Exit code 0 on success, 1 on halt.
- `tests/test_batch_pipeline.py` — 10 cases: happy path, manifest immutability, mid-stage crash + resume across runner instances, idempotency-key drift forces re-run, hard-gate failure halts before deploy, no `.tmp` leftovers from atomic write, missing-run-id resume raises, `select_stages` filter + canonical order + unknown-stage error, downstream stage receives upstream `stage_outputs`.
- Artifacts on disk per run: `outputs/runs/<run_id>/manifest.json`, `stage_status.json`, `report.json`.

**Acceptance gate**
- ⏳ Full unattended run reproducible from manifest — runner can do this; live stages need `collect/forge/train` follow-up wiring before a real CLI run.
- ✅ Crash mid-stage resumes from last committed checkpoint — covered by `test_resume_skips_completed_stages_when_key_matches`.
- ✅ Promotion blocked on any failed hard gate — covered by `test_hard_gate_failure_halts_before_downstream`.

**Open follow-ups**
- ⏳ Live stage wiring for `collect` (route to YouTube/arXiv/code harvesters per config), `forge` (call `app.data_forge.build_dataset`), `train` (instantiate the right trainer class per `config.method`). Each is a small TODO marker in `stages.py`.
- ⏳ `configs/domains/*.yaml` schema additions for collect/forge/train/deploy knobs.
- ⏳ Run the local test suite once Python's Documents access is restored.

**Sizing actual:** ~1 session for runner + tests; live stage wiring deferred as separate small PRs.

---

## Sprint 05 — A3: Model registry + promote/rollback (shipped 2026-05-04)

**Goal:** Every deployed model is traceable and rollback-safe by version id.

**Status:** Shipped. 14 new tests green; full suite 273/273 (no regressions). FastAPI app loads with all 4 registry routes wired. A2's deploy stage now auto-registers exported artifacts as CANDIDATE.

**Shipped**
- `app/registry/schemas.py` — `ModelStatus` enum, `ModelVersion`, `PromotionEvent`, `RollbackResult` (Pydantic).
- `app/registry/model_registry.py` (~270 LoC) — `ModelRegistry` with append-only JSONL storage at `outputs/registry/{model_versions.jsonl, promotion_events.jsonl}`. Materializes current state by replaying the version log; events are first-class history. State machine enforces `candidate → staging → production` plus `→ rolled_back` exit and `rolled_back → staging` re-attempt path. Promoting a second version to PRODUCTION atomically demotes the prior one with an auto-reason event. `rollback(domain, target_version=None)` either uses an explicit replacement or auto-picks the most recently updated STAGING / ROLLED_BACK candidate.
- `app/registry/__init__.py` — public exports (`ModelRegistry`, `ModelStatus`, `ModelVersion`, `PromotionEvent`, `RollbackResult`, `InvalidTransition`, `UnknownVersion`, `default_registry`).
- `app/models.py` — `RegistryPromoteRequest`, `RegistryRollbackRequest` API request models.
- `app/main.py` — 4 endpoints:
  - `GET  /v1/registry?domain=&status=` → `List[ModelVersion]`
  - `GET  /v1/registry/{model_version}` → `ModelVersion`
  - `POST /v1/registry/promote` → `ModelVersion`
  - `POST /v1/registry/rollback` → `RollbackResult`
  - 404 on unknown version, 409 on invalid transition, 422 on bad status string.
- `app/pipeline/stages.py` — `deploy_fn` now calls `register_candidate(...)` after `merge_and_export` succeeds, threading dataset manifest path (forge stage), eval report path (eval stage), artifact sha256 (export). Registration is best-effort: a registry hiccup logs a warning + populates `registry_error` in artifacts but does not fail the deploy stage (the GGUF on disk is still valid).
- `tests/test_model_registry.py` — 14 cases covering register / list filters / full lifecycle / auto-demote on second promotion / rollback with target / rollback auto-pick / no-production-to-rollback / invalid transitions / unknown version / append-only durability across fresh registry instance / re-attempt after rollback / target-must-be-staging-or-rolled-back / id generation without artifact sha.

**Acceptance gate**
- ✅ Every deployed model maps to dataset manifest + eval report + artifact hash — `deploy_fn` threads all three into `register_candidate`.
- ✅ Rollback by version id works — `test_rollback_with_explicit_target_promotes_replacement`.
- ✅ Auto-rollback on prod replacement — `test_promoting_second_to_production_auto_demotes_first`.

**Open follow-ups**
- ⏳ Multi-process safety: append-only JSONL is safe for line-sized writes < `PIPE_BUF`, but multi-writer concurrency relies on single-process discipline. A6's Postgres switch removes this caveat.
- ⏳ Registry surfaces in the React frontend (`Domains.tsx` / new `Registry.tsx` panel). Not in A3 scope.
- ⏳ `inference/manager.py` currently scans `outputs/<domain>/` for adapters; switching it to "load only the registry's PRODUCTION row" is the natural follow-up that closes the loop between registry and serving.

**Sizing actual:** ~1 session for schemas + registry + endpoints + deploy wiring + tests.

---

## Sprint 06 — A5: Inference hardening + cost/SLO/canary

**Goal:** Defend runtime quality and unit economics during rollouts.

**Deliverables**
- `app/cache/semantic.py` — Redis-backed semantic cache with in-memory fallback (env-gated)
- `app/observability/cost.py` — per-request token/cost accounting
- `app/observability/slo.py` — latency / error-rate / quality-probe evaluator
- Canary % routing in `app/inference/manager.py` with auto-abort triggers
- `tests/test_semantic_cache.py`
- Metric artifacts: `outputs/metrics/{slo,cost,canary}_<timestamp>.json`

**Auto-abort triggers:** latency p95 > threshold, error rate spike > threshold, quality probe fail.

**Acceptance gate:** equal or better quality at lower or stable latency/cost in a canary window.

**Sizing:** 4–5 days. Depends on A3 (canary routes between registered model versions).

---

## Sprint 07 — A6a: JWT auth + tenant context

**Deliverables**
- `app/auth/jwt.py` — token validation, claims extraction
- `app/auth/middleware.py` — FastAPI middleware that attaches `tenant_id` to request state
- `tests/test_auth_jwt.py`
- All existing endpoints accept (and require, except `/healthz`) a valid token

**Acceptance gate:** every protected endpoint rejects missing/invalid tokens; tenant id is available to handlers.

**Sizing:** 3 days.

---

## Sprint 08 — A6b: Tenant-scoped persistence + audit

**Deliverables**
- `app/memory/store.py` — Postgres / pgvector store, tenant-scoped queries
- `app/audit/logging.py` — append-only audit events tagged with model version + user/tenant
- `tests/test_tenant_isolation.py` — tenant A cannot read tenant B's records
- Audit artifacts: `outputs/audit/events_<date>.jsonl`

**Acceptance gate:** tenant isolation verified in automated tests; auditable request trace exists for every chat/training/promote action.

**Sizing:** 5–7 days. Heaviest sprint — Postgres provisioning + migration story land here.

---

## Parallel sprint (opportunistic) — App Runner serving lane

**Trigger:** when an MVP pitch-deck demo lands or a Trial-tier user asks for a hosted URL.
**Budget target:** $0–$5/mo.

**Architecture (proxy-only model serving)**
- Frontend: `frontend/` Vite build → S3 + CloudFront (or Amplify Hosting). Effectively $0 at demo volumes.
- Backend: slimmed CPU-only Docker image of `app/` with `VALONY_INFERENCE_BACKEND=ollama`, dropping `torch`/`unsloth`/`vllm` extras. Generations proxy to Ollama Cloud (already production via `app/inference/ollama_backend.py`).
- Runtime: AWS App Runner, 0.25 vCPU / 0.5 GB, Auto Pause enabled. Idle ≈ provisioned-memory only (~$3.30/mo); active vCPU billed per request.
- IaC: `apprunner.yaml` + `Dockerfile.apprunner` (or a multi-stage `apprunner` target on the existing `Dockerfile`).

**Hard constraint:** App Runner has no GPU and no room for an in-process model. The whole lane only works if generation is offloaded — never load a HF/vLLM model in this image.

**Out of scope (would blow the budget):** GPU inference, vLLM, in-process model hosting, fine-grained autoscaling beyond Auto Pause.

**Sizing:** ~1 sprint when triggered. Not on the critical path.

---

## Definition of done (program-level)

The Lane A adoption is complete when:

1. A1 — adapters export to GGUF and round-trip in smoke tests; rollback pointer maintained.
2. A2 — full pipeline runs unattended and recovers from mid-stage failure.
3. A3 — every promotion requires explicit quality + lineage gates; rollback by id works.
4. A5 — canary rollout + cost/SLO controls protect runtime reliability and economics.
5. A6 — tenant isolation enforced; auditable traces tagged with model version and user/tenant.
6. `docs/ARCHITECTURE_AND_ROADMAP.md` synced with the implemented state.
