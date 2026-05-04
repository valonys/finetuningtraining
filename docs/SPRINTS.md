# ValonyLabs Studio — Sprint Plan

**Last updated:** 2026-05-04
**Lane:** A (production autoregressive). Diffusion / dLLM lane out of scope.
**Strategy spec:** `docs/BLUEPRINT_ADOPTION_GUIDE.md` on `origin/main` at commit `fb75bc1` (PR #13). Not merged into `develop` by design — this file is the local execution surface, the remote doc is the strategy reference.
**Execution backlog:** `docs/BLUEPRINT_EXECUTION_BACKLOG.md` on the same remote commit.

## Sequence

```
Sprint 03 → A1     GGUF export                                  ✅ shipped
Sprint 04 → A2     Batch pipeline (collect→deploy)              ✅ shipped
Sprint 05 → A3     Model registry + promote/rollback            ✅ shipped
Sprint 06 →        Hardening hotfix                             [next, active queue]
Sprint 07 →        Frontend production + App Runner serving
Sprint 08 → A5     Inference hardening + cost/SLO/canary
Sprint 09 → A6a    JWT auth + tenant context
Sprint 10 → A6b    Tenant-scoped persistence + audit
```

**Re-slicing rationale (2026-05-04):** an external code review surfaced two
tracks the blueprint never covered — frontend production readiness and
targeted security hardening (path traversal, wildcard CORS, in-memory
job state). Sprints 06–07 are inserted to address them before A5 so we
don't pile features onto an unhardened API or an unbuildable frontend.
A5 pulls in job/run persistence (SQLite) so A6's Postgres switch
becomes a *driver swap*, not a new layer.

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

**Test execution status:** All 10 cases verified green during the A3 session (full suite 273/273 once macOS TCC restored Python's Documents access). The hand-review-only caveat from the original A2 commit no longer applies.

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

## Sprint 06 — Hardening hotfix (shipped 2026-05-04)

**Goal:** Close the security and build-correctness gaps surfaced by the 2026-05-04 external code review *before* exposing more API surface in A5 or attempting an MVP deploy.

**Status:** Shipped. Backend: 20 new unit tests green; full Python suite 293/293, no regressions; FastAPI loads cleanly with the new CORS resolver. Frontend: code-level fixes applied (build verification waits for next CI run since `npm` isn't available in this session — the CI job we added will gate it).

**Source of findings:** external readiness audit on `develop` HEAD `573eda5`.

**Shipped — security**
- `app/security/paths.py` — `validated_path(raw, *, allowlist_roots=None, must_exist=False)` resolves the path via `Path.resolve(strict=False)` then verifies it lives under one of the allowlist roots. Symlink escapes are caught because we resolve *before* checking. Default allowlist reads `VALONY_UPLOADS_DIR` / `VALONY_PROCESSED_DIR` / `VALONY_OUTPUTS_DIR` fresh on each call so test monkeypatches work.
- `app/security/cors.py` — `resolve_cors_origins()` returns the env-driven allowlist, falling back to `http://localhost:5173,http://127.0.0.1:5173` when `VALONY_CORS_ORIGINS` is unset. Empty / whitespace-only env also falls back (an empty list would block ALL origins, breaking dev silently).
- `app/security/__init__.py` — public re-exports.
- `app/main.py` — wildcard CORS replaced; `_cors_origins` logged at boot. New `_validated_paths(paths)` helper used by `/v1/forge/ingest` and `/v1/forge/build_dataset`. `/v1/forge/harvest/code` validates `req.path` (must_exist=True) and `req.output_dir` directly. All path rejections surface as 422 with a clear actionable message.
- `app/pipeline/stages.py` — `deploy_fn` validates `output_dir` before `merge_and_export` writes any bytes; failure becomes a clean `StageResult(status=FAILED)` with the rejection reason.

**Shipped — frontend build**
- `frontend/src/vite-env.d.ts` — declares `ImportMetaEnv` with `VITE_API_URL?: string`. Closes the build error on `import.meta.env` access in `frontend/src/api.ts`.
- `frontend/src/components/ChatWidget.tsx` — removed the unused `CloseIcon` definition (the source of one of the two TS strict-mode build failures).
- `frontend/package.json` — added `eslint`, `@eslint/js`, `typescript-eslint` to devDeps. Updated `lint` script to drop the eslint-9-incompatible `--ext` flag.
- `frontend/eslint.config.js` — flat-config setup (eslint 9), errors-only ruleset. `no-unused-vars` enabled with `^_` ignore convention. Permissive on `no-explicit-any` since the streaming meta blobs intentionally use it.

**Shipped — CI**
- `.github/workflows/ci.yml` — new `frontend` job: `npm ci` → `npm run lint` → `npm run build` on Node 20 with npm cache. Runs in parallel with the existing `smoke-tests` matrix; both must pass before merge.

**Shipped — tests**
- `tests/test_path_validator.py` (11 cases) — every default root accepts; root-itself accepts; `..` traversal rejects; absolute outside-allowlist rejects; symlink escape rejects (skipped on platforms without symlinks); empty path rejects; `must_exist` raises FileNotFoundError; explicit allowlist override works; default allowlist re-reads env on each call; relative path resolves against cwd.
- `tests/test_cors_config.py` (9 cases) — unset / empty / whitespace env all fall back to dev defaults; single + multiple origins parsed; whitespace around entries trimmed; empty entries dropped; all-empty-after-trim falls back; the dev-defaults list is fresh per call (mutation can't poison a later call).

**Acceptance gate**
- ✅ Path-accepting mutating endpoints reject out-of-allowlist paths (`/v1/forge/ingest`, `/v1/forge/build_dataset`, `/v1/forge/harvest/code`, deploy stage `output_dir`).
- ✅ Wildcard CORS gone; `allow_origins` resolves from env. Verified at FastAPI boot.
- ⏳ `npm run build` succeeds — pending first CI run (this session can't invoke npm).
- ✅ CI gate added — `frontend` job blocks on lint or build failure.

**Out of scope** (deferred to their proper sprints): auth (A6), tenant scoping (A6), persistence layer (A5), live pipeline stage wiring (A2 follow-up), inference manager → registry-PRODUCTION-only (A5).

**Risks / open**
- ⏳ The CI `frontend` job's first run will tell us if there are additional lint warnings in existing code that the new eslint config flags. If so, expect a small follow-up to either fix them or relax specific rules. Permissive baseline (no `no-explicit-any`, allow `_`-prefixed unused) should keep this minimal.

---

## Sprint 07 — Frontend production + App Runner serving

**Goal:** Make the frontend deployable and stand up the MVP demo lane on AWS App Runner. Promotes the previously "opportunistic" App Runner sprint to a real slot now that S06 makes the build green and the API safe to expose.

**App Runner deprecation note (2026-05-04):** AWS closed App Runner to *new customers* effective 2026-04-30. **This account already has App Runner services running and is grandfathered**, so we can keep deploying to it. Decision made with full awareness of the deprecation: user explicitly chose to reuse existing well-understood App Runner over (a) Fly.io (rejected — second cloud to manage) and (b) ECS Express Mode (deferred — fixed-cost ALB baseline risks blowing the $5/mo cap; revisit only when scale justifies the haywire). When App Runner reaches end-of-life or scale demands more, the natural exit is ECS Express, not Fly.io.

**Budget target:** $0–$5/mo (App Runner Auto Pause idle ≈ $3.30/mo provisioned-memory-only; active vCPU billed per request). Hard ceiling — see memory `serving_cost_constraints.md`.

**Deliverables — frontend serving model**
- **Decision** (record in PR): FastAPI `StaticFiles` mount at `/` with SPA fallback (`/{full_path:path}` → `index.html`) **vs** S3 + CloudFront for the SPA + API on a separate origin.
  - Default lean: **separate origins** (S3/CloudFront for SPA, App Runner for API). Cheaper, scales independently, lets the static side stay free-tier; App Runner stays single-purpose.
  - StaticFiles mount path is the fallback if you'd rather have one container handle both (simpler ops, slightly more compute per request).
- Implement chosen path. If separate origins: keep the existing Vite build, add an `npm run build && aws s3 sync dist/ s3://...` script. If StaticFiles: mount in `app/main.py` with the SPA fallback route.
- **Dockerfile harden:** the current image copies `frontend/dist` but uvicorn doesn't serve it (S06 audit item 3). Either drop the copy (separate-origins path) or wire the StaticFiles mount.

**Deliverables — App Runner**
- `apprunner.yaml` at repo root — `runtime: python311`, build command, start command (`uvicorn app.main:app --host 0.0.0.0 --port 8000`), 0.25 vCPU / 0.5 GB, Auto Pause on.
- Slimmed CPU-only Docker target (`Dockerfile.apprunner` or a new stage on the existing Dockerfile) — drop `torch`/`unsloth`/`vllm` extras, keep only what's needed for `VALONY_INFERENCE_BACKEND=ollama`.
- Reuse the existing grandfathered App Runner service rather than creating a new one (new-customer creation is closed).
- IaC bits: `infra/apprunner-service.tf` or a one-shot bootstrap script (decide based on user pref).

**Hard constraints**
- App Runner has no GPU and no room for an in-process model. Generation MUST proxy to Ollama Cloud / OpenRouter / HF Inference. Never load a HF/vLLM model in this image.
- `VALONY_CORS_ORIGINS` (from S06) must include the SPA origin (CloudFront URL or App Runner static path).
- Cost cap is hard, not aspirational. First week of real demo traffic must be checked against actual billing — if creeping above $5/mo, throttle traffic or shrink the instance before considering scope changes.

**Acceptance gate**
- ✅ `frontend/dist` served correctly in production (whichever serving model is chosen) — manual smoke + a CI check that fetches `/` from a built container.
- ✅ End-to-end `curl https://<apprunner>/healthz` and SPA loads in a browser.
- ✅ Monthly bill under $5 for a week of pitch-deck-volume traffic.

**Exit ramp (when triggered, not in S07 scope)**
- App Runner end-of-life announcement OR sustained traffic that warrants scaling beyond Auto Pause → migrate to **Amazon ECS Express Mode** (AWS-blessed successor; ALB baseline acceptable when traffic justifies it).
- Fly.io is *not* on the exit ramp — user has explicitly ruled out spinning up a second cloud.

**Sizing:** 1 sprint. Frontend serving model decision is the only non-mechanical step.

---

## Sprint 08 — A5: Inference hardening + cost/SLO/canary

**Goal:** Defend runtime quality and unit economics during rollouts.

**Re-scoped (2026-05-04):** pulls in job + run persistence (SQLite-backed) so A5's SLO/cost/canary surfaces have a durable place to read/write from. The current `job_registry` dict in `app/main.py` loses state on restart and diverges across workers — A5 closes that without waiting for A6's Postgres switch.

**Deliverables**
- `app/persistence/store.py` — abstract `JobStore` + `RunStore` interfaces with a SQLite default backend. Schema migrations in `app/persistence/migrations/`. `app/main.py` switches `job_registry` to `JobStore`; `app/pipeline/runner.py` mirrors run state into `RunStore` alongside the JSONL.
- `app/cache/semantic.py` — Redis-backed semantic cache with in-memory fallback (env-gated)
- `app/observability/cost.py` — per-request token/cost accounting
- `app/observability/slo.py` — latency / error-rate / quality-probe evaluator
- Canary % routing in `app/inference/manager.py` with auto-abort triggers
- `app/inference/manager.py` adapter resolution switches from "scan `outputs/<domain>/`" to "load the registry's PRODUCTION row" (closes the A3 follow-up loop).
- `tests/test_semantic_cache.py`, `tests/test_persistence.py`, `tests/test_canary_routing.py`
- Metric artifacts: `outputs/metrics/{slo,cost,canary}_<timestamp>.json`

**Auto-abort triggers:** latency p95 > threshold, error rate spike > threshold, quality probe fail.

**Acceptance gate:** equal or better quality at lower or stable latency/cost in a canary window; jobs survive a `uvicorn` restart.

**Sizing:** 5–6 days (4–5 base + 1 for persistence). Depends on A3 (canary routes between registered model versions) and S06 (CORS / path validators in place before exposing cost/SLO endpoints).

---

## Sprint 09 — A6a: JWT auth + tenant context

**Deliverables**
- `app/auth/jwt.py` — token validation, claims extraction
- `app/auth/middleware.py` — FastAPI middleware that attaches `tenant_id` to request state
- `tests/test_auth_jwt.py`
- All existing endpoints accept (and require, except `/healthz`) a valid token. Includes the new registry endpoints from A3 (`/v1/registry/promote`, `/v1/registry/rollback`) and S08's cost/SLO surfaces.

**Acceptance gate:** every protected endpoint rejects missing/invalid tokens; tenant id is available to handlers.

**Sizing:** 3 days.

---

## Sprint 10 — A6b: Tenant-scoped persistence + audit

**Deliverables**
- `app/persistence/postgres.py` — Postgres / pgvector backend implementing the `JobStore` + `RunStore` interfaces from S08. **Driver swap, not a new layer** — A5's SQLite is the dev/MVP default, A6 makes Postgres the production option.
- `app/registry/postgres_backend.py` — Postgres backend for `ModelRegistry` (replaces the JSONL files for production deploys; JSONL stays as the dev default).
- `app/memory/store.py` — pgvector tenant-scoped vector memory.
- `app/audit/logging.py` — append-only audit events tagged with model version + user/tenant.
- `tests/test_tenant_isolation.py` — tenant A cannot read tenant B's records.
- Audit artifacts: `outputs/audit/events_<date>.jsonl` (file backend) or audit table (Postgres backend).

**Acceptance gate:** tenant isolation verified in automated tests; auditable request trace exists for every chat/training/promote/rollback action; A5's job/run persistence and A3's registry both serve from Postgres in the production profile.

**Sizing:** 5–7 days. Heaviest sprint — Postgres provisioning + migration story land here. Lighter than originally scoped because the *interfaces* already exist (S08 created them); A6 only swaps the backend.

---

## A2 follow-up — live pipeline stage wiring

**Goal:** Replace the `collect / forge / train` stubs in `app/pipeline/stages.py` with real wiring to the existing modules so `scripts/batch_pipeline.py --domain X` runs end-to-end without TODOs.

**Deliverables**
- `collect_fn` — route to YouTube / arXiv / code harvesters per `ctx.config['collect']`. Schema decision: union-tagged dict per harvester type.
- `forge_fn` — call `app.data_forge.build_dataset(...)` with collect outputs. Surface samples count + dataset path.
- `train_fn` — instantiate the right trainer class per `ctx.config['method']` (SFT/DPO/ORPO/KTO/GRPO), call `.train()`, surface adapter path + final loss.
- `configs/domains/*.yaml` schema additions for the new collect/forge/train knobs.
- `tests/test_pipeline_stages_live.py` — integration-style tests with monkeypatched harvesters / data_forge / trainers.

**Sizing:** 1–2 sprints' worth, splittable into one PR per stage. Not on the critical path for S06–S10 — slot whenever a real pipeline run is needed.

---

## Definition of done (program-level)

The Lane A adoption is complete when:

1. A1 — adapters export to GGUF and round-trip in smoke tests; rollback pointer maintained.
2. A2 — full pipeline runs unattended and recovers from mid-stage failure (live stages wired, no TODO stubs).
3. A3 — every promotion requires explicit quality + lineage gates; rollback by id works.
4. S06 — path traversal closed; CORS allowlisted; frontend builds in CI.
5. S07 — frontend served correctly in production; App Runner demo lane reachable under $5/mo.
6. A5 — canary rollout + cost/SLO controls protect runtime reliability and economics; jobs/runs durable.
7. A6 — tenant isolation enforced; auditable traces tagged with model version and user/tenant; Postgres backend live.
8. `docs/ARCHITECTURE_AND_ROADMAP.md` synced with the implemented state.
