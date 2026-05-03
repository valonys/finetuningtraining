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

## Sprint 03 — A1: GGUF export (active)

**Goal:** Close the only "critical gap" the blueprint flags. Trained adapters become deployable artifacts that any llama.cpp / Ollama runtime can consume.

**Deliverables**
- `app/trainers/export.py` — `merge_and_export(base_model_id, adapter_path, output_dir, quant="Q4_K_M", llama_cpp_path=...) -> str`
- `scripts/export_gguf.py` — operator CLI wrapper around the above
- `scripts/install_llamacpp.sh` — vendored install/update step for llama.cpp's `convert_hf_to_gguf.py`
- `tests/test_export_gguf.py` — round-trip smoke: merge → export → load via `app/inference/llamacpp_backend.py` → answer one prompt
- Sidecar: `outputs/<domain>/artifacts/<model>.metadata.json` with `{base_model_id, adapter_sha256, quant, file_sha256, exported_at}`
- README section: "Exporting an adapter to GGUF"

**Acceptance gate**
- Smoke test passes on CPU (no GPU required).
- A pre-existing adapter from `outputs/` round-trips end to end.
- Rollback pointer (symlink or manifest entry) to the prior stable artifact survives a re-export.

**Sizing:** 1–2 days.

**Risks**
- llama.cpp's `convert_hf_to_gguf.py` evolves quickly — pin a known-good commit in `install_llamacpp.sh`.
- Disk pressure during merge (full base model materialized briefly) — gate behind a free-space check.

---

## Sprint 04 — A2: Batch pipeline

**Goal:** Unattended `collect → forge → train → eval → deploy` with crash recovery.

**Deliverables**
- `app/pipeline/runner.py` — stage state machine with idempotency keys, write-then-rename for `stage_status.json`
- `scripts/batch_pipeline.py` — orchestrator entrypoint
- `tests/test_batch_pipeline.py` — happy-path + mid-stage-crash resume
- Run artifacts under `outputs/runs/<run_id>/{manifest.json, stage_status.json}`

**Acceptance gate:** full unattended run reproducible from manifest; crash mid-stage resumes from last committed checkpoint; promotion blocked on any failed hard gate.

**Sizing:** 3–4 days. Depends on A1 (uses `merge_and_export` in the deploy stage).

---

## Sprint 05 — A3: Model registry + promote/rollback

**Goal:** Every deployed model is traceable and rollback-safe by version id.

**Deliverables**
- `app/registry/schemas.py` — Pydantic models for the version record
- `app/registry/model_registry.py` — JSONL-backed registry first; DB-backed deferred to A6
- New endpoints in `app/main.py`: `GET /v1/registry`, `POST /v1/registry/promote`, `POST /v1/registry/rollback`
- `tests/test_model_registry.py` — CRUD + promotion lifecycle + rollback
- Persistent state: `outputs/registry/{model_versions.jsonl, promotion_events.jsonl}`

**Schema fields (minimum):** `model_version`, `base_model_id`, `adapter_path`, `dataset_manifest_path`, `eval_report_path`, `artifact_sha256`, `status` ∈ {candidate, staging, production, rolled_back}, `promoted_from`.

**Acceptance gate:** every deployed model maps to dataset manifest + eval report + artifact hash; rollback by version id works.

**Sizing:** 3 days. Depends on A1 + A2.

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
