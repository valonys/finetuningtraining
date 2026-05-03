# ValonyLabs Studio Blueprint Adoption Guide (Lane A Production Systems)

**Last updated:** 2026-05-03  
**Scope:** Consolidated implementation guidance from **Blueprint fit summary** through **Sprint Runs** and **production sequencing**.  
**In-scope lane:** **Lane A only** (proven autoregressive production systems).  
**Out-of-scope lane:** Diffusion lane / experimental dLLM workflows.

---

## 1) Why this document exists

This file is the code-ready adoption blueprint for extending ValonyLabs Studio without leaving proven production paths. It unifies:

1. Current-state fit summary
2. End-to-end Sprint Runs
3. Production sequencing with priority path:
   - **A1 -> A2 -> A3 -> A5 -> A6**
4. Practical implementation guidance (modules, interfaces, artifacts, run gates)
5. Pros/cons and risk controls for each stage

Use this as the primary execution reference for roadmap and engineering implementation.

---

## 2) Blueprint fit summary (current-state alignment)

### 2.1 Architecture alignment snapshot

| Blueprint capability | Current status | Evidence in codebase | Adoption note |
|---|---|---|---|
| OOB + RAG-first strategy | **Present** | `app/rag/`, `app/main.py` docs mode | Keep default behavior |
| Nemotron synth teacher path | **Partially present** | `app/providers/ollama.py` (`nemotron-3-super` default in cloud mode) | Add first-class Nemotron provider abstraction for multi-endpoint teacher usage |
| SFT/DPO/ORPO/KTO/GRPO training stack | **Present** | `app/trainers/*_trainer.py` | Reuse as-is |
| RLVR pluggable reward architecture | **Present, expandable** | `app/trainers/reward_signals.py`, `grpo_trainer.py` | Add SQL + JSON verifiers |
| BM25 retriever | **Present** | `app/rag/retriever.py` | Keep as default until corpus scale threshold |
| GGUF export deployment path | **Missing (critical gap)** | No `app/trainers/export.py` | Highest priority closure (A1) |
| Batch pipeline orchestration | **Missing** | No full collect->forge->train->eval->deploy scheduler script | Add in A2 |
| Model registry + promotion control | **Missing** | No dedicated registry module | Add in A3 |
| Cost/ops guardrails | **Partial** | Basic health/latency surfaces | Expand in A5 |
| Enterprise auth + tenancy | **Missing** | No `app/auth/`, no tenant persistence layer | Add in A6 |

### 2.2 Strategic interpretation

- The platform already has strong middle-pipeline capabilities (forge/train/eval/inference).
- The weakest production points are lifecycle governance and deployment artifact flow:
  - exportability (GGUF),
  - automation (scheduled batch),
  - traceability (registry),
  - cost/reliability controls,
  - tenant security.

---

## 3) Lane A end-to-end blueprint (production-only)

```
Collect data
  -> Forge dataset (ingest, chunk, filter, synth)
  -> Train adapter (SFT/DPO/ORPO/KTO/GRPO)
  -> Evaluate with quality gate
  -> Package deploy artifact (GGUF where needed)
  -> Deploy (vLLM / Ollama)
  -> Observe (quality, latency, cost, failures)
  -> Retrain on schedule with promotion control
```

### Hard guardrails

- Never promote an unevaluated model.
- Never replace active production model without rollback pointer.
- Never treat synthetic-data quality as assumed; always verify and score.
- Keep BM25 fallback even when retrieval upgrades are introduced.

---

## 4) Sprint Runs (program-level)

These runs preserve the original blueprint flow while staying production-oriented.

## Sprint Run 0 — Baseline and eval hardening

**Objective:** Create trusted baseline before model changes.

**Scope**
- Build a canonical prompt set and scoring rubric.
- Benchmark current Nano vs Super behavior.
- Classify failures: knowledge-bound vs behavior-bound.

**Deliverables**
- Baseline report artifact under `outputs/evals/`.
- Failure taxonomy summary.

**Pros**
- Prevents blind optimization.
- Anchors all future regressions/improvements.

**Cons**
- Upfront effort before feature work.
- Requires discipline to maintain eval set freshness.

---

## Sprint Run 1 — OOB + RAG optimization

**Objective:** Maximize quality without training first.

**Scope**
- Keep BM25 path as baseline.
- Benchmark optional retrieval enhancements behind feature flags.

**Deliverables**
- Retrieval comparison report (quality/latency/cost).
- Decision rule for when to keep BM25 vs upgrade.

**Pros**
- Cheapest quality uplift path.
- Low operational risk.

**Cons**
- May plateau on behavior-format failures.

---

## Sprint Run 2 — Teacher-student synthetic data pipeline

**Objective:** Produce verified synthetic training sets using stronger teacher models.

**Scope**
- Add provider abstraction for Nemotron teacher endpoint(s).
- Generate candidate pairs.
- Filter with verifiable checks (start SQL execution checks first).

**Deliverables**
- Synthetic pair generation module and manifests.
- Acceptance/rejection metrics.

**Pros**
- High-quality data flywheel.
- Lower cost than large-model-only serving.

**Cons**
- Requires robust verifier logic to avoid synthetic drift.

---

## Sprint Run 3 — LoRA specialization on production-target student

**Objective:** Ship specialized adapters with controlled cost.

**Scope**
- Train LoRA adapters from verified data.
- Run quality gate and compare against baseline.

**Deliverables**
- Candidate adapter artifacts + eval reports.

**Pros**
- Fast adaptation loops.
- Good quality/cost trade-off.

**Cons**
- Needs strict lineage tracking to prevent unreproducible models.

---

## Sprint Run 4 — RLVR verifier expansion

**Objective:** Improve correctness on verifiable enterprise tasks.

**Scope**
- Extend rewards to SQL execution, JSON schema validation, code checks.
- Domain-configurable reward routing.

**Deliverables**
- New reward classes + test suite + run metrics.

**Pros**
- Directly optimizes correctness, not only style.
- Lower annotation burden than RLHF.

**Cons**
- Sandboxing and determinism complexity.

---

## Sprint Run 5 — Deployment closure and operational packaging

**Objective:** Close artifact portability gap.

**Scope**
- Implement GGUF export flow and deploy smoke tests.

**Deliverables**
- Export module, metadata, and rollback-safe deployment runbook.

**Pros**
- Removes major production blocker.

**Cons**
- Disk/compute heavy pipeline stage; needs resource guardrails.

---

## Sprint Run 6 — Cost and reliability optimization

**Objective:** Institutionalize intelligence-per-dollar operations.

**Scope**
- Cost accounting, canary controls, semantic caching, SLA dashboards.

**Deliverables**
- Per-run ROI report and promotion decision template.

**Pros**
- Long-term scalability and profitability.

**Cons**
- Additional infra complexity and monitoring overhead.

---

## 5) Production sequencing plan (execution order)

Primary release path:

1. **A1 — Deployment Closure (GGUF critical path)**
2. **A2 — Reliable Continuous Training**
3. **A3 — Model Registry + Promotion Control**
4. **A5 — Inference Hardening + Cost Controls**
5. **A6 — Enterprise Readiness (auth, tenancy, compliance)**

Rationale:
- A1-A3 stabilize release mechanics first.
- A5 optimizes runtime economics after release integrity is safe.
- A6 hardens multi-tenant enterprise operations once lifecycle control is mature.

---

## 6) Code-ready implementation guidance (A1 -> A2 -> A3 -> A5 -> A6)

## A1 — Deployment closure (GGUF)

### Target additions
- `app/trainers/export.py`
- `scripts/export_gguf.py` (optional CLI wrapper)
- `tests/test_export_gguf.py`

### Suggested API surface
```python
def merge_and_export(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    quant: str = "Q4_K_M",
    llama_cpp_path: str = "~/.local/llama.cpp",
) -> str:
    """Return path to final GGUF artifact."""
```

### Required artifacts
- `<model>.gguf`
- `<model>.metadata.json` containing:
  - base model id
  - adapter hash
  - quant scheme
  - checksum
  - export timestamp

### Exit gate
- Exported model can be loaded and queried in smoke test.
- Rollback pointer to last stable artifact exists.

---

## A2 — Reliable continuous training

### Target additions
- `scripts/batch_pipeline.py`
- `app/pipeline/runner.py` (if modularized)
- `tests/test_batch_pipeline.py`

### Pipeline contract
```text
collect -> forge -> train -> eval -> deploy
```

### Reliability controls
- Idempotency key per run.
- Resume on failure with stage-aware checkpointing.
- Promotion disabled on any failed hard gate.

### Exit gate
- Full run executes unattended with deterministic artifact manifest.

---

## A3 — Model registry + promotion control

### Target additions
- `app/registry/model_registry.py`
- `app/registry/schemas.py`
- `app/main.py` endpoints for list/promote/rollback
- `tests/test_model_registry.py`

### Minimal registry fields
- `model_version`
- `base_model_id`
- `adapter_path`
- `dataset_manifest_path`
- `eval_report_path`
- `status` (`candidate`, `staging`, `production`, `rolled_back`)
- `promoted_from` (lineage)

### Exit gate
- Every deployed model is traceable and rollbackable by version id.

---

## A5 — Inference hardening + cost controls

### Target additions
- `app/cache/semantic.py` (Redis-backed semantic cache)
- `app/observability/cost.py`
- `app/observability/slo.py`
- `tests/test_semantic_cache.py`

### Runtime controls
- Canary rollout routing by percentage.
- Regression auto-abort conditions:
  - latency threshold breach
  - error-rate spike
  - quality probe fail

### Exit gate
- Lower or stable cost/latency at equal or better quality.

---

## A6 — Enterprise readiness

### Target additions
- `app/auth/jwt.py`
- `app/auth/middleware.py`
- `app/memory/store.py` (PostgreSQL/pgvector)
- `app/audit/logging.py`
- `tests/test_auth_jwt.py`, `tests/test_tenant_isolation.py`

### Security requirements
- JWT validation and tenant extraction.
- Tenant-scoped persistence and query access control.
- Request/response audit events with model version tags.
- Optional PII redaction in ingestion path.

### Exit gate
- Tenant isolation verified in automated tests.
- Auditable request trace for compliance workflows.

---

## 7) Pros and cons summary (program-level)

### Pros
- Stays on proven production architecture.
- Leverages existing strong trainer/eval/inference investments.
- Prioritizes deployment integrity and governance before scale complexity.
- Aligns with intelligence-per-dollar strategy using measurable gates.

### Cons
- Requires upfront engineering in release plumbing (export/registry/automation).
- Operational sophistication increases (canarying, cache, observability, auth).
- Governance and compliance features can slow raw feature velocity if unmanaged.

---

## 8) Adoption checklist (execution ready)

- [ ] Baseline eval set created and versioned.
- [ ] A1 GGUF export path implemented and smoke-tested.
- [ ] A2 scheduled batch pipeline implemented with idempotency and resume behavior.
- [ ] A3 registry and promote/rollback controls active.
- [ ] A5 cost + latency + canary controls active.
- [ ] A6 auth/tenant/audit controls active.
- [ ] All hard gates encoded in CI or pre-promotion checks.
- [ ] `docs/ARCHITECTURE_AND_ROADMAP.md` synced with final implemented state.

---

## 9) Governance recommendation

Use a "candidate -> staging -> production" promotion workflow for every model release.

Promotion must require:
1. Passing eval gate (quality)
2. Passing deployment smoke gate (runtime)
3. Passing cost/SLO gate (economics and reliability)
4. Signed lineage in registry (traceability)

This ensures every release is reproducible, operable, and rollback-safe.

