# ValonyLabs Studio Companion: Blueprint Execution Backlog and Path Map

**Primary reference:** `docs/BLUEPRINT_ADOPTION_GUIDE.md`  
**This companion file:** executable backlog with explicit repository path targets.

---

## 1) Purpose

This companion document converts the adoption blueprint into a clear execution map:

- what to build,
- where it belongs in the repository,
- what artifacts must be produced,
- and what acceptance gates determine completion.

It is intentionally scoped to **Lane A production systems only**.

---

## 2) Execution order (approved production sequence)

Run in this order:

1. **A1** - Deployment closure (GGUF critical path)
2. **A2** - Reliable continuous training
3. **A3** - Model registry and promotion control
4. **A5** - Inference hardening and cost controls
5. **A6** - Enterprise readiness (auth, tenancy, audit)

---

## 3) Path location map (where work goes)

## A1 - Deployment closure (GGUF)

### New/updated paths
- `app/trainers/export.py` (new)
- `scripts/export_gguf.py` (new, optional wrapper)
- `tests/test_export_gguf.py` (new)
- `docs/ARCHITECTURE_AND_ROADMAP.md` (update status when complete)

### Artifact paths
- `outputs/<domain>/artifacts/*.gguf`
- `outputs/<domain>/artifacts/*.metadata.json`

### Acceptance gate
- Exported GGUF can be loaded and queried in smoke test.

---

## A2 - Reliable continuous training

### New/updated paths
- `scripts/batch_pipeline.py` (new)
- `app/pipeline/runner.py` (new, optional modular runner)
- `tests/test_batch_pipeline.py` (new)
- `configs/domains/*.yaml` (optional gate thresholds updates)

### Artifact paths
- `outputs/runs/<run_id>/manifest.json`
- `outputs/runs/<run_id>/stage_status.json`

### Acceptance gate
- unattended full run: `collect -> forge -> train -> eval -> deploy`
- idempotency + resume behavior validated

---

## A3 - Model registry and promotion control

### New/updated paths
- `app/registry/model_registry.py` (new)
- `app/registry/schemas.py` (new)
- `app/main.py` (new endpoints for registry/promote/rollback)
- `tests/test_model_registry.py` (new)

### Artifact paths
- `outputs/registry/model_versions.json` or DB-backed table dump
- `outputs/registry/promotion_events.jsonl`

### Acceptance gate
- every deployed model maps to dataset manifest + eval report + artifact hash
- rollback by version id works

---

## A5 - Inference hardening and cost controls

### New/updated paths
- `app/cache/semantic.py` (new)
- `app/observability/cost.py` (new)
- `app/observability/slo.py` (new)
- `app/main.py` (cost and canary-related surfaces if exposed by API)
- `tests/test_semantic_cache.py` (new)

### Artifact paths
- `outputs/metrics/slo_<timestamp>.json`
- `outputs/metrics/cost_<timestamp>.json`
- `outputs/metrics/canary_<timestamp>.json`

### Acceptance gate
- equal or better quality with lower (or stable) latency/cost in canary window

---

## A6 - Enterprise readiness

### New/updated paths
- `app/auth/jwt.py` (new)
- `app/auth/middleware.py` (new)
- `app/memory/store.py` (new)
- `app/audit/logging.py` (new)
- `tests/test_auth_jwt.py` (new)
- `tests/test_tenant_isolation.py` (new)

### Artifact paths
- `outputs/audit/events_<date>.jsonl`
- `outputs/compliance/access_trace_<date>.json`

### Acceptance gate
- tenant isolation enforced in automated tests
- auditable request trace includes model version and user/tenant context

---

## 4) Backlog board (ready for ticketing)

| ID | Stage | Task | Repo path(s) | Deliverable |
|---|---|---|---|---|
| A1-01 | A1 | Implement LoRA merge + GGUF export core | `app/trainers/export.py` | Export function + tests |
| A1-02 | A1 | Add export CLI wrapper | `scripts/export_gguf.py` | Operator-friendly command |
| A1-03 | A1 | Add export smoke test | `tests/test_export_gguf.py` | Runtime validation |
| A2-01 | A2 | Build batch orchestrator | `scripts/batch_pipeline.py` | Scheduled end-to-end run |
| A2-02 | A2 | Add idempotency/resume state | `app/pipeline/runner.py` | Recoverable pipeline |
| A2-03 | A2 | Add pipeline tests | `tests/test_batch_pipeline.py` | Reliability checks |
| A3-01 | A3 | Create model registry schemas | `app/registry/schemas.py` | Versioned model records |
| A3-02 | A3 | Add registry service | `app/registry/model_registry.py` | CRUD + lineage |
| A3-03 | A3 | Add promote/rollback endpoints | `app/main.py` | Controlled release flow |
| A5-01 | A5 | Add semantic cache | `app/cache/semantic.py` | Reuse prior answers safely |
| A5-02 | A5 | Add cost accounting | `app/observability/cost.py` | Intelligence-per-dollar metrics |
| A5-03 | A5 | Add SLO evaluator | `app/observability/slo.py` | Latency/error gates |
| A6-01 | A6 | Add JWT validation and context extraction | `app/auth/jwt.py`, `app/auth/middleware.py` | Authenticated tenancy |
| A6-02 | A6 | Add tenant memory storage interface | `app/memory/store.py` | Tenant-scoped persistence |
| A6-03 | A6 | Add audit logging | `app/audit/logging.py` | Compliance event trail |

---

## 5) Definition of done (program level)

The production sequence is complete only when:

1. A1 artifacts are deployable and rollback-safe.
2. A2 automation can run unattended and recover from failure.
3. A3 promotion is controlled by explicit quality and lineage gates.
4. A5 canary + cost/SLO checks protect runtime reliability and economics.
5. A6 enforces tenant isolation and produces auditable traces.

---

## 6) Operational notes

- Keep `docs/BLUEPRINT_ADOPTION_GUIDE.md` as the strategy and lifecycle source.
- Keep this file as the execution and path map source.
- Update both docs when path locations, stage boundaries, or release gates change.

