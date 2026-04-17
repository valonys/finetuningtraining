# ValonyLabs Studio v3.0 -- Architecture, Alignment, and Roadmap

**Last updated:** 2026-04-17
**Authors:** Ataliba Miguel, Claude Code
**Purpose:** Persistent reference document so any future contributor (human or AI) can reconstruct full context without the original conversation.

---

## 1. What this document covers

This is the single source of truth for:

- What ValonyLabs Studio v3.0 is and what it does today
- The hardware targets and deployment philosophy
- Where the Studio converges with the broader Hyper-Personalized Teams AI Assistant blueprint
- What's built, what's partially built, and what's missing
- The phased roadmap from dev system to production
- Spec-driven development approach for v3.x iterations

If this conversation is lost, this document + the README.md + the codebase itself are sufficient to continue.

---

## 2. What ValonyLabs Studio v3.0 is

A **domain-agnostic post-training, evaluation, and inference platform** with a React/TypeScript frontend and FastAPI backend. It handles the full lifecycle:

```
Collect raw data
    -> Forge training dataset (chunk, filter, synth Q/A)
    -> Train (SFT / DPO / ORPO / KTO / GRPO)
    -> Evaluate (test loss, QA accuracy, LLM-as-judge)
    -> Deploy adapter (HF Hub push, LoRA hot-swap)
    -> Serve via chat (SSE streaming, RAG, domain routing)
```

Everything auto-adapts to the available hardware. One YAML config per engagement. No code changes needed to switch between Mac, RTX, Colab, or cloud.

---

## 3. Hardware targets

### Current dev system (temporary)

- Intel i7-7700HQ, 16 GB RAM, macOS 13.7.8 x86_64
- torch 2.2.2 (ceiling for this CPU -- PyTorch dropped x86 Mac after 2.2)
- Inference: HF backend at ~0.5-0.8 tok/s on 0.5B models
- Training: CPU-only TRL, impractical for anything beyond smoke tests

### Incoming dev system (primary)

- **Apple Mac M4 Pro** -- 12-core CPU, 20-core GPU, 24-36 GB unified memory
- Training: MLX-LM LoRA (native, fast) or TRL on MPS
- Inference: MLX-LM (~15-30 tok/s on 0.5B, ~5-10 tok/s on 7B in 4-bit)
- This becomes the daily driver for development, dataset curation, and quick training iterations

### Production GPU (self-hosted)

- **NVIDIA RTX 5080 or 5090** (16-32 GB VRAM, Ada Lovelace+)
- Training: Unsloth + TRL (4-bit QLoRA, Flash Attention 2)
- Inference: vLLM with continuous batching, FP8 KV cache, LoRA hot-swap
- Target: 30-50ms/token decode, <1s TTFT on 7-14B models
- Scales by adding a second GPU with tensor parallelism

### Notebook / cloud

- **Google Colab**: T4 (free), L4, A100 (Pro) -- proven via notebooks 01-04
- **NVIDIA Brev**: A10G to H100 -- same Docker image, just `docker compose up`
- **AWS g5.xlarge**: L4 24 GB, ~$1.21/hr -- target for the managed deployment

### Ollama Cloud

- Nemotron, Llama 3.3, Qwen 2.5 72B as synth providers and catalog inference
- Used for Q/A synthesis in the Data Forge and as the LLM-as-judge evaluator
- No LoRA hot-swap -- domain names route to different Ollama model tags

---

## 4. What's built and working (as of 2026-04-17)

### Backend (app/)

| Module | What it does | Test coverage |
|--------|-------------|---------------|
| `main.py` | FastAPI app, all routes, job registry, SSE streaming | Integration |
| `models.py` | Pydantic schemas for every request/response | Type-checked |
| `hardware/` | Auto-detect GPU/CPU/MPS, resolve profiles (6 tiers) | 3 tests |
| `templates/` | Chat template registry (11 templates, auto-resolved from model id) | 17 tests |
| `data_forge/` | Ingest PDF/DOCX/XLSX/PPTX/HTML/TXT/images, chunk, filter noise, synth Q/A | 37 tests |
| `harvesters/youtube.py` | yt-dlp keyword search + youtube-transcript-api v1.x captions | 9 tests |
| `trainers/base.py` | BaseAgnosticTrainer: dataset loading, model loading, LoRA, save, loss history | 7 tests |
| `trainers/sft_trainer.py` | SFT via TRL or MLX, defensive TRL version compat | 3 tests |
| `trainers/dpo_trainer.py` | DPO via TRL | Code exists, not Colab-tested |
| `trainers/orpo_trainer.py` | ORPO via TRL | Code exists, not Colab-tested |
| `trainers/kto_trainer.py` | KTO via TRL | Code exists, not Colab-tested |
| `trainers/grpo_trainer.py` | GRPO via TRL + GSM8K reward signal | Code exists, not Colab-tested |
| `trainers/callbacks.py` | LossHistoryCallback for live per-step metrics | 6 tests |
| `trainers/backends.py` | Model loading (Unsloth / MLX / TRL), LoRA, FA2 fallback, kbit prep | 5 tests |
| `trainers/hub.py` | Push adapters to HuggingFace Hub with auto-generated model card | 5 tests |
| `inference/manager.py` | Backend router, LoRA registry, adapter scan from outputs/ | Integration |
| `inference/hf_backend.py` | HF Transformers: CPU/GPU, streaming, isin_mps patch | Working (just tested) |
| `inference/ollama_backend.py` | Ollama Cloud/local: streaming, Nemotron | Production (daily use) |
| `inference/vllm_backend.py` | vLLM: CUDA, FP8, LoRA hot-swap | Code exists, needs CUDA box |
| `inference/mlx_backend.py` | MLX-LM: Apple Silicon native | Code exists, needs M4 Pro |
| `eval/scorer.py` | Test-set loss/perplexity, QA accuracy (exact/substring match) | 5 tests |
| `eval/judge.py` | LLM-as-judge head-to-head (base vs adapted, position randomised) | 4 tests |
| `eval/runner.py` | Orchestrate eval, produce JSON report, quality gate verdict | Integration |
| `rag/` | BM25 retriever + 17-article docs corpus for in-app help | 19 tests |
| `providers/` | Ollama Cloud, OpenAI, OpenRouter for Q/A synthesis | 21 tests |

**Total: 186 unit tests, all passing. CI via GitHub Actions.**

### Frontend (frontend/)

| Component | What it does |
|-----------|-------------|
| `DataForge.tsx` | Drag-drop upload, YouTube harvester, template dropdown, build dataset |
| `Train.tsx` | Training knobs (epochs, LR, batch), auto-polling job status |
| `TrainingChart.tsx` | SVG loss curve (raw + EMA smoothed), summary chips |
| `ChatWidget.tsx` | Floating chat with SSE streaming, Docs RAG, domain dropdown, citations |
| `Domains.tsx` | Domain config CRUD |
| `Health.tsx` | Hardware, backend, provider status |

### Infrastructure

| File | What it does |
|------|-------------|
| `Dockerfile` | Multi-stage: Node (frontend build) -> Python. Two targets: `default` (CUDA 12.4) and `cpu` |
| `docker-compose.yml` | GPU service (default) + CPU profile. Persistent volumes for data/outputs/configs/HF cache |
| `.dockerignore` | Exclude .git, node_modules, outputs, data, notebooks |
| `requirements-cpu.txt` | CPU-only stack |
| `requirements-cuda.txt` | CUDA stack (Unsloth, vLLM, flash-attn, bitsandbytes) |
| `requirements-mlx.txt` | Apple Silicon stack (mlx, mlx-lm) |
| `requirements-test.txt` | CI test deps (no torch) |
| `.env.example` | All environment variables documented |

### Notebooks

| Notebook | What it does | Colab status |
|----------|-------------|-------------|
| `01_data_forge_demo.ipynb` | Ingest + build dataset | Bootstrap ready |
| `02_sft_qwen_alpaca.ipynb` | SFT training + loss chart + HF push | Fully tested (T4) |
| `03_dpo_llama3_identity.ipynb` | DPO preference alignment | Bootstrap ready, not tested |
| `04_grpo_math_reasoning.ipynb` | GRPO with GSM8K reward | Bootstrap ready, not tested |

---

## 5. What's NOT built yet (gaps vs the full blueprint)

### High priority (needed for production)

| Gap | Description | Effort | Blocked by |
|-----|-------------|--------|------------|
| **GGUF export pipeline** | After SFT/merge, convert to GGUF for Ollama deployment on RTX server. `app/trainers/export.py` with `export_gguf(adapter_path, base_model, quant="Q4_K_M") -> path.gguf` | 1 day | Nothing |
| **Batch training cron** | `scripts/batch_pipeline.py` that runs collect -> forge -> train -> eval -> deploy. Triggered by crontab or EventBridge. | 1 day | Eval module (done) |
| **Microsoft Teams Bot adapter** | Self-hosted Bot Framework SDK (Python). `app/integrations/teams.py` receiving Activities, translating to `/v1/chat/stream`, forwarding deltas as adaptive card updates. | 3-5 days | Auth module |
| **JWT auth + multi-user** | Teams tokens validate user identity. `app/auth/` middleware extracts user_id from JWT claims. PostgreSQL RLS ensures data isolation. | 2-3 days | PostgreSQL |

### Medium priority (needed for enterprise features)

| Gap | Description | Effort | Blocked by |
|-----|-------------|--------|------------|
| **PostgreSQL + pgvector memory graph** | Per-user facts, preferences, relationships with vector search. `app/memory/` module. Replaces in-memory state with persistent, queryable storage. | 2-3 days | PostgreSQL infra |
| **Redis semantic cache** | Cache (prompt_embedding -> response) with TTL. Short-circuit inference when cosine similarity > 0.95. `app/cache/semantic.py` | 1 day | Redis infra |
| **Qdrant vector retriever** | Replace BM25 with Qdrant for large doc collections (1000+ articles). `app/rag/qdrant_retriever.py` with `sentence-transformers` embeddings. | 2 days | Qdrant infra |
| **DPO data generation pipeline** | Send same prompt through base + SFT adapter, LLM-as-judge picks chosen/rejected, save as JSONL. `app/data_forge/preference_synthesis.py` | 1-2 days | Working SFT adapter |

### Future phases

| Gap | Description | Effort |
|-----|-------------|--------|
| **Agentic tools** (Jira, Graph API, SQL) | Tool server with Pydantic-validated schemas. ReAct loop or LangGraph. `app/agents/` | 3-5 days |
| **Domain-specific GRPO reward signals** | Beyond GSM8K: factual QA verifier, code test runner, constrained JSON validator | 2-3 days per signal |
| **Tensor parallelism config** | docker-compose device count for multi-GPU RTX setups | Trivial |
| **Model registry** | Track adapter versions, parent datasets, eval scores, rollback | 2-3 days |
| **AWS Terraform/CDK** | ECS cluster, S3, Secrets Manager, EventBridge, ALB | 2-3 days |

---

## 6. Convergence: Studio v3.0 meets the Teams blueprint

The two systems solve the same problem from two directions:

- **Studio v3.0** (bottom-up): training platform with dataset curation, multi-method training, eval gating, and adapter management
- **Teams blueprint** (top-down): user-facing AI assistant with memory, RAG, agents, and enterprise integration

They converge at the **FastAPI + Docker layer**. The Studio's existing endpoints (`/v1/chat/stream`, `/v1/forge/build_dataset`, `/v1/jobs/create`, etc.) are the same APIs the Teams Bot adapter will call.

```
Teams Client
    |
    v
Bot Framework Adapter (new: app/integrations/teams.py)
    |
    v
FastAPI Gateway (existing: app/main.py)
    |
    +---> /v1/chat/stream      (existing)
    +---> /v1/forge/...         (existing)
    +---> /v1/jobs/create       (existing)
    +---> /v1/eval/run          (existing)
    +---> /v1/memory/...        (new: app/memory/)
    +---> /v1/tools/...         (new: app/agents/)
    |
    v
Inference backends (existing: app/inference/)
    +---> vLLM on RTX 5090     (production)
    +---> MLX on M4 Pro        (development)
    +---> Ollama Cloud         (synth + catalog)
    +---> HF Transformers      (fallback)
```

The two bridges connecting them:
1. **GGUF export** -- so adapters trained in Studio can deploy to Ollama on the RTX production server
2. **Teams Bot adapter** -- so the Studio's chat/streaming infrastructure serves Teams users

---

## 7. Vendor lock-in analysis

| Layer | Lock-in risk | Studio's position |
|-------|-------------|-------------------|
| Training compute | Azure ML = $3,500/mo | Self-hosted: M4 Pro (dev) + RTX 5090 (prod) + Colab (notebooks). $0 recurring. |
| Inference compute | Azure A100 = $3.67/hr | vLLM on RTX 5090 ($2,000 one-time) or g5.xlarge ($1.21/hr spot). |
| Vector DB | Qdrant Cloud / Pinecone = $200+/mo | BM25 today (free, local). Qdrant self-hosted when needed. |
| Memory DB | Neo4j / Cosmos DB = $300+/mo | PostgreSQL + pgvector (free, self-hosted). |
| Model format | PyTorch-only | PEFT adapters + GGUF export = runs on any backend. |
| Synth provider | OpenAI API = $0.15/1K tokens | Ollama Cloud Nemotron = fraction of cost. Swappable via `app/providers/`. |
| Frontend | None (React, self-hosted) | No dependency on any cloud UI service. |

**12-month cost comparison:**
- Azure lock-in path: ~$42,000-48,000
- Studio self-hosted path: ~$2,600 (hardware) + $600 (electricity) = **$3,200**

---

## 8. Spec-driven development approach for v3.x

Going forward, every new feature follows this flow:

### Spec -> Implement -> Test -> Ship

```
1. SPEC        Write a one-page spec (.md) under docs/specs/
               covering: goal, non-goals, API surface, data flow,
               acceptance criteria, test plan.

2. IMPLEMENT   Build the module following existing patterns:
               - Backend: app/<module>/
               - Frontend: frontend/src/components/<Component>.tsx
               - Types: app/models.py (Pydantic) + frontend/src/types.ts

3. TEST        Unit tests in tests/ (mocked deps, no network).
               Integration test via notebook or curl.
               All existing tests must still pass.

4. SHIP        Commit with descriptive message.
               Push to develop.
               CI passes (GitHub Actions).
               If it touches inference or training: validate on at
               least one real hardware target (Mac/Colab/RTX).
```

### Spec template

```markdown
# Spec: <feature name>

## Goal
One sentence.

## Non-goals
What this does NOT do.

## API surface
New or changed endpoints / functions / components.

## Data flow
How data moves through the system.

## Acceptance criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Test plan
- Unit: what to mock, what to test
- Integration: which notebook or curl sequence validates it

## Estimated effort
X days.
```

### Version numbering

- **v3.0.x** -- bug fixes, compat patches, notebook polish (where we are now)
- **v3.1.0** -- GGUF export + batch pipeline cron (production deployment)
- **v3.2.0** -- PostgreSQL memory graph + Redis cache (personalization)
- **v3.3.0** -- Teams Bot adapter + JWT auth (enterprise integration)
- **v3.4.0** -- Qdrant vector retriever + agentic tools (scale)

---

## 9. Phased roadmap

### Phase 1: Platform core (DONE)

Everything in section 4 above. Studio runs end-to-end on Colab and locally. Adapters train, evaluate, push to HF, and serve via chat.

### Phase 2: M4 Pro validation (when hardware arrives)

- Validate MLX training path end-to-end
- Validate MLX inference (speed, quality)
- Test the full pipeline locally: forge -> train -> eval -> chat
- Benchmark: TTFT, tok/s, training time per epoch on 5k rows
- Estimated time: 1-2 days

### Phase 3: RTX production server (when hardware arrives)

- `docker compose up` on Ubuntu + NVIDIA drivers
- Validate vLLM inference (TTFT <1s, 30-50ms/token decode)
- Add GGUF export for Ollama deployment path
- Wire batch pipeline cron (crontab or systemd timer)
- Load test with 10-50 concurrent users
- Estimated time: 3-5 days

### Phase 4: Teams integration (after Phase 3)

- Bot Framework adapter (`app/integrations/teams.py`)
- JWT auth middleware (`app/auth/`)
- PostgreSQL + pgvector memory graph (`app/memory/`)
- Redis semantic cache (`app/cache/`)
- Estimated time: 2 weeks

### Phase 5: Scale and iterate (ongoing)

- Qdrant for large doc collections
- DPO preference generation pipeline
- Domain-specific GRPO reward signals
- Agentic tools (Jira, Graph API, SQL)
- Second RTX 5090 with tensor parallelism
- AWS managed deployment (Terraform/CDK)

---

## 10. Key decisions and rationale

| Decision | Rationale |
|----------|-----------|
| React + TypeScript frontend (not Gradio) | ValonyLabs company standard. Professional UI, Tw Cen MT font, floating chat widget. |
| FastAPI backend (not Flask/Django) | Async, SSE streaming, Pydantic validation, auto-docs. |
| Domain configs as YAML (not DB) | Portable, git-trackable, no infra dependency for a config change. |
| BM25 for Docs RAG (not embeddings) | 17 articles. BM25 is simpler, faster, and sufficient. Swap to Qdrant when corpus grows. |
| Ollama Cloud Nemotron for synth | Cheapest strong model for Q/A generation. No OpenAI dependency. |
| PEFT adapters (not full merge) | Smaller files, faster hot-swap, composable (stack adapters). GGUF export handles the merge-when-needed case. |
| Eval quality gate before deploy | Prevents regression. Twice-weekly batch won't push a worse adapter. |
| No vendor lock-in | Angola market: unreliable internet, cost sensitivity, need for offline capability. Self-hosted is not optional. |

---

## 11. File reference for future contributors

If you're picking this up without the conversation:

1. **Start here:** `README.md` -- quickstart, architecture, API reference
2. **Understand the hardware:** `app/hardware/` -- how profiles are resolved
3. **Understand training:** `app/trainers/base.py` -- the shared plumbing all methods use
4. **Understand inference:** `app/inference/manager.py` -- how backends are selected and LoRA adapters hot-swapped
5. **Understand data:** `app/data_forge/` -- how raw docs become training JSONL
6. **Understand eval:** `app/eval/runner.py` -- how quality gates work
7. **Environment:** `.env.example` -- every configurable env var documented
8. **Docker:** `Dockerfile` + `docker-compose.yml` -- build and run
9. **Tests:** `python -m pytest` -- 186 tests, no GPU or network needed
10. **This document:** the "why" behind every "what"

---

*This document is versioned in git alongside the code. Update it when architectural decisions change.*
