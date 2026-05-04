# ValonyLabs Studio v3.0

**Domain-agnostic post-training, evaluation, and inference platform.**

One-click workflow: *Collect data -> Forge dataset -> Train (SFT/DPO/ORPO/KTO/GRPO) -> Evaluate -> Deploy adapter -> Chat with RAG.*

| Tier | Hardware | Training | Inference | Status |
|------|----------|----------|-----------|--------|
| Consumer GPU | RTX 4090/5090 (24-32 GB) | Unsloth + TRL | vLLM / SGLang | Production |
| Notebook | Colab T4/L4/A100 | Unsloth + TRL | HF Transformers | Tested |
| Cloud GPU | AWS g5.xlarge (L4 24 GB) | Unsloth + TRL | vLLM | Target |
| Apple Silicon | M1-M4 (16-96 GB unified) | MLX-LM + LoRA | MLX-LM | Supported |
| Ollama Cloud | Nemotron, Llama 3.3, Qwen | -- (catalog) | Ollama streaming | Production |
| CPU | Any x86/arm64 | TRL + PEFT (small models) | HF Transformers | Dev/test |

---

## Architecture

```
frontend/            React 18 + TypeScript + Vite + Tailwind
  src/components/
    DataForge.tsx    Upload, YouTube harvest, dataset build
    Train.tsx        Training knobs, live loss chart, job polling
    TrainingChart.tsx SVG loss curve (raw + EMA smoothed)
    ChatWidget.tsx   Floating chat with SSE streaming + Docs RAG
    Domains.tsx      Domain config CRUD
    Health.tsx       Hardware, backend, provider status

app/                 FastAPI backend
  main.py            API routes, job registry, SSE endpoints
  models.py          Pydantic schemas (request/response models)
  hardware/          Auto-detect GPU/CPU/MPS, resolve training profile
  templates/         Chat template registry (Qwen, Llama, Alpaca, ...)
  data_forge/        Ingest (PDF/DOCX/images), chunk, filter, Q/A synth
  harvesters/        YouTube transcript harvester (yt-dlp + captions)
  trainers/
    base.py          BaseAgnosticTrainer (dataset, model, LoRA, save)
    sft_trainer.py   SFT (TRL + MLX paths)
    dpo_trainer.py   DPO (preference alignment)
    orpo_trainer.py  ORPO (single-stage SFT + DPO)
    kto_trainer.py   KTO (binary feedback)
    grpo_trainer.py  GRPO (verifiable reward, e.g. math)
    callbacks.py     LossHistoryCallback for live metrics
    backends.py      Model loading (Unsloth / MLX / TRL fallback)
    hub.py           Push adapters to HuggingFace Hub
  inference/
    manager.py       Backend router + LoRA hot-swap registry
    hf_backend.py    HF Transformers (CPU/GPU, streaming)
    ollama_backend.py Ollama Cloud / local daemon
    vllm_backend.py  vLLM (CUDA, FP8, LoRA)
    mlx_backend.py   MLX-LM (Apple Silicon)
  eval/
    scorer.py        Test-set loss, perplexity, QA accuracy
    judge.py         LLM-as-judge (base vs adapted, win rate)
    runner.py        Orchestrate eval + quality gate
  rag/               BM25 retriever + Docs corpus for in-app help
  providers/         Ollama Cloud / OpenAI / OpenRouter for Q/A synth

configs/domains/     Per-engagement YAML configs
notebooks/           Colab-ready tutorials (01-04)
tests/               177+ unit tests, CI via GitHub Actions
```

---

## Quickstart

### Local (Mac / Linux / WSL)

```bash
git clone https://github.com/valonys/finetuningtraining.git
cd finetuningtraining

# Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt          # or requirements-cuda.txt
cp .env.example .env                         # edit: add OLLAMA_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173.

### Docker

```bash
# GPU (needs nvidia-container-toolkit)
docker compose up

# CPU-only
docker compose --profile cpu up
```

### Google Colab

Open `notebooks/02_sft_qwen_alpaca.ipynb` in Colab. The bootstrap cell clones the repo, installs deps (including Unsloth on T4+), and sets up HF_TOKEN from Colab Secrets.

---

## Training methods

| Method | Dataset schema | When to use |
|--------|----------------|-------------|
| **SFT** | `{instruction, response}` or `{text}` | First-pass domain adaptation |
| **DPO** | `{prompt, chosen, rejected}` | Preference alignment from paired feedback |
| **ORPO** | `{prompt, chosen, rejected}` | Single-stage SFT + preference (no ref model) |
| **KTO** | `{prompt, completion, label}` | Binary thumbs-up/down feedback |
| **GRPO** | `{prompt, ground_truth}` | Verifiable reward (math, code tests) |

### Training via the Studio UI

1. **Domains** tab: create a domain config (system prompt + constitution).
2. **Data Forge** tab: upload files / harvest YouTube / build dataset.
3. **Train** tab: pick domain, model, method, dataset path. Queue job.
4. Watch the live loss chart. The backend streams per-step metrics via `LossHistoryCallback`.

### Training via notebooks (Colab)

```python
from app.trainers import AgnosticSFTTrainer

trainer = AgnosticSFTTrainer(
    config=config,
    base_model_id='Qwen/Qwen2.5-0.5B-Instruct',
    dataset_path='./data/processed/my_domain_sft.jsonl',
)
result = trainer.train()
# result['loss_history'] -> list of {step, loss, learning_rate, ...}
# result['adapter_path'] -> 'outputs/my_domain/'
```

### Push adapter to HuggingFace

```python
from app.trainers.hub import push_adapter_to_hub

url = push_adapter_to_hub(
    adapter_dir=result['adapter_path'],
    repo_id='amiguel/qwen-0.5b-my_domain-sft',
    private=True,
    metadata=result,
)
```

### Export adapter to GGUF (Sprint 03 / A1)

For deployment via llama.cpp or Ollama, merge the adapter into its base
model and export a quantized GGUF:

```bash
# One-time: vendor the llama.cpp toolchain (clones to ~/.local/llama.cpp)
bash scripts/install_llamacpp.sh

# Export
python scripts/export_gguf.py \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter outputs/ai_llm \
    --output-dir outputs/ai_llm/artifacts \
    --quant Q4_K_M
```

Produces `<adapter>-q4_k_m.gguf`, a `<adapter>-q4_k_m.metadata.json`
sidecar (base id, adapter sha256, quant, file sha256, timestamp), and
maintains a `latest.gguf` rollback pointer in the output directory.
See `docs/SPRINTS.md` for sprint status.

---

## Evaluation pipeline

The eval module (`app/eval/`) provides automated quality gating:

```python
from app.eval import run_eval

report = run_eval(
    domain_name='ai_llm',
    adapter_path='outputs/ai_llm/',
    base_model_id='Qwen/Qwen2.5-0.5B-Instruct',
    test_path='data/processed/ai_llm_sft_test.jsonl',
    generate_base=base_fn,
    generate_adapted=adapted_fn,
    win_rate_threshold=0.55,
)

if report['quality_gate']['passed']:
    push_adapter_to_hub(...)
```

### Three evaluation tiers

| Tier | What it measures | Cost | Frequency |
|------|-----------------|------|-----------|
| **Test-set loss** | Perplexity on held-out split | Free (local compute) | Every run |
| **QA accuracy** | Substring match on a gold Q/A bank | Free | Every run |
| **LLM-as-judge** | Win rate: adapted vs base (Nemotron judge) | ~$0.01/prompt | Weekly |

### Quality gate

The runner produces `outputs/<domain>/eval_<timestamp>.json` with a pass/fail verdict. The continuous pipeline only deploys adapters that pass.

---

## Continuous training workflow

```
Mon/Thu 02:00 UTC (EventBridge cron)
    |
    v
1. COLLECT         YouTube harvester, RSS, doc uploads
    |
    v
2. FORGE           Chunk + filter + synth Q/A (target=5000)
    |
    v
3. TRAIN           SFT (3 epochs), optionally DPO
    |
    v
4. EVAL            test_loss + QA accuracy + LLM-as-judge
    |
    v
5. DEPLOY          If quality gate passes:
                     - Push adapter to HF Hub
                     - Hot-swap on vLLM inference
                     - Log to model registry
```

### Scaling the dataset

| Source | Rows per unit | How |
|--------|---------------|-----|
| YouTube transcript | 50-200 Q/A pairs per video | `POST /v1/forge/harvest/youtube` |
| PDF / DOCX | 20-100 Q/A pairs per document | Upload in Data Forge |
| HF dataset (Alpaca, etc.) | As many as you want | `hf_dataset_config` in trainer |
| Target size multiplier | Synth generates more Q/A per chunk | `target_size` in build_dataset |

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/healthz` | Hardware, backend, provider status |
| GET | `/v1/templates` | Available chat templates |
| GET | `/v1/domains/configs` | List domain configs |
| POST | `/v1/domains/configs` | Create domain config |
| POST | `/v1/forge/upload` | Upload files (multipart) |
| POST | `/v1/forge/build_dataset` | Chunk + synth Q/A -> JSONL |
| POST | `/v1/forge/harvest/youtube` | YouTube keyword -> transcripts |
| POST | `/v1/jobs/create` | Queue a training job |
| GET | `/v1/jobs/{id}` | Job status + live loss_history |
| GET | `/v1/jobs` | List all jobs |
| GET | `/v1/domains` | List trained adapters |
| POST | `/v1/inference/reload` | Re-scan adapters |
| POST | `/v1/chat` | Chat (batched response) |
| POST | `/v1/chat/stream` | Chat (SSE streaming) |

---

## Docker deployment

### Build

```bash
# CUDA (production on g5.xlarge / L4 / A100)
docker build -t valonylabs-studio .

# CPU-only (CI, dev, Fargate control plane)
docker build -t valonylabs-studio:cpu --target cpu .
```

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLLAMA_API_KEY` | For synth | -- | Ollama Cloud key for Q/A generation |
| `HF_TOKEN` | For gated models | -- | HuggingFace token |
| `VALONY_BASE_MODEL` | No | `Qwen/Qwen2.5-7B-Instruct` | Base model HF id |
| `VALONY_INFERENCE_BACKEND` | No | auto-detected | `vllm`, `hf`, `ollama`, `mlx` |
| `VALONY_PREWARM_INFERENCE` | No | `0` | Set `1` to load model at startup |

### AWS target architecture

| Component | Service | Estimated cost |
|-----------|---------|---------------|
| Control plane (API + UI) | ECS Fargate 0.5 vCPU / 1 GB | ~$15/mo |
| Training worker | ECS on g5.xlarge (L4 24 GB), spot | ~$4/week |
| Inference | ECS on g5.xlarge + vLLM | ~$1.21/hr |
| Storage | S3 (datasets, adapters, eval logs) | < $5/mo |
| Secrets | AWS Secrets Manager | < $1/mo |
| Scheduler | EventBridge (twice-weekly cron) | Free tier |

---

## Project structure

```
files_brevNVIDIA_v3.0/
|-- app/
|   |-- __init__.py
|   |-- main.py                 # FastAPI app, all routes
|   |-- models.py               # Pydantic schemas
|   |-- config_loader.py        # YAML domain config I/O
|   |-- hardware/               # GPU/CPU/MPS detection + profiles
|   |-- templates/              # Chat template registry
|   |-- data_forge/             # Ingest, chunk, filter, synth
|   |-- harvesters/             # YouTube transcript harvester
|   |-- trainers/               # SFT/DPO/ORPO/KTO/GRPO + hub push
|   |-- inference/              # vLLM, HF, Ollama, MLX, llama.cpp
|   |-- eval/                   # Scorer, LLM judge, eval runner
|   |-- rag/                    # BM25 retriever + docs corpus
|   |-- providers/              # Ollama Cloud, OpenAI, OpenRouter
|-- frontend/                   # React + TypeScript + Vite
|-- configs/domains/            # Per-engagement YAML configs
|-- notebooks/                  # Colab tutorials (01-04)
|-- tests/                      # 177+ unit tests
|-- outputs/                    # Trained adapters + eval reports
|-- data/                       # uploads/ + processed/
|-- Dockerfile                  # Multi-stage: CPU + CUDA targets
|-- docker-compose.yml          # Local dev with GPU or CPU profile
|-- requirements-cpu.txt        # CPU-only deps
|-- requirements-cuda.txt       # CUDA deps (RTX/Colab/AWS)
|-- requirements-mlx.txt        # Apple Silicon deps
|-- requirements-test.txt       # CI test deps
|-- pyproject.toml              # Project metadata + optional extras
|-- .env.example                # Environment variable reference
|-- LICENSE                     # Apache 2.0
```

---

## Testing

```bash
# Run all tests (no GPU or external API needed)
python -m pytest

# Run a specific test file
python -m pytest tests/test_eval.py -v

# CI runs automatically via GitHub Actions on push to develop/main
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Built by [ValonyLabs](https://github.com/valonys) with [Claude Code](https://claude.ai/claude-code).
