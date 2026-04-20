# First-time setup guide

Step-by-step from `git clone` to a running system. Each section is self-contained -- pick the one that matches your hardware.

---

## Mac M4 Pro (development)

```bash
# 1. Clone
git clone https://github.com/valonys/finetuningtraining.git
cd finetuningtraining
git checkout develop

# 2. Python environment (MLX stack)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mlx.txt

# 3. Environment variables
cp .env.example .env
# Edit .env:
#   OLLAMA_API_KEY=<your key from https://ollama.com>
#   HF_TOKEN=<optional, from https://huggingface.co/settings/tokens>

# 4. Start backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. Start frontend (new terminal)
cd frontend
npm install    # first time only
npm run dev

# 6. Open
open http://localhost:5173
```

**Validate:**
```bash
curl http://localhost:8000/healthz | python3 -m json.tool
# Expect: "tier": "apple_silicon", "training_backend": "mlx"
```

---

## RTX 5080/5090 production server

### Prerequisites

- Ubuntu 22.04 LTS
- NVIDIA drivers 545+ (`sudo apt install nvidia-driver-545`)
- Docker Engine (`curl -fsSL https://get.docker.com | sh`)
- nvidia-container-toolkit:
  ```bash
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

### Deploy

```bash
git clone https://github.com/valonys/finetuningtraining.git
cd finetuningtraining

# Environment
cp .env.example .env
# Edit .env:
#   VALONY_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
#   VALONY_INFERENCE_BACKEND=vllm
#   VALONY_PREWARM_INFERENCE=1
#   OLLAMA_API_KEY=<your key>
#   HF_TOKEN=<your token>

# Build and run (GPU)
docker compose up -d

# Validate
docker compose logs -f studio    # watch startup
curl http://localhost:8000/healthz | python3 -m json.tool
# Expect: "tier": "cuda_consumer", "inference_backend": "vllm"
```

### Validate inference

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello","domain_config_name":"base","temperature":0.5,"max_new_tokens":32}'
```

### Validate training

```bash
curl -X POST http://localhost:8000/v1/jobs/create \
  -H "Content-Type: application/json" \
  -d '{"domain_config_name":"ai_llm","base_model":"Qwen/Qwen2.5-7B-Instruct","training_method":"sft","dataset_path":"./data/processed/your_dataset.jsonl"}'
```

---

## Google Colab

Open any notebook from `notebooks/` via GitHub:
- File -> Open notebook -> GitHub -> `valonys/finetuningtraining` -> branch `develop`

The bootstrap cell in each notebook handles cloning, dependency install, and secret pickup. No manual setup needed.

**Required Colab Secrets** (left sidebar -> key icon):
- `GITHUB_TOKEN` -- if repo is private
- `HF_TOKEN` -- for gated models and adapter push
- `OLLAMA_API_KEY` -- for Q/A synthesis

---

## CPU-only (Intel Mac / CI / any Linux)

```bash
git clone https://github.com/valonys/finetuningtraining.git
cd finetuningtraining && git checkout develop

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt

# Note: on Intel Mac, torch is capped at 2.2.2.
# transformers must be pinned: pip install "transformers>=4.44,<4.46"

cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
cd frontend && npm install && npm run dev
```

---

## Docker (CPU profile)

```bash
docker compose --profile cpu up -d
curl http://localhost:8000/healthz
```

---

## Verify everything works

After any setup path, run this checklist:

```bash
# 1. Health
curl -s http://localhost:8000/healthz | python3 -m json.tool | head -5

# 2. Domain configs
curl -s http://localhost:8000/v1/domains/configs | python3 -m json.tool

# 3. Templates
curl -s http://localhost:8000/v1/templates | python3 -m json.tool

# 4. Upload a test file
curl -X POST http://localhost:8000/v1/forge/upload \
  -F "files=@README.md"

# 5. Tests (from the repo root, in the venv)
python -m pytest
# Expect: 186+ passed
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `torch_dtype is deprecated` | transformers too new for torch version | `pip install "transformers>=4.44,<4.46"` |
| `No module named 'unsloth'` | Expected on non-Linux / CPU | Harmless -- falls back to TRL/PEFT |
| `FlashAttention2 cannot be used` | GPU is pre-Ampere (T4, V100) | Fixed in code -- auto-falls back to SDPA |
| `No inference backend could be initialised` | torch version mismatch | Check `python -c "import torch; print(torch.__version__)"` |
| Port 8000 busy | Previous uvicorn still running | `lsof -ti:8000 \| xargs kill -9` |
| Frontend can't reach backend | CORS or proxy issue | Check `vite.config.ts` proxy target points at :8000 |
