# ValonyLabs Studio v3.0 — Agnostic Post-Training & Inference Platform

**A local, vendor-agnostic fine-tuning + inference studio in the spirit of
[Unsloth Studio](https://unsloth.ai/docs/new/studio), runnable on:**

| Tier | Hardware | Training backend | Inference backend |
|------|----------|------------------|-------------------|
| Edge / Laptop | Apple M4 Pro (24 / 36 GB unified) | `mlx-lm` + LoRA | `mlx-lm` / `llama.cpp` |
| Consumer GPU | NVIDIA RTX 5090 (32 GB), 4090 (24 GB) | `unsloth` + `trl` | `vllm` / `sglang` |
| Notebook | Google Colab (T4, L4, A100) | `unsloth` + `trl` | `vllm` / HF `transformers` |
| Cloud | NVIDIA Brev (A10G → H100), Lambda, RunPod | `unsloth` + `trl` (+ FSDP) | `vllm` / `sglang` / TRT-LLM |
| CPU-only | any x86/arm64 | `trl` + `peft` (tiny models) | `llama.cpp` / `transformers` |

All configuration is a single `domain_config.yaml` and everything in between
(hardware detection, template routing, trainer selection, inference engine
selection) happens **automatically** — you only change code when you want to.

---

## What's new in v3.0 (vs v2.0)

| Capability | v2.0 | v3.0 |
|------------|------|------|
| Platforms | Brev (CUDA) only | Mac M-series · RTX 30/40/50 · Colab · Brev · CPU |
| Training backends | Unsloth + TRL | **Unsloth · MLX-LM · TRL/PEFT (auto-routed)** |
| Training methods | SFT · GRPO | **SFT · DPO · ORPO · KTO · GRPO · Reward modelling** |
| Dataset formats | Local JSON/CSV · HF Hub | **+ PDF · DOCX · TXT · XLSX · PPTX · HTML · MD · Images (OCR)** |
| OCR | — | **Tesseract · RapidOCR · PaddleOCR · Docling · TrOCR (auto-selected)** |
| Chat templates | Alpaca-only | **Auto-detected per base model**: Alpaca · ChatML · ShareGPT · Llama-2/3 · Qwen2/2.5/3 · Mistral · Gemma · Phi · DeepSeek |
| Inference engines | vLLM + HF fallback | **vLLM · SGLang · MLX · llama.cpp · HF (auto-routed)** |
| TTFT optimisations | KV cache only | **KV cache · RadixAttention · prefix cache · KV-quant (FP8) · speculative decoding · chunked prefill** |
| TTFT target | — | **20–50 ms/token (depending on tier)** |
| Studio UI | — | **Gradio studio** (`ui/studio.py`) |

---

## Directory layout

```
files_brevNVIDIA_v3.0/
├── README.md                       ← you are here
├── pyproject.toml                  ← uv / pip project
├── requirements-cuda.txt           ← RTX / Brev / Colab
├── requirements-mlx.txt            ← Apple Silicon
├── requirements-cpu.txt            ← Minimal CPU fallback
├── app/
│   ├── main.py                     ← FastAPI server
│   ├── models.py                   ← Pydantic schemas
│   ├── config_loader.py            ← Domain-config loader
│   ├── hardware/
│   │   ├── detect.py               ← Detect CUDA / MPS / CPU, pick tier
│   │   └── profiles.py             ← VRAM-aware defaults per tier
│   ├── data_forge/                 ← The "Data Forge" — multi-format ingestion
│   │   ├── ingest.py               ← Orchestrator
│   │   ├── parsers/                ← One parser per input type
│   │   │   ├── pdf.py              ← PyMuPDF + pdfplumber
│   │   │   ├── docx.py             ← python-docx
│   │   │   ├── xlsx.py             ← openpyxl / pandas
│   │   │   ├── pptx.py             ← python-pptx
│   │   │   ├── html.py             ← trafilatura / BS4
│   │   │   ├── txt.py              ← encoding-sniffing plain text
│   │   │   └── image.py            ← routes to OCR
│   │   ├── ocr/                    ← SOTA OCR pipeline (routable)
│   │   │   ├── base.py             ← OCR engine ABC
│   │   │   ├── pipeline.py         ← layout → text → tables → export
│   │   │   ├── rapidocr_engine.py  ← RapidOCR (ONNX, cross-platform)
│   │   │   ├── paddleocr_engine.py ← PaddleOCR (CUDA / CPU)
│   │   │   ├── docling_engine.py   ← IBM Docling (layout-aware)
│   │   │   ├── tesseract_engine.py ← Tesseract (CPU fallback)
│   │   │   └── trocr_engine.py     ← TrOCR (transformer, handwriting)
│   │   ├── chunker.py              ← Semantic + structural chunking
│   │   ├── qa_synthesis.py         ← Generate SFT Q/A pairs from docs
│   │   └── dataset_builder.py      ← Normalise → SFT/DPO/GRPO schemas
│   ├── templates/                  ← Chat-template registry
│   │   ├── registry.py             ← Model-id → template resolver
│   │   ├── alpaca.py
│   │   ├── chatml.py
│   │   ├── sharegpt.py
│   │   ├── qwen.py
│   │   ├── llama.py
│   │   ├── mistral.py
│   │   ├── gemma.py
│   │   ├── phi.py
│   │   └── deepseek.py
│   ├── trainers/
│   │   ├── base.py                 ← Trainer ABC
│   │   ├── backends.py             ← Unsloth / MLX / TRL selector
│   │   ├── sft_trainer.py
│   │   ├── dpo_trainer.py
│   │   ├── orpo_trainer.py
│   │   ├── kto_trainer.py
│   │   ├── grpo_trainer.py         ← GSM8K-style verifiable reward
│   │   └── reward_signals.py
│   ├── inference/
│   │   ├── manager.py              ← Router
│   │   ├── vllm_backend.py
│   │   ├── sglang_backend.py
│   │   ├── mlx_backend.py
│   │   ├── llamacpp_backend.py
│   │   ├── hf_backend.py
│   │   └── cache/
│   │       ├── prefix_cache.py     ← Cross-request cache manager
│   │       └── kv_cache.py
│   └── evaluation/
│       ├── runners.py
│       ├── metrics.py
│       └── leaderboard.py
├── configs/
│   ├── domains/
│   │   ├── _template.yaml          ← Annotated blueprint for new domains
│   │   └── examples/               ← Seed examples (never auto-loaded)
│   │       ├── asset_integrity.yaml
│   │       ├── customer_grasps.yaml
│   │       └── ai_llm.yaml
│   ├── models/
│   │   └── model_catalog.yaml      ← Supported base models + default template
│   └── hardware/
│       ├── m4_pro.yaml
│       ├── rtx_5090.yaml
│       ├── rtx_4090.yaml
│       ├── colab_t4.yaml
│       ├── brev_a10g.yaml
│       └── brev_a100.yaml
├── scripts/
│   ├── install.sh                  ← Auto-detects HW, installs right stack
│   ├── run_studio.sh               ← Launches FastAPI + Gradio
│   └── preflight.py                ← HW check, model download, smoke test
├── ui/
│   └── studio.py                   ← Gradio UI
├── notebooks/
│   ├── 01_data_forge_demo.ipynb
│   ├── 02_sft_qwen_alpaca.ipynb
│   ├── 03_dpo_llama3_identity.ipynb
│   └── 04_grpo_math_reasoning.ipynb
├── data/
│   ├── uploads/                    ← drop files here
│   └── processed/                  ← normalised datasets land here
└── outputs/                        ← adapters, merged models, logs
```

---

## Installation

The installer auto-detects your platform and installs the matching stack.

```bash
cd files_brevNVIDIA_v3.0
bash scripts/install.sh
```

If you prefer manual, pick one of:

```bash
# Apple Silicon (M1/M2/M3/M4)
pip install -r requirements-mlx.txt

# NVIDIA GPU (RTX 30/40/50, Colab, Brev)
pip install -r requirements-cuda.txt

# CPU-only
pip install -r requirements-cpu.txt
```

---

## Domains — you define them, per engagement

A **domain** is anything you want the model to be. No default is hardcoded.
You create one domain per engagement (`asset_integrity`, `customer_grasps`,
`ai_llm`, `legal_nda_review`, `medical_intake`, `recipes`, ...) and each one
produces its own LoRA adapter at `outputs/<domain_name>/` that can be trained,
swapped, and served independently against any base model.

`configs/domains/` ships **empty**. You create configs via any of:

```bash
# 1) CLI (argparse, no deps)
python scripts/new_domain.py create asset_integrity \
    --system "You are an Asset Integrity engineer specialising in FPSO inspections..." \
    --rule  "Always prioritise safety over uptime." \
    --rule  "Cite API 570 / ASME / ISO 55000 where relevant."

# 2) Seed from a shipped example (asset_integrity / customer_grasps / ai_llm)
python scripts/new_domain.py create my_support --copy-from customer_grasps

# 3) List what's available (user configs + seed examples)
python scripts/new_domain.py list
```

```python
# 4) Python — from a notebook or script
from app.config_loader import create_domain_config, domain_config_exists

if not domain_config_exists("ai_llm"):
    create_domain_config(
        name="ai_llm",
        system_prompt="You are a senior AI engineer specialising in LLM post-training.",
        constitution=["Cite primary sources", "Prefer reproducible benchmarks"],
    )
```

```bash
# 5) REST API
curl -X POST http://localhost:8000/v1/domains/configs \
    -H 'content-type: application/json' \
    -d '{
        "name": "legal_nda_review",
        "system_prompt": "You are a contracts attorney reviewing NDAs...",
        "constitution": ["Flag jurisdiction clauses", "Never give legal advice"]
    }'

# List, read, template
curl http://localhost:8000/v1/domains/configs
curl http://localhost:8000/v1/domains/configs/legal_nda_review
curl http://localhost:8000/v1/domains/template
```

Or use the Gradio Studio's **🏷️ Domains** tab — form-based create, a live
preview of existing YAMLs, and dropdowns on the Train/Chat tabs that refresh
automatically when you add a new domain.

The full blueprint with explanatory comments lives at
`configs/domains/_template.yaml`. Name validation is strict:
`^[a-z][a-z0-9_]{0,63}$` (filesystem-safe, identifier-safe).

---

## Quick start — Data Forge

```python
from app.data_forge.ingest import DataForge

forge = DataForge()

# Ingest anything — PDF, DOCX, XLSX, PPTX, HTML, images, folders...
records = forge.ingest("./data/uploads/api_570_spec.pdf")

# Build an SFT dataset with auto-generated Q/A pairs from the doc
dataset = forge.build_dataset(
    records,
    task="sft",                      # "sft" | "dpo" | "grpo"
    base_model="Qwen/Qwen2.5-7B-Instruct",  # auto-selects Qwen chat template
    synth_qa=True,                    # synthesise Q/A pairs via an LLM
    target_size=500,
)

dataset.save_to_disk("./data/processed/asset_integrity_sft")
```

Images are routed through the OCR pipeline (layout → text → table extraction):

```python
records = forge.ingest("./data/uploads/inspection_report.png")
# → {"text": "...", "tables": [...], "layout": {...}, "source": "..."}
```

---

## Quick start — training

```python
from app.config_loader import create_domain_config, load_domain_config, domain_config_exists
from app.trainers.sft_trainer import AgnosticSFTTrainer

DOMAIN = "ai_llm"   # ← any name you like (asset_integrity, customer_grasps, legal_nda_review...)

# Create the domain config on the fly if it doesn't exist yet
if not domain_config_exists(DOMAIN):
    create_domain_config(
        name=DOMAIN,
        system_prompt="You are a senior AI engineer specialising in LLM post-training.",
        constitution=["Cite primary sources", "Prefer reproducible benchmarks"],
    )

config = load_domain_config(DOMAIN)

trainer = AgnosticSFTTrainer(
    config=config,
    base_model_id="Qwen/Qwen2.5-7B-Instruct",
    dataset_path=f"./data/processed/{DOMAIN}_sft",
)
result = trainer.train()
print(result)   # → {adapter_path: 'outputs/ai_llm', method, backend, final_loss, ...}
```

The trainer will:

1. Detect the hardware tier (`hardware.detect`)
2. Pick the right backend (`Unsloth` on CUDA, `MLX-LM` on Apple, `TRL+PEFT` fallback)
3. Resolve the **chat template** from the base model ID (`templates.registry`)
4. Format the dataset with that template automatically
5. Pick VRAM-safe defaults (LoRA rank, seq len, batch size)
6. Save the adapter to `outputs/{domain_name}/`

Swap `AgnosticSFTTrainer` for `AgnosticDPOTrainer`, `AgnosticGRPOTrainer`,
`AgnosticORPOTrainer`, or `AgnosticKTOTrainer` — same interface.

---

## Quick start — inference

```python
from app.inference.manager import get_inference_engine

engine = get_inference_engine("Qwen/Qwen2.5-7B-Instruct")

# Register any trained adapter — the domain name is yours to choose
engine.register_adapter("ai_llm",           "./outputs/ai_llm")
engine.register_adapter("customer_grasps",  "./outputs/customer_grasps")
engine.register_adapter("legal_nda_review", "./outputs/legal_nda_review")

# Route at inference time by domain_name
response = engine.generate(
    prompt="Explain speculative decoding and when it helps vs. hurts.",
    domain_name="ai_llm",
    temperature=0.4,
    max_new_tokens=512,
)
```

The manager routes to the best available backend:

- **vLLM** if CUDA (Ampere+) and `vllm` installed → RadixAttention + paged KV
- **SGLang** if CUDA and `sglang[all]` installed → `sgl.gen` with prefix cache
- **MLX-LM** if Apple Silicon and `mlx-lm` installed
- **llama.cpp** if `llama-cpp-python` installed and a GGUF is available
- **HF transformers** as the always-available fallback

All backends share the same `generate()` interface.

---

## TTFT targets

Tier | Target p50 TTFT | Key optimisations
-----|-----------------|------------------
M4 Pro (MLX, 7B 4-bit) | 35–45 ms/token | MLX quant, KV cache, small model
RTX 5090 (vLLM / SGLang, 7B FP8) | 20–30 ms/token | RadixAttention, FP8 KV, FlashAttention-3, speculative decoding
Colab T4 (HF, 7B 4-bit) | 80–120 ms/token | bitsandbytes 4-bit, chunked prefill
Brev A10G (vLLM, 13B FP8) | 30–45 ms/token | paged attention, continuous batching
Brev H100 (TRT-LLM / vLLM, 70B FP8) | 15–25 ms/token | TP=2, FP8, speculative + continuous batching

The inference engine measures and streams p50/p90/p99 TTFT via `/healthz/latency`.

---

## Supported base models (auto-routed templates)

Family | Examples | Template resolver
-------|----------|------------------
Qwen | `Qwen/Qwen2.5-*`, `Qwen/Qwen3-*` | `qwen.QwenChatTemplate`
Llama | `meta-llama/Llama-3.*`, `unsloth/llama-3-*` | `llama.Llama3ChatTemplate`
Llama 2 | `meta-llama/Llama-2-*` | `llama.Llama2ChatTemplate`
Mistral | `mistralai/Mistral-*`, `mistralai/Mixtral-*` | `mistral.MistralChatTemplate`
Gemma | `google/gemma-*`, `google/gemma-2-*`, `google/gemma-3-*` | `gemma.GemmaChatTemplate`
Phi | `microsoft/Phi-3-*`, `microsoft/Phi-4-*` | `phi.PhiChatTemplate`
DeepSeek | `deepseek-ai/DeepSeek-R1-*`, `DeepSeek-V2-*` | `deepseek.DeepSeekChatTemplate`
Generic | Alpaca, ChatML, ShareGPT | `alpaca` / `chatml` / `sharegpt`

When you pick `Qwen/Qwen2.5-7B-Instruct` for SFT over a `{question, answer}`
dataset, the pipeline **automatically** wraps each row in the Qwen ChatML
format, applies the correct special tokens, and trains only on the response
tokens (via `DataCollatorForCompletionOnlyLM`).

---

## Design principles

1. **Agnostic at the core** — no vendor lock-in. If a dependency is missing,
   we fall back to a simpler one. CUDA is never assumed.
2. **One config, many backends** — the same `domain_config.yaml` trains on
   M4 Pro and serves on Brev without edits.
3. **Fail loud, recover soft** — preflight checks warn early; runtime
   fallbacks keep the UI usable.
4. **Templates are first-class** — bad chat templates are the #1 cause of
   silent training failure. The registry is the single source of truth.
5. **TTFT is a metric, not a slogan** — every backend reports p50/p90/p99.

---

## Credits & references

- Unsloth — https://github.com/unslothai/unsloth  (training speedups we mirror)
- Unsloth Studio — https://unsloth.ai/docs/new/studio  (UX inspiration)
- SGLang — RadixAttention, structured generation
- vLLM — PagedAttention, continuous batching
- HuggingFace TRL — SFT / DPO / GRPO / ORPO / KTO reference implementations
- IBM Docling — layout-aware document parsing
- Washington University *Intro to Post-Training* — curriculum grounding
- *Inference Engineering* (Brev/NVIDIA) — TTFT optimisation playbook
