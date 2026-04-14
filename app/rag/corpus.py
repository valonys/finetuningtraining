"""
app/rag/corpus.py
-----------------
The Docs corpus the assistant retrieves from.

Each `DocArticle` has a section, a title, an id (matches the front-end
`docs.tsx` article id so links can deep-link in the future), and the
full markdown body. Bodies are written as production-grade markdown:
proper headings, lists, tables, code blocks -- exactly the structure
the assistant should emit in its responses.

If you change an article here, also update the corresponding entry in
`frontend/src/docs.tsx` so the in-app rendered docs stay aligned.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocArticle:
    section: str
    title: str
    article_id: str
    body: str

    @property
    def slug(self) -> str:
        return f"{self.section} > {self.title}"


# ----------------------------------------------------------------------
# Articles
# ----------------------------------------------------------------------
ARTICLES: list[DocArticle] = [
    # -- Getting Started ---------------------------------------------
    DocArticle(
        section="Getting Started",
        title="What is ValonyLabs Studio?",
        article_id="overview",
        body="""\
ValonyLabs Studio is a domain-agnostic **post-training** platform. You bring
raw documents and a base model (Qwen, Llama, Mistral, Gemma, Phi, DeepSeek,
...), pick a training method (SFT / DPO / ORPO / KTO / GRPO), and the Studio
handles chat-template routing, hardware-aware backend selection
(Unsloth / MLX / TRL+PEFT), and adapter hot-swapping at inference time.

### Where it runs

- Apple M-series (M1-M4) via `mlx-lm`
- NVIDIA RTX 30/40/50 and datacenter Ampere+ via `unsloth` + `vllm`
- Colab (T4/L4/A100) via `trl` + `transformers`
- NVIDIA Brev, Lambda, RunPod -- any CUDA box
- CPU-only as a last resort for tiny models (~0.5-3B)
- Ollama Cloud (Nemotron) for synth + chat without a local GPU

> The Studio does **post-training** -- fine-tuning on top of an existing
> base model. For pre-training from scratch see the FAQ article.
""",
    ),
    DocArticle(
        section="Getting Started",
        title="Installation",
        article_id="install",
        body="""\
### Backend

```bash
cd files_brevNVIDIA_v3.0
pip install -r requirements-cpu.txt      # or -mlx, or -cuda
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd files_brevNVIDIA_v3.0/frontend
npm install
npm run dev                               # -> http://localhost:5173
```

### One-liner

```bash
bash scripts/run_studio.sh                # launches both
```
""",
    ),
    DocArticle(
        section="Getting Started",
        title="Environment variables",
        article_id="env",
        body="""\
Copy `.env.example` to `.env` and fill in what you need.

### Essentials

- `OLLAMA_API_KEY` -- Ollama Cloud key, unlocks Nemotron for synth + chat
- `HF_TOKEN` -- HuggingFace token for gated models (Llama 3, Gemma)
- `VALONY_BASE_MODEL` -- default base model (HF id or Ollama tag)
- `VALONY_INFERENCE_BACKEND` -- override auto-detection (`ollama`, `vllm`, `mlx`, ...)
- `VALONY_PREWARM_INFERENCE` -- set to `1` to load the inference engine at startup;
  default is lazy (loads on first chat call)
""",
    ),

    # -- Uploading Raw Data ------------------------------------------
    DocArticle(
        section="Uploading Raw Data",
        title="Supported formats",
        article_id="formats",
        body="""\
Drop files into the Data Forge tab via drag-and-drop, or click to browse.
The parser is chosen automatically from the file extension.

### File types

| Format | Extensions | Parser |
| --- | --- | --- |
| PDF | `.pdf` | PyMuPDF + pdfplumber, OCR fallback for image-only pages |
| Word | `.docx` | python-docx, paragraphs + tables + heading detection |
| PowerPoint | `.pptx` | python-pptx, slides + tables + speaker notes |
| Excel / CSV | `.xlsx`, `.csv`, `.tsv` | pandas, per-sheet extraction |
| HTML | `.html`, `.htm` | Trafilatura + BS4 fallback |
| Text | `.txt`, `.md`, `.rst` | encoding auto-detect via chardet |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp` | OCR pipeline |
| JSON | `.json`, `.jsonl` | passthrough for pre-built datasets |

### Folders

If you pass a directory, all supported files inside it (recursively) are
ingested in one batch.

### Limits

- **512 MB per file** by default (override via `VALONY_MAX_UPLOAD_BYTES`)
- Filenames are sanitized to ASCII + a `.` `-` allowlist
""",
    ),
    DocArticle(
        section="Uploading Raw Data",
        title="OCR pipelines",
        article_id="ocr",
        body="""\
Images and text-less PDF pages route through an OCR engine. The default
engine is picked from your hardware profile, with graceful fallback:

1. **RapidOCR** -- default on CPU / MLX / consumer GPU. ONNX runtime,
   fully cross-platform.
2. **PaddleOCR** -- default on CUDA Linux. Best accuracy, CUDA-accelerated.
3. **Docling** (IBM) -- layout-aware document conversion, shines on
   multi-column technical PDFs.
4. **Tesseract** -- universal CPU fallback when nothing else is installed.
5. **TrOCR** -- transformer-based, excellent for handwriting.

Override per-call with `ocr_engine="paddleocr"` (or any other name) in
`DataForge.ingest()`.
""",
    ),
    DocArticle(
        section="Uploading Raw Data",
        title="Building a training dataset",
        article_id="build-dataset",
        body="""\
The **Data Forge** tab wraps three phases:

1. **Ingest** -- route each file to the right parser, produce
   `IngestedRecord` objects.
2. **Chunk** -- semantic chunking (heading-aware, paragraph-packing,
   target ~1200 chars per chunk).
3. **Synthesize** -- produce instruction/response pairs (SFT) or
   chosen/rejected pairs (DPO) via rule-based heuristics or the
   configured synth provider.

### Output format

The build writes **two files** per dataset under `./data/processed/`:

- `<task>_<hex>.jsonl` -- full dataset, one row per line. This is the
  primary artifact, loadable by `load_dataset("json", data_files=...)`.
- `<task>_<hex>_preview.json` -- first 10 rows pretty-printed for sneak-peek
  in any editor (`jq '.' < <preview>`).

> **Tip:** Set `OLLAMA_API_KEY` to route Q/A synthesis and DPO pair
> generation through Nemotron-3-Super on Ollama Cloud -- dramatically
> cheaper than frontier APIs and better than local 7B for synthetic
> data quality.
""",
    ),

    # -- Domains --------------------------------------------------
    DocArticle(
        section="Domains",
        title="What is a domain?",
        article_id="what-is-a-domain",
        body="""\
A **domain** is whatever the model is supposed to be -- `asset_integrity`,
`customer_grasps`, `ai_llm`, `legal_nda_review`, whatever fits your
engagement. Each domain is a YAML file in `configs/domains/` and produces
its own trained LoRA adapter at `outputs/<name>/`.

The folder ships **empty**. You create one config per use case -- nothing
is hardcoded.
""",
    ),
    DocArticle(
        section="Domains",
        title="Creating a domain",
        article_id="creating-domains",
        body="""\
Four ways to create a domain config -- pick whichever fits your workflow.

### 1. CLI

```bash
python scripts/new_domain.py create asset_integrity \\
    --system "You are an Asset Integrity engineer..." \\
    --rule "Always prioritise safety over uptime." \\
    --rule "Cite API 570 / ASME / ISO 55000."
```

### 2. REST

```http
POST /v1/domains/configs
{
  "name": "legal_nda_review",
  "system_prompt": "You are a contracts attorney...",
  "constitution": ["Flag jurisdiction clauses"]
}
```

### 3. Python

```python
from app.config_loader import create_domain_config

create_domain_config(
    name="ai_llm",
    system_prompt="You are a senior AI engineer...",
    constitution=["Cite primary sources"],
)
```

### 4. UI

Use the **Domains** tab. Fill in the form and click Create. The
copy-from-example dropdown lets you seed from an existing template.
""",
    ),

    # -- Training Methods -----------------------------------------
    DocArticle(
        section="Training Methods",
        title="SFT -- Supervised Fine-Tuning",
        article_id="sft",
        body="""\
**When to use SFT:** you have `(instruction, response)` pairs and you
want the model to produce responses like those.

### Data shape

Any of:

- `{"instruction": "...", "response": "..."}`
- `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- An already-formatted `"text"` field

### Strengths

- Fast, cheap, robust
- The workhorse for behavior shaping

### Weaknesses

- Can't learn preferences -- you have to show the model what to do, not
  what not to do
""",
    ),
    DocArticle(
        section="Training Methods",
        title="DPO / ORPO / KTO -- Preference Alignment",
        article_id="dpo-orpo-kto",
        body="""\
### DPO (Direct Preference Optimization)

Provide `(prompt, chosen, rejected)` triples. The model learns to
produce the chosen style and avoid the rejected style **without** a
separate reward model.

### ORPO

A one-stage variant that doesn't need a reference model. Slightly less
memory, similar results.

### KTO (Kahneman-Tversky Optimization)

Works with **single-sided** thumbs-up / thumbs-down labels instead of
paired preferences -- useful if you have real user feedback.

> v3.0's `pair_synthesis.py` auto-generates plausible rejections using
> Nemotron when you set `OLLAMA_API_KEY`. Without that, the fallback
> produces weak rejections (truncations + hedges) -- usable for testing,
> not for production.
""",
    ),
    DocArticle(
        section="Training Methods",
        title="GRPO -- Reinforcement Learning with Verifiable Rewards",
        article_id="grpo",
        body="""\
**GRPO** generates multiple candidates per prompt, scores each with a
reward function you supply, and updates the model to produce
higher-reward responses. **No value/critic model needed.**

### When to use

- **Math** -- ground-truth-matching reward (ships as `GSM8KRewardSignal`)
- **Code** -- unit-test-pass reward (`CodeUnitTestReward`)
- **Structured output** -- regex-extract reward (`RegexExtractReward`)
- **Function calling** -- sandbox-execution reward (roll your own)

Subclass `RewardSignal` and pass your instance to
`AgnosticGRPOTrainer(reward_signal=...)`.
""",
    ),
    DocArticle(
        section="Training Methods",
        title="Pre-training vs Post-training",
        article_id="pretraining-vs-posttraining",
        body="""\
### Short answer

This Studio does **post-training only**. It takes an existing base
model (Qwen, Llama, Mistral, ...) and adapts it via SFT / DPO / ORPO /
KTO / GRPO. It does **not** pre-train a model from scratch on a raw
corpus.

### Why not pre-training from scratch?

| Concern | Pre-training | Post-training (this Studio) |
| --- | --- | --- |
| Cost | $200K (7B) to $10M+ (70B) | $0.50-5 per SFT run |
| Data | 1-15 trillion curated tokens | 500-50,000 examples |
| Infra | Distributed FSDP / DeepSpeed cluster | Single GPU / laptop |
| Result | Often *worse* than post-training a good open base | Wins almost every time |

### Continued Pretraining (DAPT) -- the practical middle ground

If you have raw domain text and want the model to absorb domain
vocabulary **before** SFT:

1. Start from a base model (not Instruct) -- e.g., `Llama-3.1-8B`
2. Continue next-token prediction on your raw text -- no instruction
   pairs, no chat template
3. Train a few epochs on 10M-1B tokens of domain text
4. Run SFT on instruction pairs on top

Continued Pretraining is not yet a first-class method in this Studio.
The infrastructure exists; it's a matter of adding a `cpt_trainer.py`.
Tell us if it's a priority for your use case.
""",
    ),

    # -- Inference & Chat -----------------------------------------
    DocArticle(
        section="Inference & Chat",
        title="Using the chat widget",
        article_id="chat-widget",
        body="""\
The chat widget is always available in the bottom-right corner, on
every tab. Click the blue bubble to open it.

### Modes

- **Docs** (default) -- RAG over this documentation. Ask "how do I
  upload a PDF?" and the assistant answers with citations from the
  Docs section.
- **base** -- talk to the raw base model with no adapter
- **`<your-domain>`** -- talk to a trained adapter you've created

### Features

- **Streaming** -- typewriter effect via Server-Sent Events
- **Per-message telemetry** -- backend, TTFT (ms), tokens/sec under each
  response
- **Keyboard** -- Enter to send, Shift+Enter for newline
- **Settings** -- domain selector, temperature slider, max-tokens cap,
  all in a slide-down drawer inside the panel

### Telemetry

Each assistant response shows: `<backend> | <ttft>ms TTFT | <tokens> tok (<rate>/s)`.
""",
    ),
    DocArticle(
        section="Inference & Chat",
        title="Hardware backends",
        article_id="backends",
        body="""\
The inference manager auto-picks the best backend for your hardware:

- **vLLM** -- CUDA Ampere+. PagedAttention, prefix caching, FP8 KV,
  hot-swap LoRA.
- **SGLang** -- CUDA, RadixAttention, best on MoE models.
- **MLX-LM** -- Apple Silicon native.
- **llama.cpp** -- GGUF, CPU / Metal / CUDA, works everywhere.
- **HF Transformers** -- universal fallback.
- **Ollama** -- opt-in via `VALONY_INFERENCE_BACKEND=ollama`. Local
  daemon or Ollama Cloud.

Override with `VALONY_INFERENCE_BACKEND=<name>` in `.env`.

### LoRA semantics across backends

- **vLLM / SGLang / MLX / HF / llama.cpp** -- `register_adapter(name, path)`
  loads a HF-format LoRA on top of the base model and hot-swaps per
  request via `domain_name`.
- **Ollama** -- `register_adapter(name, ollama_tag)` maps a domain to a
  *different Ollama model tag*. No LoRA hot-swap; you pre-pull the
  Ollama models you need.
""",
    ),

    # -- Troubleshooting ------------------------------------------
    DocArticle(
        section="Troubleshooting",
        title='"API unreachable" on the Health tab',
        article_id="api-unreachable",
        body="""\
If the Health tab shows a red "API unreachable" banner, the FastAPI
backend returned a 500 (or isn't running).

### Steps

1. Confirm uvicorn is running: `curl http://localhost:8000/healthz`
2. Check the uvicorn terminal for a Python traceback -- that tells you
   which dep is missing.
3. Install the full deps for your platform:
   `pip install -r requirements-cpu.txt` (or `-mlx`, or `-cuda`).
4. Restart uvicorn and refresh the browser.

> Since v3.0.1, `/healthz` degrades gracefully -- missing optional deps
> show as `{"error": "..."}` fields in the response instead of returning
> 500. If you still see the red banner, the backend isn't running at all.
""",
    ),
    DocArticle(
        section="Training Metrics",
        title="Interpreting SFT training loss",
        article_id="sft-loss",
        body="""\
The SFT trainer logs a `loss` value at each step. Here is how to read
it and decide whether the run is healthy.

### Typical final loss ranges (7B base + LoRA)

| Dataset size | Expected final train loss | Warning signs |
| --- | --- | --- |
| 1K--10K pairs | 0.8 -- 1.5 | below 0.3 suggests memorization |
| 10K--100K pairs | 0.5 -- 1.2 | below 0.3 suggests memorization |
| 100K+ pairs | 0.4 -- 1.0 | below 0.3 suggests memorization |

These are **soft** ranges. Tokenizer choice, chat template overhead,
and dataset linguistic diversity all shift the floor up or down by
0.2 -- 0.4.

### What 1.5 -- 1.8 final loss means on a 65K dataset

On the high end of healthy. Not broken, but **under-converged**. In
priority order, the usual causes are:

1. **LoRA rank too low.** If `lora_r=8`, bump to `16` or `32`. Loss
   usually falls ~0.3 per doubling.
2. **Learning rate too low.** `2e-4` is standard for SFT with LoRA.
   If you are at `5e-5`, you are leaving progress on the table.
3. **Not enough epochs.** One epoch on 65K is often insufficient;
   try 2 -- 3.
4. **Chat template mismatch.** A ChatML-formatted dataset fed to a
   Llama-2 base will plateau high because the model is also learning
   delimiters. Check `app/templates/registry.py`.
5. **Dataset heterogeneity.** If the 65K pairs span wildly different
   styles, loss cannot go below ~1.2 because the target distribution
   is genuinely multimodal. This is fine; evaluation quality is what
   matters.

### Loss-curve patterns

| Pattern | Diagnosis |
| --- | --- |
| Loss decreasing smoothly | Healthy |
| Loss flat after step ~100 | LR too low, plateau, or not enough epochs |
| Loss spiky or oscillating | LR too high -- halve it |
| Train loss 0.3, eval loss 2.0 | Overfitting -- reduce epochs, add dropout |
| Both losses plateau high | Dataset quality issue or wrong base model |

### Training loss is a weak quality metric

Prioritise in this order:

1. **Manual spot-checks.** Send the trained adapter 5--10 representative
   prompts via the chat widget (select it from the dropdown). Compare
   directly to the base model. This is the only signal that matters for
   commercial rollout.
2. **Eval loss on a held-out 10% split.** If eval loss is lower than
   before training AND train loss is still decreasing, you are fine
   regardless of the absolute value.
3. **Task-specific metrics.** Exact-match on math, pass@1 on code,
   ROUGE on summarization, human rating on chat.
4. **Training loss.** Useful for "is it progressing" but poor for
   "is it good".

> A model that ends at train loss 1.7 but produces **excellent
> responses** is worth infinitely more than one that ends at 0.3 but
> has memorized its training set.
""",
    ),
    DocArticle(
        section="Troubleshooting",
        title="Out of memory during training",
        article_id="oom",
        body="""\
The profile resolver auto-picks LoRA rank, sequence length, and batch
size for your hardware, but if you're getting OOM you can override in
your domain config:

```yaml
training_args:
  lora_r: 8                           # smaller (was 16)
  max_seq_length: 1024                # smaller (was 2048)
  batch_size: 1                       # always safest
  gradient_accumulation_steps: 16     # compensates for small batch
```

On Apple Silicon, close other GPU-heavy apps (Xcode, Lightroom, Chrome
video) before training -- unified memory is shared.
""",
    ),

    # -- Architecture / Reference ---------------------------------
    DocArticle(
        section="Inference & Chat",
        title="Streaming chat (SSE)",
        article_id="streaming",
        body="""\
The chat widget uses Server-Sent Events for the typewriter effect via
`POST /v1/chat/stream`. Frame protocol:

```
data: {"sources": [{"section":"Getting Started","title":"Installation"}, ...]}
data: {"delta": "Hello"}
data: {"delta": " world"}
data: {"meta": {"backend":"ollama","model":"nemotron-3-super","ttft_ms":1034,...}}
data: [DONE]
```

The `sources` frame appears only in **Docs mode** (RAG). Other frames
appear in every chat call.

### Curling the stream

```bash
curl -N -X POST http://localhost:8000/v1/chat/stream \\
    -H 'Content-Type: application/json' \\
    -d '{"message":"Hi","domain_config_name":"docs","temperature":0.4,"max_new_tokens":200}'
```

`-N` disables curl's own buffering so deltas reach stdout immediately.

### Backend support

Currently only the `ollama` backend exposes `stream()`. Other backends
(vLLM, MLX, HF, llama.cpp) return HTTP 501 on `/v1/chat/stream` -- they
all *can* stream, the wrappers just aren't written yet. Use the
batched `/v1/chat` endpoint with those backends until then.
""",
    ),
]


# -- Convenience helpers ---------------------------------------------
def find_article(article_id: str) -> DocArticle | None:
    """Look up an article by its `article_id`."""
    for a in ARTICLES:
        if a.article_id == article_id:
            return a
    return None


def all_sections() -> list[str]:
    seen, out = set(), []
    for a in ARTICLES:
        if a.section not in seen:
            seen.add(a.section)
            out.append(a.section)
    return out
