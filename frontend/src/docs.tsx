/**
 * Documentation content for the in-app Docs sidebar.
 *
 * Each DocSection is a navigable chapter; each DocArticle inside it is a
 * single page rendered in the right-hand pane. Articles are plain React
 * components so we can embed links, inline code, lists, tables, etc.
 * without adding an MDX / markdown renderer dependency.
 */
import type { ReactNode } from "react";

export interface DocArticle {
  id: string;
  title: string;
  summary: string;
  body: ReactNode;
}

export interface DocSection {
  id: string;
  title: string;
  articles: DocArticle[];
}

/* ── Reusable prose helpers ────────────────────────────────── */
const H = ({ children }: { children: ReactNode }) => (
  <h2 className="text-lg font-semibold text-gray-900 mt-6 mb-2">{children}</h2>
);
const H3 = ({ children }: { children: ReactNode }) => (
  <h3 className="text-sm font-semibold uppercase tracking-wide text-gray-500 mt-5 mb-2">{children}</h3>
);
const P = ({ children }: { children: ReactNode }) => (
  <p className="text-sm text-gray-700 leading-relaxed mb-3">{children}</p>
);
const Code = ({ children }: { children: ReactNode }) => (
  <code className="px-1.5 py-0.5 rounded bg-gray-100 text-[12.5px] font-mono text-gray-800">{children}</code>
);
const Pre = ({ children }: { children: ReactNode }) => (
  <pre className="bg-gray-900 text-green-300 text-xs font-mono rounded-lg p-4 overflow-x-auto mb-3 whitespace-pre">
    {children}
  </pre>
);
const Ul = ({ children }: { children: ReactNode }) => (
  <ul className="list-disc list-outside ml-5 text-sm text-gray-700 space-y-1 mb-3">{children}</ul>
);
const Callout = ({ tone, children }: { tone: "info" | "warn" | "tip"; children: ReactNode }) => {
  const map = {
    info: "bg-blue-50 border-blue-300 text-blue-900",
    warn: "bg-amber-50 border-amber-300 text-amber-900",
    tip:  "bg-emerald-50 border-emerald-300 text-emerald-900",
  } as const;
  return (
    <div className={`border-l-4 rounded-r px-4 py-2 text-sm mb-3 ${map[tone]}`}>{children}</div>
  );
};

/* ── Sections ──────────────────────────────────────────────── */
export const DOC_SECTIONS: DocSection[] = [
  {
    id: "getting-started",
    title: "Getting Started",
    articles: [
      {
        id: "overview",
        title: "What is ValonyLabs Studio?",
        summary: "Agnostic post-training + inference platform.",
        body: (
          <>
            <P>
              ValonyLabs Studio is a domain-agnostic <b>post-training</b> platform. You bring
              raw documents and a base model (Qwen, Llama, Mistral, Gemma, Phi, DeepSeek, ...),
              pick a training method (SFT / DPO / ORPO / KTO / GRPO), and the Studio handles
              chat-template routing, hardware-aware backend selection (Unsloth / MLX /
              TRL+PEFT), and adapter hot-swapping at inference time.
            </P>
            <H>Where it runs</H>
            <Ul>
              <li>Apple M-series (M1–M4) via <Code>mlx-lm</Code></li>
              <li>NVIDIA RTX 30/40/50 and datacenter Ampere+ via <Code>unsloth</Code> + <Code>vllm</Code></li>
              <li>Colab (T4/L4/A100) via <Code>trl</Code> + <Code>transformers</Code></li>
              <li>NVIDIA Brev, Lambda, RunPod — any CUDA box</li>
              <li>CPU-only as a last resort for tiny models (~0.5–3B)</li>
              <li>Ollama Cloud (Nemotron) for synth + chat without a local GPU</li>
            </Ul>
            <Callout tone="info">
              This Studio does <b>post-training</b> (fine-tuning on top of an existing base
              model). For the distinction versus pre-training from scratch, see the FAQ
              article "Pre-training vs post-training".
            </Callout>
          </>
        ),
      },
      {
        id: "install",
        title: "Installation",
        summary: "Backend + frontend setup.",
        body: (
          <>
            <H>Backend</H>
            <Pre>{`cd files_brevNVIDIA_v3.0
pip install -r requirements-cpu.txt      # or -mlx, or -cuda
uvicorn app.main:app --reload --port 8000`}</Pre>
            <H>Frontend</H>
            <Pre>{`cd files_brevNVIDIA_v3.0/frontend
npm install
npm run dev                               # → http://localhost:5173`}</Pre>
            <H>One-liner</H>
            <Pre>{`bash scripts/run_studio.sh                # launches both`}</Pre>
          </>
        ),
      },
      {
        id: "env",
        title: "Environment variables",
        summary: "Token, model, backend overrides.",
        body: (
          <>
            <P>Copy <Code>.env.example</Code> to <Code>.env</Code> and fill in what you need.</P>
            <H>Essentials</H>
            <Ul>
              <li><Code>OLLAMA_API_KEY</Code> — Ollama Cloud key, unlocks Nemotron for synth + chat</li>
              <li><Code>HF_TOKEN</Code> — HuggingFace token for gated models (Llama 3, Gemma)</li>
              <li><Code>VALONY_BASE_MODEL</Code> — default base model (HF id or Ollama tag)</li>
              <li><Code>VALONY_INFERENCE_BACKEND</Code> — override auto-detection (<Code>ollama</Code>, <Code>vllm</Code>, <Code>mlx</Code>, ...)</li>
            </Ul>
          </>
        ),
      },
    ],
  },

  {
    id: "uploading-data",
    title: "Uploading Raw Data",
    articles: [
      {
        id: "formats",
        title: "Supported formats",
        summary: "Everything Data Forge can ingest.",
        body: (
          <>
            <P>Drop files into <Code>./data/uploads/</Code> (or pass any path to the Data Forge tab). The parser is chosen automatically from the file extension.</P>
            <H>File types</H>
            <Ul>
              <li><b>PDF</b> (<Code>.pdf</Code>) — PyMuPDF for text layers, pdfplumber for tables. Empty-text pages fall back to OCR automatically.</li>
              <li><b>Word</b> (<Code>.docx</Code>) — python-docx, extracts paragraphs and tables with heading detection.</li>
              <li><b>PowerPoint</b> (<Code>.pptx</Code>) — python-pptx, extracts slide text, tables, and speaker notes.</li>
              <li><b>Excel / CSV / TSV</b> (<Code>.xlsx</Code>, <Code>.xls</Code>, <Code>.csv</Code>, <Code>.tsv</Code>) — pandas, per-sheet extraction.</li>
              <li><b>HTML</b> (<Code>.html</Code>, <Code>.htm</Code>) — Trafilatura for clean content, BS4 fallback.</li>
              <li><b>Text / Markdown</b> (<Code>.txt</Code>, <Code>.md</Code>, <Code>.rst</Code>) — encoding auto-detection via chardet.</li>
              <li><b>Images</b> (<Code>.png</Code>, <Code>.jpg</Code>, <Code>.jpeg</Code>, <Code>.webp</Code>, <Code>.tiff</Code>, <Code>.bmp</Code>) — routed through the OCR pipeline.</li>
              <li><b>JSON / JSONL</b> — passthrough, useful for pre-built datasets.</li>
            </Ul>
            <H>Folders</H>
            <P>If you pass a directory, all supported files inside it (recursively) are ingested in one batch.</P>
          </>
        ),
      },
      {
        id: "ocr",
        title: "OCR pipelines",
        summary: "Which engine handles images and scanned PDFs.",
        body: (
          <>
            <P>Images and text-less PDF pages route through an OCR engine. The default engine is picked from your hardware profile and falls back gracefully:</P>
            <Ul>
              <li><b>RapidOCR</b> (default on CPU / MLX / consumer GPU) — ONNX runtime, fully cross-platform.</li>
              <li><b>PaddleOCR</b> (default on CUDA Linux) — best accuracy, CUDA-accelerated.</li>
              <li><b>Docling</b> (IBM) — layout-aware document conversion, shines on multi-column technical PDFs.</li>
              <li><b>Tesseract</b> — universal CPU fallback when nothing else is installed.</li>
              <li><b>TrOCR</b> — transformer-based, excellent for handwriting.</li>
            </Ul>
            <P>Override per-call with <Code>ocr_engine="paddleocr"</Code> (or any other name) in <Code>DataForge.ingest()</Code>.</P>
          </>
        ),
      },
      {
        id: "build-dataset",
        title: "Building a training dataset",
        summary: "From raw docs to HF Dataset ready for TRL.",
        body: (
          <>
            <P>The <b>Data Forge</b> tab wraps three phases:</P>
            <Ul>
              <li><b>Ingest</b> — route each file to the right parser, produce <Code>IngestedRecord</Code>s.</li>
              <li><b>Chunk</b> — semantic chunking (heading-aware, paragraph-packing, target ~1200 chars per chunk).</li>
              <li><b>Synthesize</b> — produce instruction/response pairs (SFT) or chosen/rejected pairs (DPO) via rule-based heuristics or the configured synth provider.</li>
            </Ul>
            <Callout tone="tip">
              Set <Code>OLLAMA_API_KEY</Code> to route Q/A synthesis and DPO pair generation
              through <b>Nemotron-70B</b> on Ollama Cloud — dramatically cheaper than frontier
              APIs and better than any local 7B for synthetic data quality.
            </Callout>
          </>
        ),
      },
    ],
  },

  {
    id: "domains",
    title: "Domains",
    articles: [
      {
        id: "what-is-a-domain",
        title: "What is a domain?",
        summary: "One YAML config per engagement.",
        body: (
          <>
            <P>
              A <b>domain</b> is whatever the model is supposed to be — <Code>asset_integrity</Code>,
              <Code>customer_grasps</Code>, <Code>ai_llm</Code>, <Code>legal_nda_review</Code>, whatever
              fits your engagement. Each domain is a YAML file in <Code>configs/domains/</Code> and
              produces its own trained LoRA adapter at <Code>outputs/&lt;name&gt;/</Code>.
            </P>
            <P>
              The folder ships empty. You create one config per use case — nothing is hardcoded.
            </P>
          </>
        ),
      },
      {
        id: "creating-domains",
        title: "Creating a domain",
        summary: "CLI, REST, Python, UI — pick one.",
        body: (
          <>
            <H3>CLI</H3>
            <Pre>{`python scripts/new_domain.py create asset_integrity \\
    --system "You are an Asset Integrity engineer..." \\
    --rule "Always prioritise safety over uptime." \\
    --rule "Cite API 570 / ASME / ISO 55000."`}</Pre>

            <H3>REST</H3>
            <Pre>{`POST /v1/domains/configs
{
  "name": "legal_nda_review",
  "system_prompt": "You are a contracts attorney...",
  "constitution": ["Flag jurisdiction clauses"]
}`}</Pre>

            <H3>Python</H3>
            <Pre>{`from app.config_loader import create_domain_config

create_domain_config(
    name="ai_llm",
    system_prompt="You are a senior AI engineer...",
    constitution=["Cite primary sources"],
)`}</Pre>

            <H3>UI</H3>
            <P>Use the <b>Domains</b> tab. Fill in the form and click Create. Copy-from-example is a dropdown there too.</P>
          </>
        ),
      },
    ],
  },

  {
    id: "training",
    title: "Training Methods",
    articles: [
      {
        id: "sft",
        title: "SFT — Supervised Fine-Tuning",
        summary: "Teach the model to imitate example responses.",
        body: (
          <>
            <P><b>When</b>: you have <i>(instruction, response)</i> pairs and you want the model to produce responses like those.</P>
            <P><b>Data shape</b>: any of</P>
            <Ul>
              <li><Code>{`{"instruction": "...", "response": "..."}`}</Code></li>
              <li><Code>{`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`}</Code></li>
              <li>Or an already-formatted <Code>"text"</Code> field</li>
            </Ul>
            <P><b>Strengths</b>: fast, cheap, robust. The workhorse for behavior shaping.</P>
            <P><b>Weaknesses</b>: can't learn preferences — you have to show it what to do, not what not to do.</P>
          </>
        ),
      },
      {
        id: "dpo-orpo-kto",
        title: "DPO / ORPO / KTO — Preference Alignment",
        summary: "Learn from contrastive examples.",
        body: (
          <>
            <P><b>DPO</b> (Direct Preference Optimization): you provide <i>(prompt, chosen, rejected)</i> triples. The model learns to produce the chosen style and avoid the rejected style without a separate reward model.</P>
            <P><b>ORPO</b>: a one-stage variant that doesn't need a reference model. Slightly less memory, similar results.</P>
            <P><b>KTO</b> (Kahneman-Tversky Optimization): works with single-sided thumbs-up/thumbs-down labels instead of paired preferences — useful if you have real user feedback.</P>
            <Callout tone="tip">
              v3.0's <Code>pair_synthesis.py</Code> auto-generates plausible rejections using
              Nemotron when you set <Code>OLLAMA_API_KEY</Code>. Without that, the fallback
              produces weak rejections (truncations + hedges) — usable for testing, not for
              production.
            </Callout>
          </>
        ),
      },
      {
        id: "grpo",
        title: "GRPO — Reinforcement Learning with Verifiable Rewards",
        summary: "When you can score answers programmatically.",
        body: (
          <>
            <P>
              <b>GRPO</b> generates multiple candidates per prompt, scores each with a reward
              function you supply, and updates the model to produce higher-reward responses.
              No value/critic model needed.
            </P>
            <H>When to use</H>
            <Ul>
              <li><b>Math</b> — ground-truth-matching reward (ships as <Code>GSM8KRewardSignal</Code>)</li>
              <li><b>Code</b> — unit-test-pass reward (<Code>CodeUnitTestReward</Code>)</li>
              <li><b>Structured output</b> — regex-extract reward (<Code>RegexExtractReward</Code>)</li>
              <li><b>Function calling</b> — sandbox-execution reward (roll your own)</li>
            </Ul>
            <P>Subclass <Code>RewardSignal</Code> and pass your instance to <Code>AgnosticGRPOTrainer(reward_signal=...)</Code>.</P>
          </>
        ),
      },
      {
        id: "pretraining-vs-posttraining",
        title: "Pre-training vs Post-training",
        summary: "Why this Studio doesn't do pre-training from scratch.",
        body: (
          <>
            <H>The short answer</H>
            <P>
              <b>This Studio does post-training only.</b> It takes an existing base model
              (Qwen, Llama, Mistral, ...) and adapts it via SFT / DPO / ORPO / KTO / GRPO.
              It does <b>not</b> pre-train a model from scratch on a raw corpus.
            </P>
            <H>Why not pre-training from scratch?</H>
            <Ul>
              <li><b>Cost</b> — GPT-3 class models take millions of GPU-hours. A 7B from scratch is ~$200K on consumer cloud GPUs. A 70B is ~$10M+. This is not a one-person workflow.</li>
              <li><b>Data</b> — you need 1–15T tokens of curated, deduplicated, quality-filtered corpus. Most enterprises don't have anywhere near that volume of domain text.</li>
              <li><b>Infrastructure</b> — distributed training with FSDP / DeepSpeed / Megatron-LM across dozens of nodes with high-bandwidth interconnects. Not a <Code>pip install</Code> situation.</li>
              <li><b>Diminishing returns</b> — post-training a good open-source base model almost always beats pre-training a worse one from scratch on the same budget.</li>
            </Ul>
            <H>The practical middle ground: Continued Pretraining (DAPT)</H>
            <P>
              If you have raw domain text (technical manuals, historical reports, code corpus)
              and want the model to absorb domain vocabulary and style <b>before</b> SFT, the
              right move is <b>Continued Pretraining</b> (a.k.a. Domain-Adaptive Pretraining):
            </P>
            <Ul>
              <li>Start from an existing base (e.g. <Code>Llama-3.1-8B</Code>, not Instruct).</li>
              <li>Continue the next-token prediction objective on your raw text — no instruction pairs, no chat template.</li>
              <li>Train for a few epochs on ~10M–1B tokens of domain text (orders of magnitude less than true pre-training).</li>
              <li>Then run SFT on instruction pairs on top.</li>
            </Ul>
            <Callout tone="info">
              Continued Pretraining is not yet a first-class method in this Studio — it would
              be a sixth trainer alongside SFT / DPO / ORPO / KTO / GRPO. The infrastructure
              (Unsloth + TRL <Code>SFTTrainer</Code> with <Code>packing=True</Code> and no chat
              template) already supports it; it's a matter of adding the trainer class and CLI
              method. Let us know if this is a priority for your use case.
            </Callout>
          </>
        ),
      },
    ],
  },

  {
    id: "inference",
    title: "Inference & Chat",
    articles: [
      {
        id: "chat-widget",
        title: "Using the chat widget",
        summary: "The floating assistant in the bottom-right.",
        body: (
          <>
            <P>The chat widget is always available in the bottom-right corner, on every tab. Click the blue bubble to open it.</P>
            <H>Features</H>
            <Ul>
              <li><b>Domain selector</b> — pick any trained adapter, or <Code>base</Code> for the raw model</li>
              <li><b>Temperature</b> — 0 for deterministic, up to 2 for creative</li>
              <li><b>Max tokens</b> — cap on response length</li>
              <li><b>Per-message telemetry</b> — backend, TTFT ms, tokens/sec shown under each response</li>
              <li><b>Keyboard</b> — Enter to send, Shift+Enter for newline</li>
              <li><b>Unread badge</b> — increments when responses arrive with the widget closed</li>
            </Ul>
          </>
        ),
      },
      {
        id: "backends",
        title: "Hardware backends",
        summary: "vLLM / SGLang / MLX / llama.cpp / HF / Ollama.",
        body: (
          <>
            <P>The inference manager auto-picks the best backend for your hardware:</P>
            <Ul>
              <li><b>vLLM</b> — CUDA Ampere+. PagedAttention, prefix caching, FP8 KV, hot-swap LoRA.</li>
              <li><b>SGLang</b> — CUDA, RadixAttention, best on MoE models.</li>
              <li><b>MLX-LM</b> — Apple Silicon native.</li>
              <li><b>llama.cpp</b> — GGUF, CPU / Metal / CUDA, works everywhere.</li>
              <li><b>HF Transformers</b> — universal fallback.</li>
              <li><b>Ollama</b> — opt-in via <Code>VALONY_INFERENCE_BACKEND=ollama</Code>. Local daemon or Ollama Cloud.</li>
            </Ul>
            <P>Override with <Code>VALONY_INFERENCE_BACKEND=&lt;name&gt;</Code> in <Code>.env</Code>.</P>
          </>
        ),
      },
    ],
  },

  {
    id: "troubleshooting",
    title: "Troubleshooting",
    articles: [
      {
        id: "api-unreachable",
        title: "\"API unreachable\" on the Health tab",
        summary: "Most common cause: missing Python deps.",
        body: (
          <>
            <P>If the Health tab shows a red "API unreachable" banner, the FastAPI backend returned a 500 (or isn't running).</P>
            <H>Steps</H>
            <Ul>
              <li>Confirm uvicorn is running: <Code>curl http://localhost:8000/healthz</Code></li>
              <li>Check the uvicorn terminal for a Python traceback — that tells you which dep is missing.</li>
              <li>Install the full deps for your platform: <Code>pip install -r requirements-cpu.txt</Code> (or <Code>-mlx</Code>, or <Code>-cuda</Code>).</li>
              <li>Restart uvicorn and refresh the browser.</li>
            </Ul>
            <Callout tone="warn">
              Since v3.0.1, <Code>/healthz</Code> degrades gracefully — missing optional deps
              show as <Code>{`{"error": "..."}`}</Code> fields in the response instead of
              returning 500. If you still see the red banner, the backend isn't running at all.
            </Callout>
          </>
        ),
      },
      {
        id: "oom",
        title: "Out of memory during training",
        summary: "VRAM / unified memory tuning.",
        body: (
          <>
            <P>The profile resolver auto-picks LoRA rank, sequence length, and batch size for your hardware, but if you're getting OOM you can override in your domain config:</P>
            <Pre>{`training_args:
  lora_r: 8           # smaller (was 16)
  max_seq_length: 1024 # smaller (was 2048)
  batch_size: 1        # always safest
  gradient_accumulation_steps: 16  # compensates for small batch`}</Pre>
            <P>On Apple Silicon, close other GPU-heavy apps (Xcode, Lightroom, Chrome video) before training — unified memory is shared.</P>
          </>
        ),
      },
    ],
  },
];
