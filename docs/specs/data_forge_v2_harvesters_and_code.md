# Spec: Data Forge v2 -- arXiv harvester, code harvester, enhanced noise filter

## Goal

Expand the Data Forge's data collection and processing capabilities so a CTO-grade SFT dataset can be built from three new source types (arXiv papers, Python code, and tech books) alongside the existing YouTube transcripts and document uploads. The resulting dataset should give the fine-tuned model substantive knowledge of:

- LLM post-training (SFT, DPO, GRPO, distillation)
- Model evaluation and benchmarking
- RAG architectures and vector databases
- Agentic systems (tool use, multi-agent orchestration)
- Production inference (vLLM, quantisation, scaling)
- Python implementation patterns for all of the above

## Non-goals

- We are NOT building a general-purpose code model (Codex, StarCoder). We want code *understanding* for a domain expert, not code *completion* for an IDE.
- We are NOT training on pirated material. All content is legally owned via Manning subscriptions, free PDFs, and open arXiv papers.
- We are NOT replacing the existing YouTube/upload harvesters. We're adding alongside them.

---

## Source inventory (what we have)

### Manning (~/Documents/Manning/)

| Source | Files | Content | SFT value |
|--------|-------|---------|-----------|
| reasoning-from-scratch .py files | 81 | GRPO training loops, Qwen3 inference, math verification, distillation, batched generation, LLM-as-judge | **Very high** -- exactly what the CTO model needs |
| reasoning-from-scratch .ipynb | 19 | Chapter walkthroughs with explanation + code | **Very high** -- natural instruction/response pairs |
| ai-evaluations .ipynb | 5 | Offline eval, diagnostics, counterfactual, drift monitoring | **High** -- evaluation knowledge |
| deep-learning-in-motion .ipynb | 18 | Neural net fundamentals, backprop, CNNs | **Medium** -- foundational but basic |
| PDFs (2 MEAP books) | 2 | AI Model Evaluation (7.2M), Build a Reasoning Model (29M) | **Very high** -- deep explanatory text |

### Weaviate (~/Documents/Weaviate/)

| Source | Files | Content | SFT value |
|--------|-------|---------|-----------|
| Weaviate PDFs | 4 | RAG techniques, agentic architectures, AI strategy | **High** -- CTO-level architecture knowledge |

### Other

| Source | Files | Content | SFT value |
|--------|-------|---------|-----------|
| AI Agents Handbook | 1 PDF | Agent design patterns | **High** |
| CrewAI course | 1 PDF + 2 .py + 1 .ipynb | Multi-agent orchestration | **Medium-high** |

### arXiv (to be harvested)

| Query | Expected papers | SFT value |
|-------|----------------|-----------|
| "GRPO reinforcement learning" | 10-20 | **Very high** -- cutting edge |
| "LoRA fine-tuning efficient" | 20-30 | **High** |
| "RAG retrieval augmented generation" | 20-30 | **High** |
| "LLM evaluation benchmark" | 15-25 | **High** |
| "agentic LLM tool use" | 10-20 | **High** |
| "knowledge distillation language model" | 15-25 | **High** |

---

## Three new harvesters

### A. arXiv Harvester (`app/harvesters/arxiv.py`)

**How it works:**

```
Keyword query
    |
    v
arXiv API (http://export.arxiv.org/api/query)
    |
    v
Parse XML -> list of {title, authors, abstract, pdf_url, arxiv_id, categories}
    |
    +---> Mode 1: Abstract-only (fast, lightweight)
    |     -> Write abstract as .txt to data/uploads/
    |     -> Q/A synthesis generates "explain this paper" pairs
    |
    +---> Mode 2: Full paper (deep, heavier)
          -> Download PDF
          -> Route through existing Data Forge pipeline (pymupdf -> chunk -> filter -> synth)
          -> Enhanced noise filter strips references, acknowledgements, figure captions
```

**API surface:**

```python
# app/harvesters/arxiv.py
class ArxivHarvester:
    def search(self, query: str, max_results: int = 20,
               categories: list[str] | None = None) -> list[ArxivSearchResult]
    def fetch_abstract(self, arxiv_id: str) -> str
    def fetch_pdf(self, arxiv_id: str, output_dir: str) -> str  # returns file path

# FastAPI endpoint
POST /v1/forge/harvest/arxiv
Body: {"query": "GRPO reinforcement learning", "max_papers": 20, "mode": "abstract"}
Response: ArxivHarvestResponse (mirrors YouTubeHarvestResponse shape)
```

**Dependencies:** `arxiv` Python package (lightweight, well-maintained) or direct HTTP to the API (no extra dep).

**Effort:** 1 day.

---

### B. Code Harvester (`app/harvesters/code.py`)

**How it works:**

```
Directory path (e.g. ~/Documents/Manning/reasoning-from-scratch-main/)
    |
    v
Recursively find .py and .ipynb files
    |
    v
Extract units:
    .py  -> functions, classes, and module-level code blocks (via AST parsing)
    .ipynb -> code cells paired with their preceding markdown cell (natural instruction/response)
    |
    v
For each unit, generate SFT pairs using one of these strategies:
    |
    +---> "explain": "Explain what this code does." -> code + LLM explanation
    +---> "implement": "How would you implement X?" -> explanation as instruction, code as response
    +---> "review": "Review this code for bugs or improvements." -> code + LLM review
    +---> "docstring": "Add a docstring to this function." -> code + LLM docstring
    |
    v
Template-format into the target chat template (Qwen/Llama/Alpaca/...)
    |
    v
Output: .jsonl ready for SFT training
```

**Key design decisions:**

1. **AST-based extraction for .py files.** We parse with Python's `ast` module to extract functions and classes cleanly, preserving decorator/type annotations and docstrings. Raw text splitting would lose function boundaries.

2. **Markdown+code cell pairing for .ipynb files.** Notebooks are naturally structured as "explanation cell followed by code cell" -- this maps directly to instruction/response pairs without any LLM synthesis needed. The markdown IS the instruction; the code IS the response.

3. **Context window awareness.** Large files (e.g. `qwen3_optimized.py` at 25K) get chunked into individual functions. Small files get treated as a single unit.

4. **Source attribution.** Every generated SFT row includes a `source` field so the CTO knows which book/chapter a data point came from. This also enables filtering (e.g. "only train on reasoning-model code, not deep-learning-basics").

**API surface:**

```python
# app/harvesters/code.py
class CodeHarvester:
    def harvest_directory(
        self, path: str, *,
        extensions: list[str] = [".py", ".ipynb"],
        strategy: str = "implement",  # "explain" | "implement" | "review" | "docstring" | "all"
        min_lines: int = 5,           # skip trivial snippets
        max_lines: int = 200,         # chunk large files
        source_label: str = "",       # e.g. "Manning/reasoning-from-scratch"
    ) -> CodeHarvestReport

# FastAPI endpoint
POST /v1/forge/harvest/code
Body: {"path": "/path/to/repo", "strategy": "implement", "source_label": "Manning/Build-Reasoning-Model"}
```

**Effort:** 2-3 days (AST parsing + notebook extraction + 4 synthesis strategies).

---

### C. Enhanced Noise Filter (`app/data_forge/chunk_filter.py` update)

**Current state:** 9 rejection rules (too_short, digit_dense, all_caps, toc_like, front_matter, short_sentences, etc.)

**New rules to add:**

| Rule | Detects | How |
|------|---------|-----|
| `bibliography` | Reference lists at the end of papers | Lines matching `^\[\d+\]` or `^[A-Z][a-z]+,\s[A-Z]\.` (author patterns) in clusters of 5+ consecutive lines |
| `acknowledgements` | "This work was supported by..." sections | Header detection (`Acknowledg(e?)ments`) + short section heuristic |
| `figure_caption` | "Figure 3: Architecture of..." | Lines starting with `Figure \d+` or `Table \d+` that are under 200 chars |
| `running_header` | Repeated text at top/bottom of each page | Detect strings that appear identically on 3+ consecutive page boundaries |
| `index_page` | Alphabetical lists with page numbers | Lines matching `^[A-Z][a-z]+.*\d+$` in clusters of 10+ |
| `citation_cluster` | Paragraphs that are mostly [1,2,3] references | Ratio of `\[\d+\]` tokens to total tokens > 0.3 |
| `latex_artifacts` | Residual LaTeX commands from PDF extraction | Lines with high density of `\`, `{`, `}`, `\begin`, `\end` |

**Effort:** 1 day.

---

## CTO Dataset Composition Strategy

The goal is a balanced dataset that gives the model **deep AI/ML engineering knowledge** without losing general instruction-following capability.

### Recommended mix (target: 10,000-15,000 rows)

| Source | Rows | Percentage | What it teaches |
|--------|------|------------|----------------|
| **Manning code (reasoning-from-scratch)** | 2,000-3,000 | 20-25% | GRPO, distillation, inference optimization, Qwen internals |
| **Manning code (ai-evaluations + deep-learning)** | 500-800 | 5-7% | Evaluation methods, neural net fundamentals |
| **Manning PDFs (chapter text)** | 1,500-2,000 | 15% | Deep explanatory content, reasoning model theory |
| **arXiv papers (abstracts + key sections)** | 1,500-2,000 | 15% | Cutting-edge research, formal definitions, method comparisons |
| **Weaviate + AI Agents PDFs** | 800-1,200 | 8-10% | RAG architecture, agentic design, production strategy |
| **YouTube transcripts (existing harvester)** | 1,500-2,000 | 15% | Practitioner knowledge, tutorials, conference talks |
| **General capability (Alpaca subset)** | 1,500-2,000 | 15% | Instruction-following, formatting, general knowledge |

### Dataset quality tiers

```
Tier 1 (highest quality): Code with AST-extracted functions + notebook markdown/code pairs
    -> Natural instruction/response structure, no LLM synthesis needed
    -> ~3,000 rows from Manning repos

Tier 2 (high quality): LLM-synthesised Q/A from PDF chapters and paper abstracts
    -> Strong source material, well-structured questions
    -> ~4,000 rows from PDFs + arXiv

Tier 3 (good quality): YouTube transcript Q/A synthesis
    -> Variable quality, but covers practitioner perspectives
    -> ~2,000 rows

Tier 4 (baseline): Alpaca/general instruction subset
    -> Prevents capability regression
    -> ~2,000 rows
```

---

## Implementation roadmap

### Sprint 1: Enhanced noise filter (1 day)

Update `app/data_forge/chunk_filter.py` with 7 new rules. Test against the Manning PDFs to verify bibliography/TOC/index stripping works. All 186+ existing tests still pass.

**Deliverable:** Better chunks from PDF ingestion.

### Sprint 2: arXiv harvester (1-2 days)

Build `app/harvesters/arxiv.py`. Wire `/v1/forge/harvest/arxiv` endpoint. Add to DataForge.tsx UI (mirrors YouTube card). Test with "GRPO reinforcement learning" query.

**Deliverable:** arXiv papers flowing into Data Forge.

### Sprint 3: Code harvester (2-3 days)

Build `app/harvesters/code.py` with AST parser + notebook extractor. Wire `/v1/forge/harvest/code` endpoint. Test against reasoning-from-scratch repo.

**Deliverable:** Python code + notebooks → SFT-ready JSONL.

### Sprint 4: CTO dataset build (1 day)

Orchestrate all harvesters:
1. Harvest arXiv (5-6 keyword queries, ~100 papers)
2. Harvest Manning code repos (3 repos, ~126 files)
3. Ingest Manning + Weaviate PDFs through Data Forge
4. Harvest YouTube (10+ keyword queries)
5. Mix with Alpaca subset
6. Build final JSONL with chat template formatting

**Deliverable:** `data/processed/cto_knowledge_sft.jsonl` (~10k-15k rows)

### Sprint 5: Train + eval (1-2 days)

SFT on Qwen 0.5B (smoke test) then Qwen 7B (real run on Colab/RTX). 3 epochs. Eval against a 50-question CTO knowledge bank. LLM-as-judge vs base model.

**Deliverable:** `amiguel/qwen-7b-cto-knowledge-sft` on HuggingFace.

### Total estimated effort: 6-10 days

---

## Code harvester: detailed design for notebook extraction

Since Manning notebooks are the highest-quality source (natural instruction/response pairs), here's the extraction logic:

```python
def extract_notebook_pairs(ipynb_path: str) -> list[dict]:
    """Extract (markdown_instruction, code_response) pairs from a .ipynb.

    Strategy: walk cells. When we see a markdown cell followed by a
    code cell, treat the markdown as the instruction and the code as
    the response. Skip:
      - Code cells with no preceding markdown (orphaned)
      - Markdown cells that are just headers (< 20 chars)
      - Code cells that only import or configure (< 5 lines)
      - Code cells with no output (likely setup/config)
    """
```

**Example from reasoning-from-scratch ch06_main.ipynb:**

```
Markdown cell: "## 6.2 GRPO Training Loop
                 The GRPO algorithm samples multiple completions per prompt,
                 scores them with a reward function, and updates the policy
                 using a clipped surrogate objective..."

Code cell:     def grpo_step(model, prompts, reward_fn, ...):
                   completions = [model.generate(p) for p in prompts]
                   rewards = [reward_fn(c) for c in completions]
                   ...
```

This becomes:
```json
{
  "instruction": "Explain the GRPO training loop and show a Python implementation. The GRPO algorithm samples multiple completions per prompt, scores them with a reward function, and updates the policy using a clipped surrogate objective.",
  "response": "```python\ndef grpo_step(model, prompts, reward_fn, ...):\n    completions = [model.generate(p) for p in prompts]\n    rewards = [reward_fn(c) for c in completions]\n    ...\n```",
  "source": "Manning/Build-a-Reasoning-Model/ch06",
  "type": "code_implementation"
}
```

Zero LLM synthesis needed — the book's own markdown is the instruction.

---

## Code harvester: detailed design for .py file extraction

```python
def extract_python_functions(py_path: str) -> list[dict]:
    """Extract function/class definitions with their docstrings.

    Uses Python's ast module to parse the file and extract:
      - Function name, signature, decorators
      - Docstring (if present)
      - Full function body
      - Module-level context (imports, constants)

    For each function, generate an SFT pair based on the chosen strategy.
    """
```

**Example from reasoning-from-scratch ch06.py:**

```python
def compute_group_rewards(completions, ground_truths, reward_fn):
    """Score each completion against its ground truth.
    Returns a tensor of shape (batch, num_generations).
    """
    rewards = []
    for comp_group, gt in zip(completions, ground_truths):
        group_rewards = [reward_fn(c, gt) for c in comp_group]
        rewards.append(group_rewards)
    return torch.tensor(rewards)
```

**Strategy "implement":**
```json
{
  "instruction": "Implement a function called `compute_group_rewards` that scores each completion against its ground truth using a reward function. It should accept completions (grouped by prompt), ground_truths, and a reward_fn. Return a tensor of shape (batch, num_generations).",
  "response": "```python\ndef compute_group_rewards(completions, ground_truths, reward_fn):\n    ...\n```"
}
```

**Strategy "explain":**
```json
{
  "instruction": "Explain what the following Python function does:\n```python\ndef compute_group_rewards(completions, ground_truths, reward_fn):\n    ...\n```",
  "response": "This function scores each completion against its ground truth. It takes three arguments: completions (a list of groups where each group contains multiple generated completions for the same prompt), ground_truths (the correct answers), and reward_fn (a callable that rates a single completion against a ground truth). It iterates over each group, applies the reward function to every completion in that group, and returns the results as a PyTorch tensor with shape (batch_size, num_generations). This is used in the GRPO training loop to compute the advantage of each sampled completion."
}
```

---

## Acceptance criteria

- [ ] arXiv harvester returns papers for "GRPO reinforcement learning" with abstracts
- [ ] Code harvester extracts 50+ functions from reasoning-from-scratch .py files
- [ ] Code harvester extracts 100+ instruction/response pairs from .ipynb notebooks
- [ ] Enhanced noise filter rejects bibliography/TOC/index chunks from the Manning PDFs
- [ ] All generated JSONL rows are valid and loadable by `load_dataset("json", ...)`
- [ ] All existing 186+ tests still pass
- [ ] New tests cover each harvester and each new filter rule
- [ ] Dataset can be built end-to-end with a single script or notebook

## Test plan

- **Unit:** Mock arxiv API responses, mock file system for code harvester, test each new filter rule against handcrafted examples
- **Integration:** Run the full pipeline on the Manning repo, verify JSONL output loads and has the expected row count + column schema
- **Quality:** Manually inspect 20 random rows from each source to verify instruction/response quality

## Dependencies

- Enhanced noise filter: none (updates existing module)
- arXiv harvester: `arxiv` Python package (or zero-dep HTTP approach)
- Code harvester: Python `ast` module (stdlib, zero deps)
- CTO dataset build: all three above + existing Data Forge

---

*This spec follows the template from docs/specs/_TEMPLATE.md. Implementation begins after user review and approval.*
