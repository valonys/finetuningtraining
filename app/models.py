"""
app/models.py
─────────────
Pydantic schemas for ValonyLabs Studio v3.0 API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Training ───────────────────────────────────────────────────
class HFDatasetConfig(BaseModel):
    repo_id: str
    token: Optional[str] = None
    split: str = "train"
    subset: Optional[str] = None
    input_column: str = "input"
    output_column: str = "output"
    max_samples: Optional[int] = None


class TrainingJobRequest(BaseModel):
    domain_config_name: str = Field(
        ...,
        description="Name of a domain config you've created under configs/domains/ "
                    "(e.g., 'asset_integrity', 'customer_grasps', 'ai_llm'). "
                    "Create one first via POST /v1/domains/configs, the CLI, "
                    "or the Studio UI.",
    )
    base_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="HF model id — template is auto-resolved from this",
    )
    training_method: Literal["sft", "dpo", "orpo", "kto", "grpo"] = "sft"
    dataset_path: Optional[str] = None
    hf_dataset: Optional[HFDatasetConfig] = None


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | training | completed | failed
    progress: float = 0.0
    method: Optional[str] = None
    backend: Optional[str] = None
    template: Optional[str] = None
    hardware: Optional[str] = None
    current_loss: Optional[float] = None
    final_loss: Optional[float] = None
    dataset_source: Optional[str] = None
    samples_loaded: Optional[int] = None
    error_message: Optional[str] = None
    adapter_path: Optional[str] = None
    # Per-step metrics captured by LossHistoryCallback. Each entry:
    # {"step": int, "loss": float|None, "learning_rate": float|None,
    #  "grad_norm": float|None, "epoch": float|None, "ts": float}.
    # The UI polls /v1/jobs/{id} and renders a live loss curve from this.
    loss_history: List[Dict[str, Any]] = Field(default_factory=list)


# ── Data Forge ─────────────────────────────────────────────────
class ForgeIngestRequest(BaseModel):
    paths: List[str]
    ocr_engine: Optional[str] = Field(
        default=None,
        description="Override OCR engine: rapidocr / paddleocr / docling / tesseract / trocr",
    )


class UploadedFileInfo(BaseModel):
    name: str
    path: str
    size: int


class UploadListResponse(BaseModel):
    files: List[UploadedFileInfo]
    total_bytes: int


class UploadResponse(BaseModel):
    uploaded: List[UploadedFileInfo]
    skipped: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Files rejected (e.g. too large, empty). Each entry has "
                    "`name` and `reason`.",
    )


# ── YouTube harvester ─────────────────────────────────────────
class YouTubeHarvestRequest(BaseModel):
    query: str = Field(..., description="Keyword search, e.g. 'asset integrity inspection'")
    max_videos: int = Field(default=10, ge=1, le=25)
    min_chars: int = Field(
        default=400,
        description="Skip videos whose transcript is shorter than this.",
    )
    output_dir: str = "./data/uploads"


class YouTubeHarvestedFile(BaseModel):
    title: str
    url: str
    channel: str
    language: str
    auto_generated: bool
    char_count: int
    duration_s: int
    file_path: str


class YouTubeHarvestResponse(BaseModel):
    query: str
    max_requested: int
    harvested: List[YouTubeHarvestedFile]
    skipped: List[Dict[str, str]] = Field(default_factory=list)


# ── arXiv harvester ──────────────────────────────────────────
class ArxivHarvestRequest(BaseModel):
    query: str = Field(..., description="Keyword search, e.g. 'GRPO reinforcement learning'")
    max_papers: int = Field(default=20, ge=1, le=100)
    mode: Literal["abstract", "full"] = Field(
        default="abstract",
        description="'abstract' writes .txt files; 'full' downloads PDFs.",
    )
    min_chars: int = Field(default=200, description="Skip papers with shorter abstracts.")
    output_dir: str = "./data/uploads"


class ArxivHarvestedPaperInfo(BaseModel):
    arxiv_id: str
    title: str
    authors: str
    categories: str
    published: str
    char_count: int
    file_path: str
    mode: str


class ArxivHarvestResponse(BaseModel):
    query: str
    max_requested: int
    harvested: List[ArxivHarvestedPaperInfo]
    skipped: List[Dict[str, str]] = Field(default_factory=list)


# ── Code harvester ───────────────────────────────────────────
class CodeHarvestRequest(BaseModel):
    path: str = Field(..., description="Directory to scan for .py and .ipynb files")
    strategy: Literal["implement", "explain", "review", "docstring", "all"] = "implement"
    source_label: str = Field(default="", description="Label for provenance, e.g. 'Manning/Reasoning'")
    min_lines: int = Field(default=5, ge=1)
    max_lines: int = Field(default=200, ge=10)
    output_dir: str = "./data/uploads"


class CodeHarvestResponse(BaseModel):
    files_scanned: int
    files_skipped: int
    total_units: int
    output_path: str


class ForgeBuildRequest(BaseModel):
    paths: List[str]
    task: Literal["sft", "dpo", "orpo", "kto", "grpo"] = "sft"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    template: Optional[Literal[
        "auto", "alpaca", "chatml", "deepseek", "gemma",
        "llama2", "llama3", "mistral", "phi", "qwen", "sharegpt",
    ]] = Field(
        default="auto",
        description="Chat template to apply to each row. 'auto' resolves "
                    "from base_model. Pick an explicit name to override "
                    "(e.g. format Llama data in Alpaca style).",
    )
    system_prompt: str = "You are a helpful assistant."
    synth_qa: bool = True
    synth_mode: Literal["auto", "rule_based", "llm"] = "auto"
    target_size: Optional[int] = None
    output_dir: str = "./data/processed"
    filter_noise: bool = Field(
        default=True,
        description="Drop TOC / cover / bibliography / digit-dense fragments "
                    "before Q/A synthesis. Set False for full-text cases "
                    "where every chunk is legitimately short/structured.",
    )


class ForgeBuildResponse(BaseModel):
    output_path: str = Field(
        description="Path to the primary JSONL dataset (one row per line). "
                    "Loadable via `load_dataset('json', data_files=...)`."
    )
    preview_path: str = Field(
        description="Path to a pretty-printed JSON file containing the first "
                    "~10 rows of the dataset — for peeking in any editor."
    )
    task: str
    template: str
    num_examples: int
    sources: List[str]


# ── Multimodal pipeline ───────────────────────────────────────
class MultimodalIndexRequest(BaseModel):
    paths: List[str] = Field(
        default_factory=list,
        description="Server-side paths to ingest. Must resolve under uploads, processed, or outputs.",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Optional tenant override. Defaults to JWT/dev tenant.",
    )
    collection: str = Field(default="default", min_length=1, max_length=128)
    source_type: Optional[Literal[
        "text", "audio", "image", "slide", "video", "document", "code",
    ]] = Field(
        default=None,
        description="Force all indexed records to this modality. Omit for Data Forge auto-detection.",
    )
    ocr_engine: Optional[str] = None
    chunk_target_chars: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap_chars: int = Field(default=160, ge=0, le=2000)
    embedding_dim: int = Field(default=384, ge=16, le=4096)
    embed_provider: Literal["hash", "openai_compat"] = "hash"


class MultimodalIndexResponse(BaseModel):
    tenant_id: str
    collection: str
    records_indexed: int
    chunks_indexed: int
    stats: Dict[str, Any]


class MultimodalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tenant_id: Optional[str] = None
    collection: str = Field(default="default", min_length=1, max_length=128)
    top_k: int = Field(default=8, ge=1, le=50)
    source_type: Optional[Literal[
        "text", "audio", "image", "slide", "video", "document", "code",
    ]] = None
    embedding_dim: int = Field(default=384, ge=16, le=4096)
    embed_provider: Literal["hash", "openai_compat"] = "hash"


class MultimodalSearchResult(BaseModel):
    chunk_id: str
    record_id: str
    text: str
    score: float
    source_type: str
    source_uri: str
    title: Optional[str] = None
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultimodalSearchResponse(BaseModel):
    tenant_id: str
    collection: str
    query: str
    results: List[MultimodalSearchResult]


class MultimodalRAGRequest(MultimodalSearchRequest):
    generate: bool = Field(
        default=False,
        description="If true, call the configured inference backend. If false, return cited context only.",
    )
    domain_config_name: str = "base"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=800, ge=1, le=4096)
    max_context_chars: int = Field(default=12000, ge=1000, le=60000)


class MultimodalRAGResponse(BaseModel):
    tenant_id: str
    collection: str
    query: str
    answer: str
    sources: List[str]
    context: str
    results: List[MultimodalSearchResult]


class MultimodalStatsResponse(BaseModel):
    tenant_id: str
    collection: str
    chunk_count: int
    by_modality: Dict[str, int]


# ── Domain configs (user-defined per engagement) ───────────────
class DomainConfigCreateRequest(BaseModel):
    """Create a new domain config YAML under configs/domains/<name>.yaml."""
    name: str = Field(
        ...,
        description="Domain identifier. Must match ^[a-z][a-z0-9_]{0,63}$ — "
                    "e.g., 'asset_integrity', 'customer_grasps', 'ai_llm', "
                    "'legal_nda_review'.",
        examples=["asset_integrity", "customer_grasps", "ai_llm"],
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Persona/role the model adopts. Required unless copy_from is set.",
    )
    constitution: Optional[List[str]] = Field(
        default=None,
        description="Domain-specific guardrail rules (injected after system prompt).",
    )
    training_args: Optional[Dict[str, Any]] = None
    dpo_args: Optional[Dict[str, Any]] = None
    orpo_args: Optional[Dict[str, Any]] = None
    kto_args: Optional[Dict[str, Any]] = None
    grpo_args: Optional[Dict[str, Any]] = None
    copy_from: Optional[str] = Field(
        None,
        description="Name of an example in configs/domains/examples/ to seed from. "
                    "Any fields you supply above override the example's values.",
    )
    overwrite: bool = False


class DomainConfigInfo(BaseModel):
    name: str
    path: str
    config: Dict[str, Any]


class DomainConfigListResponse(BaseModel):
    configs: List[str] = Field(
        description="User-created domain configs under configs/domains/"
    )
    examples: List[str] = Field(
        description="Seed examples under configs/domains/examples/ "
                    "(copy with `copy_from` at create time)"
    )


# ── Inference ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    domain_config_name: Optional[str] = Field(
        default="base",
        description="Which trained adapter to use, or 'base' for the raw base model.",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=512, ge=1, le=8192)
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    domain: str
    model: str
    backend: str
    tokens_generated: int = 0
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0


class DomainInfo(BaseModel):
    """Info about a trained domain adapter on disk (not the YAML config)."""
    domain_name: str
    adapter_path: str
    method: Optional[str] = None


# ── Model registry (A3) ───────────────────────────────────────
class RegistryPromoteRequest(BaseModel):
    """POST /v1/registry/promote — move a model_version to a new status."""
    model_version: str
    to_status: str = Field(
        ...,
        description="Target status: 'staging' | 'production' | 'rolled_back'",
    )
    actor: Optional[str] = None
    reason: Optional[str] = None


class RegistryRollbackRequest(BaseModel):
    """POST /v1/registry/rollback — demote the current production for a domain."""
    domain: str
    target_version: Optional[str] = Field(
        default=None,
        description="Optional version to promote as the replacement. If omitted, "
                    "the registry auto-picks the most recently updated "
                    "STAGING / ROLLED_BACK version for the domain.",
    )
    actor: Optional[str] = None
    reason: Optional[str] = None


# ── Health ─────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    hardware: dict
    profile: dict
    registered_domains: List[str]
    inference_backend: str
    latency_stats: dict
    available_ocr: List[str]
    available_templates: List[str]
    synth_provider: Dict[str, Any] = Field(
        default_factory=dict,
        description="Active LLM provider used by the Data Forge for Q/A and "
                    "pair synthesis (Ollama Cloud Nemotron by default when "
                    "OLLAMA_API_KEY is set).",
    )
