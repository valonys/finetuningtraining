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


# ── Data Forge ─────────────────────────────────────────────────
class ForgeIngestRequest(BaseModel):
    paths: List[str]
    ocr_engine: Optional[str] = Field(
        default=None,
        description="Override OCR engine: rapidocr / paddleocr / docling / tesseract / trocr",
    )


class ForgeBuildRequest(BaseModel):
    paths: List[str]
    task: Literal["sft", "dpo", "orpo", "kto", "grpo"] = "sft"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: str = "You are a helpful assistant."
    synth_qa: bool = True
    synth_mode: Literal["auto", "rule_based", "llm"] = "auto"
    target_size: Optional[int] = None
    output_dir: str = "./data/processed"


class ForgeBuildResponse(BaseModel):
    output_path: str
    task: str
    template: str
    num_examples: int
    sources: List[str]


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
