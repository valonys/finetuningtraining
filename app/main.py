"""
app/main.py
───────────
FastAPI entrypoint for ValonyLabs Studio v3.0.

Endpoints
─────────
  GET  /healthz                          Hardware + backend + telemetry
  GET  /v1/templates                     List registered chat templates
  GET  /v1/ocr/engines                   List available OCR engines

  POST /v1/forge/ingest                  Ingest files → list of records
  POST /v1/forge/build_dataset           Ingest → normalise → HF dataset

  GET  /v1/domains/configs               List user-created domain configs + seed examples
  GET  /v1/domains/configs/{name}        Read one domain config (dict)
  POST /v1/domains/configs               Create a new domain config YAML
  GET  /v1/domains/template              Get the blueprint template (for UI forms)

  POST /v1/jobs/create                   Start SFT/DPO/ORPO/KTO/GRPO training
  GET  /v1/jobs/{job_id}                 Poll job status
  GET  /v1/jobs                          List all jobs

  GET  /v1/domains                       List trained adapters on disk
  POST /v1/inference/reload              Re-scan outputs/ for new adapters

  POST /v1/chat                          Inference against a trained adapter (or 'base')
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config_loader import (
    DomainConfigError,
    create_domain_config,
    get_domain_template,
    list_domain_configs,
    list_domain_examples,
    load_domain_config,
)
from app.data_forge import DataForge
from app.data_forge.ocr.pipeline import list_available_engines
from app.hardware import detect_hardware, resolve_profile
from app.inference.manager import (
    GenerationRequest,
    get_inference_engine,
    reset_engine,
)
from app.providers import describe_active_provider
from app.models import (
    ChatRequest,
    ChatResponse,
    DomainConfigCreateRequest,
    DomainConfigInfo,
    DomainConfigListResponse,
    DomainInfo,
    ForgeBuildRequest,
    ForgeBuildResponse,
    ForgeIngestRequest,
    HealthResponse,
    JobStatus,
    TrainingJobRequest,
)
from app.templates import list_templates
from app.trainers import (
    AgnosticDPOTrainer,
    AgnosticGRPOTrainer,
    AgnosticKTOTrainer,
    AgnosticORPOTrainer,
    AgnosticSFTTrainer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

job_registry: Dict[str, JobStatus] = {}

DEFAULT_BASE_MODEL = os.environ.get("VALONY_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# Opt-in override — set VALONY_INFERENCE_BACKEND=ollama to route generation
# through a local Ollama daemon or Ollama Cloud. Leave unset to let the
# hardware profile pick (vLLM on CUDA, MLX on Apple, etc.).
DEFAULT_INFERENCE_BACKEND = os.environ.get("VALONY_INFERENCE_BACKEND") or None


# ──────────────────────────────────────────────────────────────
# Lifespan — pre-warm the inference engine on startup
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 ValonyLabs Studio v3.0 starting")
    hw = detect_hardware()
    prof = resolve_profile(hw)
    logger.info(f"   Hardware: {hw.tier} | {hw.device_name} | mem={hw.effective_memory_gb} GB")
    logger.info(f"   Training backend: {prof.training_backend}")
    logger.info(f"   Inference backend: {DEFAULT_INFERENCE_BACKEND or prof.inference_backend}")
    try:
        engine = get_inference_engine(DEFAULT_BASE_MODEL, backend=DEFAULT_INFERENCE_BACKEND)
        logger.info(f"   ✅ Inference engine: {engine.backend}")
    except Exception as e:
        logger.warning(f"   ⚠️  Inference pre-warm skipped: {e}")
    yield
    logger.info("ValonyLabs Studio shutting down")


app = FastAPI(
    title="ValonyLabs Studio",
    version=__version__,
    description="Agnostic post-training & inference platform (Mac / RTX / Colab / Brev)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Health & introspection
# ──────────────────────────────────────────────────────────────
@app.get("/healthz", response_model=HealthResponse)
async def health():
    hw = detect_hardware()
    prof = resolve_profile(hw)
    try:
        engine = get_inference_engine(DEFAULT_BASE_MODEL, backend=DEFAULT_INFERENCE_BACKEND)
        domains = engine.registered_domains
        backend = engine.backend
        stats = engine.latency_stats()
    except Exception:
        domains, backend, stats = [], "not_loaded", {}

    return HealthResponse(
        status="ok",
        version=__version__,
        hardware=hw.as_dict(),
        profile={
            "training_backend": prof.training_backend,
            "inference_backend": prof.inference_backend,
            "torch_dtype": prof.torch_dtype,
            "max_seq_length": prof.max_seq_length,
            "lora_r": prof.lora_r,
        },
        registered_domains=domains,
        inference_backend=backend,
        latency_stats=stats,
        available_ocr=list_available_engines(),
        available_templates=list_templates(),
        synth_provider=describe_active_provider(),
    )


@app.get("/v1/templates")
async def templates():
    return {"templates": list_templates()}


@app.get("/v1/ocr/engines")
async def ocr_engines():
    return {"engines": list_available_engines()}


# ──────────────────────────────────────────────────────────────
# Domain configs (user-defined, one per engagement)
# ──────────────────────────────────────────────────────────────
@app.get("/v1/domains/configs", response_model=DomainConfigListResponse)
async def domain_configs_list():
    """List user-created domain configs and the available seed examples."""
    return DomainConfigListResponse(
        configs=list_domain_configs(),
        examples=list_domain_examples(),
    )


@app.get("/v1/domains/configs/{name}", response_model=DomainConfigInfo)
async def domain_configs_get(name: str):
    """Read one user-created domain config."""
    try:
        cfg = load_domain_config(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return DomainConfigInfo(
        name=name,
        path=f"configs/domains/{name}.yaml",
        config=cfg,
    )


@app.post("/v1/domains/configs", response_model=DomainConfigInfo, status_code=201)
async def domain_configs_create(req: DomainConfigCreateRequest):
    """Create a new domain config at configs/domains/<name>.yaml."""
    if req.system_prompt is None and req.copy_from is None:
        raise HTTPException(
            422,
            "Provide `system_prompt` or `copy_from` so the domain has a persona.",
        )
    try:
        path = create_domain_config(
            name=req.name,
            system_prompt=req.system_prompt,
            constitution=req.constitution,
            training_args=req.training_args,
            dpo_args=req.dpo_args,
            orpo_args=req.orpo_args,
            kto_args=req.kto_args,
            grpo_args=req.grpo_args,
            copy_from=req.copy_from,
            overwrite=req.overwrite,
        )
    except DomainConfigError as e:
        raise HTTPException(409, str(e))

    cfg = load_domain_config(req.name)
    return DomainConfigInfo(name=req.name, path=path, config=cfg)


@app.get("/v1/domains/template")
async def domain_configs_template():
    """Return the blueprint template (dict form) for UI forms and CLIs."""
    try:
        return {"template": get_domain_template()}
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))


# ──────────────────────────────────────────────────────────────
# Data Forge
# ──────────────────────────────────────────────────────────────
@app.post("/v1/forge/ingest")
async def forge_ingest(req: ForgeIngestRequest):
    forge = DataForge(ocr_engine=req.ocr_engine)
    try:
        records = forge.ingest(req.paths)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "num_records": len(records),
        "records": [
            {
                "source": r.source,
                "source_type": r.source_type,
                "preview": r.text[:280],
                "metadata": r.metadata,
                "num_tables": len(r.tables),
            }
            for r in records
        ],
    }


@app.post("/v1/forge/build_dataset", response_model=ForgeBuildResponse)
async def forge_build(req: ForgeBuildRequest):
    forge = DataForge()
    try:
        records = forge.ingest(req.paths)
        ds = forge.build_dataset(
            records,
            task=req.task,
            base_model=req.base_model,
            system_prompt=req.system_prompt,
            synth_qa=req.synth_qa,
            synth_mode=req.synth_mode,
            target_size=req.target_size,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    out_dir = os.path.join(req.output_dir, f"{req.task}_{os.urandom(3).hex()}")
    os.makedirs(req.output_dir, exist_ok=True)
    ds.save_to_disk(out_dir)

    from app.templates import get_template_for
    template = get_template_for(req.base_model)
    return ForgeBuildResponse(
        output_path=out_dir,
        task=req.task,
        template=template.name,
        num_examples=len(ds),
        sources=[r.source for r in records],
    )


# ──────────────────────────────────────────────────────────────
# Training jobs
# ──────────────────────────────────────────────────────────────
_TRAINER_CLASSES = {
    "sft":  AgnosticSFTTrainer,
    "dpo":  AgnosticDPOTrainer,
    "orpo": AgnosticORPOTrainer,
    "kto":  AgnosticKTOTrainer,
    "grpo": AgnosticGRPOTrainer,
}


async def _run_training(job_id: str, req: TrainingJobRequest, config: Dict[str, Any]):
    job = job_registry[job_id]
    job.status = "training"

    def progress(p: float):
        job.progress = p

    try:
        TrainerCls = _TRAINER_CLASSES[req.training_method]
        hf_cfg = req.hf_dataset.model_dump() if req.hf_dataset else None

        trainer = TrainerCls(
            config=config,
            base_model_id=req.base_model,
            dataset_path=req.dataset_path,
            hf_dataset_config=hf_cfg,
            progress_callback=progress,
        )
        result = await asyncio.to_thread(trainer.train)

        # Register the new adapter so it's immediately usable
        engine = get_inference_engine(req.base_model)
        engine.register_adapter(config["domain_name"], result["adapter_path"])

        job.status = "completed"
        job.progress = 1.0
        job.final_loss = result.get("final_loss")
        job.backend = result.get("backend")
        job.template = result.get("template")
        job.hardware = result.get("hardware")
        job.adapter_path = result.get("adapter_path")
        logger.info(f"[{job_id}] ✅ Training complete")

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        logger.exception(f"[{job_id}] ❌ Training failed")


@app.post("/v1/jobs/create", response_model=JobStatus, status_code=202)
async def create_job(req: TrainingJobRequest, background_tasks: BackgroundTasks):
    if not req.dataset_path and not req.hf_dataset:
        raise HTTPException(422, "Provide dataset_path or hf_dataset")
    if req.dataset_path and req.hf_dataset:
        raise HTTPException(422, "Provide only one of dataset_path / hf_dataset")

    try:
        config = load_domain_config(req.domain_config_name)
    except FileNotFoundError as e:
        raise HTTPException(
            404,
            f"{e} Create one first via POST /v1/domains/configs "
            f"or `python scripts/new_domain.py create {req.domain_config_name}`.",
        )

    if req.training_method not in _TRAINER_CLASSES:
        raise HTTPException(422, f"Unsupported training method: {req.training_method}")

    job_id = str(uuid.uuid4())
    job_registry[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        progress=0.0,
        method=req.training_method,
        dataset_source="huggingface" if req.hf_dataset else "local",
    )
    background_tasks.add_task(_run_training, job_id, req, config)
    logger.info(f"Job {job_id} queued | method={req.training_method}")
    return job_registry[job_id]


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    if job_id not in job_registry:
        raise HTTPException(404, "Job not found")
    return job_registry[job_id]


@app.get("/v1/jobs", response_model=List[JobStatus])
async def list_jobs():
    return list(job_registry.values())


# ──────────────────────────────────────────────────────────────
# Domains & inference
# ──────────────────────────────────────────────────────────────
@app.get("/v1/domains", response_model=List[DomainInfo])
async def list_domains():
    engine = get_inference_engine(DEFAULT_BASE_MODEL, backend=DEFAULT_INFERENCE_BACKEND)
    return [
        DomainInfo(domain_name=name, adapter_path=path)
        for name, path in engine.lora_registry.items()
    ]


@app.post("/v1/inference/reload")
async def reload_inference():
    reset_engine()
    engine = get_inference_engine(DEFAULT_BASE_MODEL, backend=DEFAULT_INFERENCE_BACKEND)
    return {"status": "reloaded", "domains": engine.registered_domains}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    engine = get_inference_engine(DEFAULT_BASE_MODEL, backend=DEFAULT_INFERENCE_BACKEND)

    # Build a proper prompt using the right template for the base model
    from app.templates import get_template_for
    template = get_template_for(engine.base_model_id)

    # Try to pick up a domain-specific system prompt
    system_prompt = "You are a helpful assistant."
    if req.domain_config_name and req.domain_config_name != "base":
        try:
            cfg = load_domain_config(req.domain_config_name)
            system_prompt = cfg.get("system_prompt", system_prompt)
        except FileNotFoundError:
            pass

    prompt = template.format_prompt_only(system=system_prompt, instruction=req.message)

    gen_req = GenerationRequest(
        prompt=prompt,
        domain_name=req.domain_config_name or "base",
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
    )
    try:
        resp = await asyncio.to_thread(engine.generate_full, gen_req)
    except ValueError as e:
        raise HTTPException(404, str(e))

    return ChatResponse(
        response=resp.text,
        domain=resp.domain,
        model=resp.model,
        backend=resp.backend,
        tokens_generated=resp.tokens_generated,
        ttft_ms=resp.ttft_ms,
        latency_ms=resp.latency_ms,
        tokens_per_second=resp.tokens_per_second,
    )
