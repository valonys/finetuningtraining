"""
app/main.py
───────────
FastAPI entrypoint for ValonyLabs Studio v3.0.

Endpoints
─────────
  GET  /healthz                          Hardware + backend + telemetry
  GET  /v1/templates                     List registered chat templates
  GET  /v1/ocr/engines                   List available OCR engines

  POST /v1/forge/upload                  Upload one or more files (multipart)
  GET  /v1/forge/uploads                 List currently-uploaded files
  DELETE /v1/forge/uploads               Clear every uploaded file
  DELETE /v1/forge/uploads/{filename}    Delete one uploaded file
  POST /v1/forge/harvest/youtube         Keyword-search YouTube, save transcripts

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

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

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
    get_loaded_engine,
    is_engine_loaded,
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
    UploadListResponse,
    UploadResponse,
    UploadedFileInfo,
    YouTubeHarvestRequest,
    YouTubeHarvestResponse,
    YouTubeHarvestedFile,
)
from app.uploads import (
    UploadError,
    clear_uploads,
    delete_upload,
    list_uploads,
    save_upload,
    total_upload_bytes,
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
# Pre-warming the inference engine downloads the base model into the HF
# cache (~14 GB for a 7B). On a fresh machine that blocks startup for
# minutes — uvicorn won't accept any requests until the lifespan
# completes. So pre-warm is OPT-IN: set VALONY_PREWARM_INFERENCE=1 to
# enable it. Default is lazy — the first /v1/chat call triggers the
# load, and meanwhile every other endpoint (Health, Domains, Data
# Forge, Train) responds instantly.
PREWARM_INFERENCE = os.environ.get("VALONY_PREWARM_INFERENCE", "").lower() in (
    "1", "true", "yes", "on"
)


# ──────────────────────────────────────────────────────────────
# Lifespan — fast startup, lazy inference engine
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 ValonyLabs Studio v3.0 starting")
    try:
        hw = detect_hardware()
        prof = resolve_profile(hw)
        logger.info(f"   Hardware: {hw.tier} | {hw.device_name} | mem={hw.effective_memory_gb} GB")
        logger.info(f"   Training backend: {prof.training_backend}")
        logger.info(f"   Inference backend: {DEFAULT_INFERENCE_BACKEND or prof.inference_backend}")
    except Exception as e:
        logger.warning(f"   ⚠️  Hardware detection failed at startup: {e}")

    if PREWARM_INFERENCE:
        # Eager load — useful in production where you want a deterministic
        # first-request latency, but blocks startup until the model is
        # downloaded (potentially many GB).
        logger.info(f"   Pre-warming inference engine with {DEFAULT_BASE_MODEL} ...")
        try:
            engine = get_inference_engine(DEFAULT_BASE_MODEL, backend=DEFAULT_INFERENCE_BACKEND)
            logger.info(f"   ✅ Inference engine: {engine.backend}")
        except Exception as e:
            logger.warning(f"   ⚠️  Inference pre-warm failed: {e}")
    else:
        logger.info(
            "   Inference engine: lazy (will load on first /v1/chat call). "
            "Set VALONY_PREWARM_INFERENCE=1 to load eagerly at startup."
        )
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
    """
    Resilient health check — every subsystem is fetched in its own
    try/except block so that a missing optional dep (numpy, torch,
    transformers, ...) can't take the whole endpoint down. Any failed
    section is logged and replaced with a safe default. The endpoint
    itself should never return 500.
    """
    # Hardware detection + profile resolution
    try:
        hw = detect_hardware()
        hardware_dict = hw.as_dict()
    except Exception as e:
        logger.warning(f"healthz: detect_hardware failed: {e}")
        hardware_dict = {"tier": "unknown", "error": str(e)}

    try:
        prof = resolve_profile()
        profile_dict = {
            "training_backend": prof.training_backend,
            "inference_backend": prof.inference_backend,
            "torch_dtype": prof.torch_dtype,
            "max_seq_length": prof.max_seq_length,
            "lora_r": prof.lora_r,
        }
    except Exception as e:
        logger.warning(f"healthz: resolve_profile failed: {e}")
        profile_dict = {"error": str(e)}

    # Inference engine — read state ONLY if already loaded. Never force a
    # cold-init here: that would download multi-GB model weights on the
    # very first /healthz call and block every other request until done.
    try:
        engine = get_loaded_engine()
        if engine is not None:
            domains = engine.registered_domains
            backend = engine.backend
            stats = engine.latency_stats()
        else:
            domains, backend, stats = [], "lazy_not_loaded", {}
    except Exception as e:
        logger.warning(f"healthz: inference engine state read failed: {e}")
        domains, backend, stats = [], "error", {}

    # OCR engines — wrapped because rapidocr_engine imports numpy at module level
    try:
        available_ocr = list_available_engines()
    except Exception as e:
        logger.warning(f"healthz: list_available_engines failed: {e}")
        available_ocr = []

    # Chat templates
    try:
        available_templates = list_templates()
    except Exception as e:
        logger.warning(f"healthz: list_templates failed: {e}")
        available_templates = []

    # Synth provider descriptor
    try:
        synth_provider = describe_active_provider()
    except Exception as e:
        logger.warning(f"healthz: describe_active_provider failed: {e}")
        synth_provider = {"provider": "unknown", "error": str(e)}

    return HealthResponse(
        status="ok",
        version=__version__,
        hardware=hardware_dict,
        profile=profile_dict,
        registered_domains=domains,
        inference_backend=backend,
        latency_stats=stats,
        available_ocr=available_ocr,
        available_templates=available_templates,
        synth_provider=synth_provider,
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
# Data Forge — uploads
# ──────────────────────────────────────────────────────────────
@app.post("/v1/forge/upload", response_model=UploadResponse)
async def forge_upload(files: List[UploadFile] = File(...)):
    """
    Stream one or more files to `./data/uploads/`. The frontend passes
    the returned paths to `/v1/forge/build_dataset`. Rejected files (empty,
    oversize) are listed in `skipped` with a human-readable reason so the
    UI can show per-file feedback without failing the whole batch.
    """
    uploaded: List[UploadedFileInfo] = []
    skipped: List[Dict[str, str]] = []
    for f in files:
        try:
            rec = await save_upload(f)
            uploaded.append(
                UploadedFileInfo(name=rec.name, path=rec.path, size=rec.size)
            )
        except UploadError as e:
            skipped.append({"name": f.filename or "(unnamed)", "reason": str(e)})
        except Exception as e:
            logger.exception(f"upload failed for {f.filename}")
            skipped.append({"name": f.filename or "(unnamed)", "reason": str(e)})
    return UploadResponse(uploaded=uploaded, skipped=skipped)


@app.get("/v1/forge/uploads", response_model=UploadListResponse)
async def forge_list_uploads():
    files = list_uploads()
    return UploadListResponse(
        files=[UploadedFileInfo(name=f.name, path=f.path, size=f.size) for f in files],
        total_bytes=total_upload_bytes(),
    )


@app.delete("/v1/forge/uploads")
async def forge_clear_uploads():
    """Delete every file in the uploads directory."""
    count = clear_uploads()
    return {"deleted": count}


@app.delete("/v1/forge/uploads/{filename}")
async def forge_delete_upload(filename: str):
    try:
        ok = delete_upload(filename)
    except UploadError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(404, f"No uploaded file named '{filename}'")
    return {"deleted": filename}


# ──────────────────────────────────────────────────────────────
# Data Forge — YouTube transcript harvester
# ──────────────────────────────────────────────────────────────
@app.post("/v1/forge/harvest/youtube", response_model=YouTubeHarvestResponse)
async def forge_harvest_youtube(req: YouTubeHarvestRequest):
    """
    Keyword-search YouTube, fetch transcripts for the top N matches, and
    write each as a plain-text file under `./data/uploads/`. After this,
    the same file list shows up in `/v1/forge/uploads` and can be fed
    straight into `/v1/forge/build_dataset` like any other upload.

    Uses yt-dlp for search (no API key) + youtube-transcript-api for
    captions (fast, no STT). Videos without captions are skipped and
    listed in the response's `skipped` field.
    """
    try:
        from app.harvesters.youtube import YouTubeHarvester
    except Exception as e:
        raise HTTPException(
            500,
            f"Harvester unavailable: {e}. "
            f"Run `pip install yt-dlp youtube-transcript-api` and restart.",
        )

    try:
        harvester = YouTubeHarvester()
        # The yt-dlp search and transcript fetch are synchronous network
        # calls; push them to a thread so they don't block the event loop.
        report = await asyncio.to_thread(
            harvester.harvest,
            req.query,
            max_results=req.max_videos,
            output_dir=req.output_dir,
            min_chars=req.min_chars,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception("forge_harvest_youtube failed")
        raise HTTPException(500, f"Harvest failed: {e}")

    return YouTubeHarvestResponse(
        query=report.query,
        max_requested=report.max_requested,
        harvested=[
            YouTubeHarvestedFile(
                title=h.title,
                url=h.url,
                channel=h.channel,
                language=h.language,
                auto_generated=h.auto_generated,
                char_count=h.char_count,
                duration_s=h.duration_s,
                file_path=h.file_path,
            )
            for h in report.harvested
        ],
        skipped=report.skipped,
    )


# ──────────────────────────────────────────────────────────────
# Data Forge — ingest / build
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
        # Normalise "auto" sentinel to None so the DatasetBuilder
        # auto-resolves the template from the base model id.
        template_override = (
            None if not req.template or req.template == "auto" else req.template
        )
        ds = forge.build_dataset(
            records,
            task=req.task,
            base_model=req.base_model,
            system_prompt=req.system_prompt,
            synth_qa=req.synth_qa,
            synth_mode=req.synth_mode,
            target_size=req.target_size,
            template_override=template_override,
            filter_noise=req.filter_noise,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Persist in human-readable form rather than HF's binary Arrow dataset.
    # JSONL is:
    #   - loadable back by TRL / HF datasets via
    #     load_dataset("json", data_files="...jsonl", split="train"),
    #   - greppable / diffable / peekable in any editor,
    #   - the de-facto interchange format for SFT / DPO / GRPO datasets.
    # We also drop a pretty-printed `_preview.json` with the first 10 rows
    # so you can sneak-peek a sample in any editor without streaming the
    # whole file.
    import json as _json
    os.makedirs(req.output_dir, exist_ok=True)
    stem = f"{req.task}_{os.urandom(3).hex()}"
    jsonl_path = os.path.join(req.output_dir, f"{stem}.jsonl")
    preview_path = os.path.join(req.output_dir, f"{stem}_preview.json")

    # Full dataset as JSONL (one row per line).
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(_json.dumps(row, ensure_ascii=False) + "\n")

    # First 10 rows, pretty-printed, for human inspection.
    preview_rows = [ds[i] for i in range(min(10, len(ds)))]
    with open(preview_path, "w", encoding="utf-8") as f:
        _json.dump(preview_rows, f, ensure_ascii=False, indent=2)

    from app.templates import get_template_for
    template = get_template_for(req.base_model)
    return ForgeBuildResponse(
        output_path=jsonl_path,
        preview_path=preview_path,
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
    """List trained adapters that are registered with the inference engine.

    Same lazy pattern as /healthz — if the engine isn't loaded yet, return
    an empty list rather than triggering a multi-GB model download just to
    enumerate the (empty) adapter registry. Once the engine is loaded
    (e.g. after the first /v1/chat call), this returns the real list.
    """
    engine = get_loaded_engine()
    if engine is None:
        return []
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


# ──────────────────────────────────────────────────────────────
# Streaming chat — Server-Sent Events (SSE)
# ──────────────────────────────────────────────────────────────
@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Stream the assistant response back as text/event-stream so the UI
    can show the typewriter effect — tokens appear in the chat bubble
    as Nemotron (or whichever model) emits them, instead of the user
    staring at a spinner for 5 seconds before the whole reply appears.

    Frame format (one per line, separated by `\\n\\n`):
        data: {"delta": "Hello"}
        data: {"delta": " world"}
        data: {"meta": {"backend":"ollama","ttft_ms":950, ...}}
        data: [DONE]

    An errored stream emits a final `data: {"error": "..."}` frame
    before [DONE] so the frontend can surface the message rather than
    the request silently aborting.

    Only backends that expose `stream()` on the underlying backend
    class support this endpoint — currently `ollama`. Others return
    HTTP 501.
    """
    engine = get_inference_engine(DEFAULT_BASE_MODEL, backend=DEFAULT_INFERENCE_BACKEND)

    from app.templates import get_template_for
    template = get_template_for(engine.base_model_id)

    # ── Three modes ──────────────────────────────────────────────
    #   "docs"   → RAG over the in-app Docs corpus, force-formatted
    #              MS-Office-style markdown response, citations
    #   <name>   → use the named domain config's system_prompt
    #   "base"   → talk to the raw base model
    docs_mode = (req.domain_config_name or "").lower() == "docs"
    sources_payload: List[Dict[str, Any]] = []

    if docs_mode:
        from app.rag import build_rag_prompt
        from app.rag.prompts import hits_to_sources
        system_prompt, hits = build_rag_prompt(req.message, k=5)
        sources_payload = hits_to_sources(hits)
    else:
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

    def _sync_generator():
        """Yield backend deltas (strings) and the final meta dict."""
        try:
            yield from engine.generate_stream(gen_req)
        except NotImplementedError as e:
            yield {"__error__": True, "message": str(e), "status": 501}
        except Exception as e:
            logger.exception("chat_stream: backend error")
            yield {"__error__": True, "message": str(e), "status": 500}

    async def event_source():
        # iterate_in_threadpool bridges our sync generator to the async
        # event loop without blocking it — each backend-yielded chunk
        # arrives as soon as Ollama hands it over the wire.
        import json as _json

        # Emit retrieved sources up-front (only in docs mode) so the UI
        # can render citation badges before the answer streams in.
        if sources_payload:
            yield f"data: {_json.dumps({'sources': sources_payload})}\n\n"

        try:
            async for item in iterate_in_threadpool(_sync_generator()):
                if isinstance(item, dict) and item.get("__error__"):
                    yield f"data: {_json.dumps({'error': item['message']})}\n\n"
                elif isinstance(item, dict) and item.get("__meta__"):
                    meta = {k: v for k, v in item.items() if not k.startswith("__")}
                    if sources_payload:
                        meta["sources_count"] = len(sources_payload)
                        meta["mode"] = "docs"
                    yield f"data: {_json.dumps({'meta': meta})}\n\n"
                else:
                    yield f"data: {_json.dumps({'delta': item})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        # Explicit charset=utf-8 so any browser / proxy that defaults
        # to Latin-1 / Windows-1252 doesn't mangle em dashes or other
        # non-ASCII glyphs into mojibake.
        media_type="text/event-stream; charset=utf-8",
        headers={
            # Disable buffering from nginx / Vite proxy so bytes flow immediately
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
