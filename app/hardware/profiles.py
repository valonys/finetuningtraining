"""
app/hardware/profiles.py
────────────────────────
Tier-specific defaults: LoRA rank, seq length, batch size, precision,
which trainer/inference backend to pick, and which OCR engine is sane.

These are "safe but fast" defaults — the user can override any of them
in their domain config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .detect import HardwareProfile, detect_hardware


@dataclass
class ResolvedProfile:
    # Training
    training_backend: Literal["unsloth", "mlx", "trl", "trl_cpu"]
    load_in_4bit: bool
    load_in_8bit: bool
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    lora_r: int
    lora_alpha: int
    max_seq_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    use_flash_attn: bool
    # Inference
    inference_backend: Literal["vllm", "sglang", "mlx", "llamacpp", "hf"]
    inference_dtype: Literal["bfloat16", "float16", "fp8", "int8", "int4"]
    kv_cache_dtype: Literal["auto", "fp16", "fp8", "int8"]
    # Data Forge
    default_ocr: Literal["rapidocr", "paddleocr", "docling", "tesseract", "trocr"]


# ──────────────────────────────────────────────────────────────
# Tier table (sane defaults — ordered by increasing capability)
# ──────────────────────────────────────────────────────────────
HARDWARE_TIERS: dict[str, ResolvedProfile] = {
    "cpu": ResolvedProfile(
        training_backend="trl_cpu",
        load_in_4bit=False,
        load_in_8bit=False,
        torch_dtype="float32",
        lora_r=4,
        lora_alpha=8,
        max_seq_length=512,
        per_device_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        use_flash_attn=False,
        inference_backend="llamacpp",
        inference_dtype="int4",
        kv_cache_dtype="auto",
        default_ocr="tesseract",
    ),
    "apple_silicon": ResolvedProfile(
        training_backend="mlx",
        load_in_4bit=True,          # mlx-lm supports 4-bit quant training
        load_in_8bit=False,
        torch_dtype="bfloat16",     # M2+ on torch 2.4+
        lora_r=16,
        lora_alpha=32,
        max_seq_length=2048,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        use_flash_attn=False,
        inference_backend="mlx",
        inference_dtype="int4",
        kv_cache_dtype="auto",
        default_ocr="rapidocr",
    ),
    "cuda_legacy": ResolvedProfile(
        training_backend="trl",      # Unsloth needs Ampere; T4 uses PEFT+bnb
        load_in_4bit=True,
        load_in_8bit=False,
        torch_dtype="float16",       # no bf16 on Turing
        lora_r=8,
        lora_alpha=16,
        max_seq_length=1024,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        use_flash_attn=False,
        inference_backend="hf",      # vLLM requires Ampere+
        inference_dtype="int4",
        kv_cache_dtype="auto",
        default_ocr="rapidocr",
    ),
    "cuda_consumer": ResolvedProfile(
        training_backend="unsloth",
        load_in_4bit=True,
        load_in_8bit=False,
        torch_dtype="bfloat16",
        lora_r=16,
        lora_alpha=32,
        max_seq_length=4096,
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        use_flash_attn=True,
        inference_backend="vllm",
        inference_dtype="fp8",       # 5090 / 4090: FP8 via vLLM
        kv_cache_dtype="fp8",
        default_ocr="paddleocr",
    ),
    "cuda_datacenter": ResolvedProfile(
        training_backend="unsloth",
        load_in_4bit=False,          # A100/H100 can afford bf16
        load_in_8bit=False,
        torch_dtype="bfloat16",
        lora_r=32,
        lora_alpha=64,
        max_seq_length=8192,
        per_device_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        use_flash_attn=True,
        inference_backend="vllm",    # or sglang — both fine
        inference_dtype="fp8",
        kv_cache_dtype="fp8",
        default_ocr="docling",
    ),
    "rocm": ResolvedProfile(
        training_backend="trl",      # Unsloth AMD path is new; keep safe
        load_in_4bit=True,
        load_in_8bit=False,
        torch_dtype="bfloat16",
        lora_r=16,
        lora_alpha=32,
        max_seq_length=2048,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        use_flash_attn=False,
        inference_backend="vllm",    # vLLM has ROCm wheels
        inference_dtype="float16",
        kv_cache_dtype="auto",
        default_ocr="rapidocr",
    ),
    "xpu": ResolvedProfile(
        training_backend="trl",
        load_in_4bit=False,
        load_in_8bit=True,
        torch_dtype="bfloat16",
        lora_r=8,
        lora_alpha=16,
        max_seq_length=1024,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        use_flash_attn=False,
        inference_backend="hf",
        inference_dtype="float16",
        kv_cache_dtype="auto",
        default_ocr="rapidocr",
    ),
}


def resolve_profile(hw: HardwareProfile | None = None) -> ResolvedProfile:
    """Return the appropriate ResolvedProfile for the current (or supplied) hardware."""
    hw = hw or detect_hardware()
    prof = HARDWARE_TIERS.get(hw.tier, HARDWARE_TIERS["cpu"])

    # Memory-aware downscaling: if effective memory is too small for the tier,
    # degrade gracefully rather than OOM.
    mem = hw.effective_memory_gb
    if mem and mem < 12:
        prof = _downgrade(prof, lora_r=4, seq_len=1024, batch=1)
    elif mem and mem < 20:
        prof = _downgrade(prof, lora_r=8, seq_len=2048, batch=1)
    return prof


def _downgrade(base: ResolvedProfile, *, lora_r: int, seq_len: int, batch: int) -> ResolvedProfile:
    """Return a copy of `base` with tighter defaults."""
    return ResolvedProfile(
        training_backend=base.training_backend,
        load_in_4bit=True,
        load_in_8bit=False,
        torch_dtype=base.torch_dtype,
        lora_r=min(base.lora_r, lora_r),
        lora_alpha=min(base.lora_alpha, lora_r * 2),
        max_seq_length=min(base.max_seq_length, seq_len),
        per_device_batch_size=min(base.per_device_batch_size, batch),
        gradient_accumulation_steps=base.gradient_accumulation_steps,
        gradient_checkpointing=True,
        use_flash_attn=base.use_flash_attn,
        inference_backend=base.inference_backend,
        inference_dtype=base.inference_dtype,
        kv_cache_dtype=base.kv_cache_dtype,
        default_ocr=base.default_ocr,
    )
