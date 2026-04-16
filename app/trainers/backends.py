"""
Backend resolver: Unsloth vs MLX-LM vs plain TRL/PEFT/bnb.

Every trainer calls `load_model_and_tokenizer(...)` and `apply_lora(...)` and
doesn't care which backend answered. Each branch is wrapped in a try/except so
a missing dep demotes to the next-best option.
"""
from __future__ import annotations

import logging
from typing import Tuple

from app.hardware import HardwareProfile
from app.hardware.profiles import ResolvedProfile

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Model / tokenizer loading
# ──────────────────────────────────────────────────────────────
def load_model_and_tokenizer(
    *, model_id: str, profile: ResolvedProfile, hardware: HardwareProfile
) -> Tuple[object, object]:
    backend = profile.training_backend
    if backend == "unsloth":
        try:
            return _load_unsloth(model_id, profile)
        except Exception as e:
            logger.warning(f"⚠️  Unsloth load failed ({e}) — falling back to TRL")
            backend = "trl"
    if backend == "mlx":
        try:
            return _load_mlx(model_id, profile)
        except Exception as e:
            logger.warning(f"⚠️  MLX load failed ({e}) — falling back to TRL (MPS)")
            backend = "trl"
    if backend == "trl_cpu":
        return _load_trl(model_id, profile, hardware, force_cpu=True)
    return _load_trl(model_id, profile, hardware)


def _load_unsloth(model_id: str, profile: ResolvedProfile):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=profile.max_seq_length,
        dtype=None,                      # auto bf16 on Ampere+
        load_in_4bit=profile.load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _load_mlx(model_id: str, profile: ResolvedProfile):
    """
    MLX-LM returns (model, tokenizer) objects that are **not** PyTorch modules.
    The MLX trainer branches in sft_trainer / dpo_trainer detect this and call
    the MLX training APIs directly.
    """
    from mlx_lm import load
    model, tokenizer = load(model_id)
    return model, tokenizer


def _resolve_attn_impl(prefer_flash: bool, *, force_cpu: bool) -> str | None:
    """Pick a safe `attn_implementation` value for AutoModelForCausalLM.

    The hardware profile flips `use_flash_attn=True` for any CUDA box,
    but FA2 actually needs *both*:
      * the `flash_attn` package importable, AND
      * an Ampere-or-newer GPU (compute capability >= 8.0).

    Colab's free T4 (Turing, cc 7.5) trips this — it gets matched as
    `cuda_consumer` in the profile resolver, then transformers raises
    "FlashAttention2 has been toggled on, but it cannot be used".

    We probe both conditions and gracefully fall back to `sdpa` (PyTorch
    Scaled Dot Product Attention — fast on Turing+, no extra deps) when
    FA2 isn't viable. Returning None lets transformers auto-pick, which
    is also fine; we pick "sdpa" explicitly so the choice is logged.
    """
    if not prefer_flash or force_cpu:
        return None

    # 1. Is the `flash_attn` package importable?
    try:
        import importlib
        if importlib.util.find_spec("flash_attn") is None:
            logger.info("ℹ️  flash_attn not installed — using SDPA attention instead")
            return "sdpa"
    except Exception:
        return "sdpa"

    # 2. Does the GPU's compute capability support FA2 (>= 8.0)?
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 8:
                name = torch.cuda.get_device_name(0)
                logger.info(
                    f"ℹ️  GPU {name!r} (compute {major}.x) is pre-Ampere — "
                    f"flash-attention-2 unsupported, using SDPA"
                )
                return "sdpa"
    except Exception:
        return "sdpa"

    return "flash_attention_2"


def _load_trl(
    model_id: str,
    profile: ResolvedProfile,
    hardware: HardwareProfile,
    *,
    force_cpu: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[profile.torch_dtype]

    quant_cfg = None
    if profile.load_in_4bit and not force_cpu:
        try:
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            logger.warning("⚠️  bitsandbytes unavailable — loading at full precision")

    device_map = "auto"
    if force_cpu:
        device_map = {"": "cpu"}
    elif hardware.tier == "apple_silicon":
        device_map = {"": "mps"}

    attn_impl = _resolve_attn_impl(profile.use_flash_attn, force_cpu=force_cpu)

    # transformers 4.46+ renamed `torch_dtype` -> `dtype` and emits a
    # DeprecationWarning on the old name. Pick the right one by probing
    # the from_pretrained signature so we don't pin to a transformers
    # version. Fall back to `torch_dtype` for legacy installs.
    import inspect as _inspect
    _fp_params = set(_inspect.signature(AutoModelForCausalLM.from_pretrained).parameters)
    dtype_kwarg = "dtype" if "dtype" in _fp_params else "torch_dtype"

    load_kwargs = dict(
        quantization_config=quant_cfg,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    load_kwargs[dtype_kwarg] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


# ──────────────────────────────────────────────────────────────
# LoRA application
# ──────────────────────────────────────────────────────────────
def apply_lora(model, *, profile: ResolvedProfile, backend: str):
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    if backend == "unsloth":
        try:
            from unsloth import FastLanguageModel
            return FastLanguageModel.get_peft_model(
                model,
                r=profile.lora_r,
                target_modules=target_modules,
                lora_alpha=profile.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
        except Exception as e:
            logger.warning(f"⚠️  Unsloth LoRA failed ({e}) — falling back to PEFT")

    if backend == "mlx":
        # MLX-LM applies LoRA inside its own training loop via `lora_layers` arg.
        # We return the model unchanged here and let sft_trainer handle it.
        return model

    from peft import LoraConfig, get_peft_model, TaskType

    # 4-bit / 8-bit quantized models loaded via bitsandbytes need an extra
    # `prepare_model_for_kbit_training` pass before LoRA: it casts the
    # layernorms to fp32, enables gradient checkpointing reentrancy, and
    # makes sure the input embeddings produce gradients. Without it the
    # LoRA adapters either fail to receive gradients or the very first
    # backward call raises "element 0 of tensors does not require grad".
    if profile.load_in_4bit or profile.load_in_8bit:
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=profile.gradient_checkpointing,
            )
        except Exception as e:
            logger.warning(f"⚠️  prepare_model_for_kbit_training failed: {e}")

    cfg = LoraConfig(
        r=profile.lora_r,
        lora_alpha=profile.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, cfg)
