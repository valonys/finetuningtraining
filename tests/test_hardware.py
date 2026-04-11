"""Hardware detection + profile resolution sanity tests."""
from __future__ import annotations

from app.hardware import detect_hardware, resolve_profile
from app.hardware.profiles import HARDWARE_TIERS


def test_detection_runs():
    hw = detect_hardware()
    assert hw.tier in HARDWARE_TIERS
    assert hw.accelerator in {"mps", "cuda", "rocm", "xpu", "cpu"}


def test_profile_keys_present():
    hw = detect_hardware()
    p = resolve_profile(hw)
    assert p.training_backend in {"unsloth", "mlx", "trl", "trl_cpu"}
    assert p.inference_backend in {"vllm", "sglang", "mlx", "llamacpp", "hf"}
    assert p.max_seq_length > 0
    assert p.lora_r > 0


def test_tier_defaults_consistent():
    # Every tier must have a valid OCR default
    for name, prof in HARDWARE_TIERS.items():
        assert prof.default_ocr in {"rapidocr", "paddleocr", "docling", "tesseract", "trocr"}
        assert prof.torch_dtype in {"bfloat16", "float16", "float32"}
