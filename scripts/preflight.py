"""
Preflight — run before every install/launch.

Checks:
  1. Hardware detection
  2. Resolved training + inference profile
  3. Template registry (imports cleanly)
  4. OCR engine availability
  5. Optional: quick matmul smoke test on the chosen accelerator
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    from app.hardware import detect_hardware, resolve_profile
    from app.templates import list_templates
    from app.data_forge.ocr.pipeline import list_available_engines
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def main():
    hw = detect_hardware()
    prof = resolve_profile(hw)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  ValonyLabs Studio v3.0 — Preflight Check")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Hardware tier     : {hw.tier}")
    print(f"Device            : {hw.device_name}")
    print(f"Accelerator       : {hw.accelerator}")
    print(f"Effective memory  : {hw.effective_memory_gb} GB")
    print(f"bf16 supported    : {hw.supports_bf16}")
    print(f"fp8 supported     : {hw.supports_fp8}")
    print(f"FlashAttention    : {hw.supports_flash_attn}")
    print(f"Host env          : {hw.env.get('host', 'local')}")
    print()
    print("Training backend  :", prof.training_backend)
    print("Inference backend :", prof.inference_backend)
    print("torch_dtype       :", prof.torch_dtype)
    print("max_seq_length    :", prof.max_seq_length)
    print("lora_r / alpha    :", f"{prof.lora_r} / {prof.lora_alpha}")
    print()
    print("Templates registered :", ", ".join(list_templates()))
    print("OCR engines ready    :", ", ".join(list_available_engines()) or "(none)")
    print()

    _matmul_smoke_test()


def _matmul_smoke_test():
    """Tiny tensor op on the preferred device — confirms the stack is wired."""
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        a = torch.randn(1024, 1024, device=dev)
        b = torch.randn(1024, 1024, device=dev)
        c = a @ b
        print(f"Matmul smoke test : {dev} ✅ (result norm = {c.norm().item():.2f})")
    except Exception as e:
        print(f"Matmul smoke test : ⚠️  {e}")


if __name__ == "__main__":
    main()
