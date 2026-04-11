"""
app/hardware/detect.py
──────────────────────
Single source of truth for "what machine am I running on?"

Rules:
  - Apple Silicon (M1..M4) → tier="apple_silicon", accelerator="mps"
  - NVIDIA CUDA (Ampere+)   → tier="cuda_datacenter" if vram ≥ 40, else "cuda_consumer"
  - NVIDIA CUDA (Turing)    → tier="cuda_legacy" (T4 / RTX 20xx)
  - AMD ROCm                → tier="rocm"
  - Intel / XPU             → tier="xpu"
  - Otherwise               → tier="cpu"

Everything downstream (trainers, inference engines, data forge) asks
`detect_hardware()` first and branches on `profile.tier`.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
from dataclasses import dataclass, asdict, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    tier: str                        # apple_silicon | cuda_consumer | cuda_datacenter | cuda_legacy | rocm | xpu | cpu
    accelerator: str                 # mps | cuda | rocm | xpu | cpu
    device_name: str
    vram_gb: int                     # 0 on CPU / shared memory → use unified_memory_gb
    unified_memory_gb: int           # Apple unified memory; 0 otherwise
    compute_capability: Optional[tuple[int, int]] = None   # (major, minor) for CUDA
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_flash_attn: bool = False
    os_name: str = ""
    arch: str = ""
    env: dict = field(default_factory=dict)   # "colab", "kaggle", "brev", etc.

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def is_mac(self) -> bool:
        return self.tier == "apple_silicon"

    @property
    def is_cuda(self) -> bool:
        return self.accelerator == "cuda"

    @property
    def effective_memory_gb(self) -> int:
        """Memory budget the trainers / inference engines should target."""
        return self.vram_gb or self.unified_memory_gb


# ──────────────────────────────────────────────────────────────
# Cached detection (hardware doesn't change inside one process)
# ──────────────────────────────────────────────────────────────
_cache: Optional[HardwareProfile] = None


def detect_hardware(force: bool = False) -> HardwareProfile:
    global _cache
    if _cache and not force:
        return _cache

    os_name = platform.system()          # "Darwin" | "Linux" | "Windows"
    arch = platform.machine()            # "arm64" | "x86_64"
    env_hints = _sniff_env()

    # ── Apple Silicon first — if we're on Darwin+arm64 we prefer MPS ──
    if os_name == "Darwin" and arch == "arm64":
        unified = _apple_unified_memory_gb()
        supports_bf16 = _mps_supports_bf16()
        profile = HardwareProfile(
            tier="apple_silicon",
            accelerator="mps",
            device_name=_apple_chip_name(),
            vram_gb=0,
            unified_memory_gb=unified,
            supports_bf16=supports_bf16,
            supports_fp8=False,
            supports_flash_attn=False,
            os_name=os_name,
            arch=arch,
            env=env_hints,
        )
        _cache = profile
        logger.info(f"🍎 Detected Apple Silicon: {profile.device_name} ({unified} GB unified)")
        return profile

    # ── NVIDIA CUDA ───────────────────────────────────────────
    cuda_profile = _try_cuda(os_name, arch, env_hints)
    if cuda_profile is not None:
        _cache = cuda_profile
        return cuda_profile

    # ── AMD ROCm ──────────────────────────────────────────────
    rocm_profile = _try_rocm(os_name, arch, env_hints)
    if rocm_profile is not None:
        _cache = rocm_profile
        return rocm_profile

    # ── Intel XPU (oneAPI) ────────────────────────────────────
    xpu_profile = _try_xpu(os_name, arch, env_hints)
    if xpu_profile is not None:
        _cache = xpu_profile
        return xpu_profile

    # ── CPU fallback ──────────────────────────────────────────
    profile = HardwareProfile(
        tier="cpu",
        accelerator="cpu",
        device_name=platform.processor() or "CPU",
        vram_gb=0,
        unified_memory_gb=_total_system_ram_gb(),
        supports_bf16=False,
        supports_fp8=False,
        supports_flash_attn=False,
        os_name=os_name,
        arch=arch,
        env=env_hints,
    )
    _cache = profile
    logger.info(f"🖥️  CPU-only runtime on {profile.device_name}")
    return profile


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _try_cuda(os_name: str, arch: str, env_hints: dict) -> Optional[HardwareProfile]:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(0)
    vram_gb = int(props.total_memory // (1024 ** 3))
    cap = torch.cuda.get_device_capability(0)        # (major, minor)

    supports_bf16 = cap[0] >= 8                      # Ampere+
    supports_fp8 = cap[0] >= 9                       # Hopper+ (H100/B200)
    supports_flash_attn = cap[0] >= 8
    name = props.name

    # Tier classification
    if cap[0] < 8:
        tier = "cuda_legacy"                         # T4 / RTX 20xx / P100
    elif vram_gb >= 40:
        tier = "cuda_datacenter"                     # A100 40/80, H100, B200
    else:
        tier = "cuda_consumer"                       # RTX 30/40/50, A10G, L4

    profile = HardwareProfile(
        tier=tier,
        accelerator="cuda",
        device_name=name,
        vram_gb=vram_gb,
        unified_memory_gb=0,
        compute_capability=tuple(cap),
        supports_bf16=supports_bf16,
        supports_fp8=supports_fp8,
        supports_flash_attn=supports_flash_attn,
        os_name=os_name,
        arch=arch,
        env=env_hints,
    )
    logger.info(
        f"🟢 Detected CUDA: {name} | {vram_gb} GB VRAM | "
        f"SM {cap[0]}.{cap[1]} | tier={tier} | bf16={supports_bf16} fp8={supports_fp8}"
    )
    return profile


def _try_rocm(os_name: str, arch: str, env_hints: dict) -> Optional[HardwareProfile]:
    try:
        import torch
    except ImportError:
        return None
    # torch.version.hip is set when built against ROCm
    hip_version = getattr(torch.version, "hip", None)
    if not hip_version:
        return None
    if not torch.cuda.is_available():       # ROCm pretends to be "cuda" in torch
        return None
    props = torch.cuda.get_device_properties(0)
    vram_gb = int(props.total_memory // (1024 ** 3))
    profile = HardwareProfile(
        tier="rocm",
        accelerator="rocm",
        device_name=props.name,
        vram_gb=vram_gb,
        unified_memory_gb=0,
        supports_bf16=True,
        supports_fp8=False,
        supports_flash_attn=False,
        os_name=os_name,
        arch=arch,
        env=env_hints,
    )
    logger.info(f"🔴 Detected ROCm: {props.name} | {vram_gb} GB")
    return profile


def _try_xpu(os_name: str, arch: str, env_hints: dict) -> Optional[HardwareProfile]:
    try:
        import torch
    except ImportError:
        return None
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return None
    props = torch.xpu.get_device_properties(0)
    vram_gb = int(getattr(props, "total_memory", 0) // (1024 ** 3))
    profile = HardwareProfile(
        tier="xpu",
        accelerator="xpu",
        device_name=getattr(props, "name", "Intel XPU"),
        vram_gb=vram_gb,
        unified_memory_gb=0,
        supports_bf16=True,
        supports_fp8=False,
        supports_flash_attn=False,
        os_name=os_name,
        arch=arch,
        env=env_hints,
    )
    logger.info(f"🔵 Detected Intel XPU: {profile.device_name} | {vram_gb} GB")
    return profile


def _apple_unified_memory_gb() -> int:
    """sysctl hw.memsize → unified memory in GB."""
    sysctl = shutil.which("sysctl")
    if not sysctl:
        return _total_system_ram_gb()
    try:
        import subprocess
        out = subprocess.check_output([sysctl, "-n", "hw.memsize"], text=True).strip()
        return int(int(out) // (1024 ** 3))
    except Exception:
        return _total_system_ram_gb()


def _apple_chip_name() -> str:
    """`sysctl machdep.cpu.brand_string` → e.g. 'Apple M4 Pro'."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        return out or "Apple Silicon"
    except Exception:
        return "Apple Silicon"


def _mps_supports_bf16() -> bool:
    """MPS added bfloat16 in torch 2.4+; check runtime."""
    try:
        import torch
        if not torch.backends.mps.is_available():
            return False
        # PyTorch ≥ 2.4 supports bf16 on M2+; M1 still has issues
        return int(torch.__version__.split(".")[0]) >= 2 and int(
            torch.__version__.split(".")[1]
        ) >= 4
    except Exception:
        return False


def _total_system_ram_gb() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total // (1024 ** 3))
    except ImportError:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size // (1024 ** 3))
        except Exception:
            return 0


def _sniff_env() -> dict:
    """Detect hosted environments: Colab, Kaggle, Brev, Lambda, RunPod, Sagemaker."""
    env = {}
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        env["host"] = "colab"
    elif "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        env["host"] = "kaggle"
    elif "BREV_API_KEY" in os.environ or os.path.exists("/usr/local/brev"):
        env["host"] = "brev"
    elif "RUNPOD_POD_ID" in os.environ:
        env["host"] = "runpod"
    elif "LAMBDA_INSTANCE_ID" in os.environ:
        env["host"] = "lambda"
    elif "SAGEMAKER_REGION" in os.environ:
        env["host"] = "sagemaker"
    else:
        env["host"] = "local"
    return env


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prof = detect_hardware()
    from pprint import pprint
    pprint(prof.as_dict())
