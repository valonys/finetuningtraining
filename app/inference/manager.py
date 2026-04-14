"""
app/inference/manager.py
────────────────────────
Unified inference router.

Routing rules (highest priority wins):
  1. Explicit `backend=...` argument
  2. Hardware profile → `profile.inference_backend`
  3. First backend whose dependency imports cleanly

Every backend implements the same `generate(req) → resp` interface, reports
p50/p90/p99 TTFT, and exposes a shared `register_adapter(name, path)` method
so LoRA hot-swap works even without re-loading the base model.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.hardware import detect_hardware, resolve_profile

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Shared dataclasses
# ──────────────────────────────────────────────────────────────
@dataclass
class GenerationRequest:
    prompt: str
    domain_name: str                          # selects the LoRA adapter (or "base")
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1
    stop: Optional[List[str]] = None
    stream: bool = False


@dataclass
class GenerationResponse:
    text: str
    backend: str
    model: str
    domain: str
    ttft_ms: float = 0.0                       # time to first token
    latency_ms: float = 0.0                    # total latency
    tokens_generated: int = 0
    tokens_per_second: float = 0.0


# ──────────────────────────────────────────────────────────────
# Main manager
# ──────────────────────────────────────────────────────────────
class InferenceManager:

    def __init__(self, base_model_id: str, *, backend: Optional[str] = None):
        self.base_model_id = base_model_id
        self.lora_registry: Dict[str, str] = {}
        self._scan_adapters()

        hw = detect_hardware()
        profile = resolve_profile(hw)
        self._hw_tier = hw.tier
        chosen = backend or profile.inference_backend
        logger.info(f"🎯 Inference target backend: {chosen}")

        # Try the preferred backend first, then degrade gracefully
        order = _ordered_fallback(chosen, hw.tier)
        self._backend = None
        for name in order:
            try:
                self._backend = self._init_backend(name, profile)
                self._backend_name = name
                logger.info(f"🚀 Inference backend: {name}")
                break
            except Exception as e:
                logger.warning(f"⚠️  Backend '{name}' unavailable: {e}")
        if self._backend is None:
            raise RuntimeError(
                "No inference backend could be initialised. "
                "Install at least one of: vllm, sglang, mlx-lm, llama-cpp-python, transformers."
            )

        # TTFT telemetry (rolling window)
        self._ttft_samples: list[float] = []
        self._lat_samples: list[float] = []

    # ── Public API ────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        domain_name: str = "base",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> str:
        req = GenerationRequest(
            prompt=prompt,
            domain_name=domain_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        t0 = time.perf_counter()
        resp = self._backend.generate(req)
        t1 = time.perf_counter()

        # Record telemetry
        self._lat_samples.append((t1 - t0) * 1000)
        if resp.ttft_ms:
            self._ttft_samples.append(resp.ttft_ms)
        _trim(self._ttft_samples)
        _trim(self._lat_samples)

        return resp.text

    def generate_full(self, req: GenerationRequest) -> GenerationResponse:
        return self._backend.generate(req)

    def generate_stream(self, req: GenerationRequest):
        """
        Incremental generation — yields text deltas as they arrive, then
        a final dict with summary metadata (backend, model, ttft_ms,
        latency_ms, tokens_generated, tokens_per_second).

        Downstream SSE endpoint wraps each yielded item into a
        `data: {json}\\n\\n` frame. Backends that don't support token-
        level streaming (vLLM's simple path, mlx without `stream_generate`,
        etc.) may fall back to a single final-delta emission — the
        contract is preserved, just coarser.

        For now the Ollama backend is the authoritative streaming path
        (Server-Sent Events from Ollama Cloud or the local daemon). Other
        backends raise NotImplementedError which the endpoint converts
        to a 501 with a helpful message.
        """
        import time
        if not hasattr(self._backend, "stream"):
            raise NotImplementedError(
                f"Backend '{self._backend_name}' does not yet support "
                f"token streaming. Use /v1/chat for a batched response, "
                f"or switch to the ollama backend via "
                f"VALONY_INFERENCE_BACKEND=ollama."
            )
        t_start = time.perf_counter()
        ttft_ms = 0.0
        n_tokens = 0
        model_used = getattr(self._backend, "model_id", self.base_model_id)

        for delta in self._backend.stream(req):
            if n_tokens == 0:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            n_tokens += 1
            yield delta

        t_end = time.perf_counter()
        lat_ms = (t_end - t_start) * 1000
        tps = (n_tokens / (t_end - t_start)) if (t_end > t_start and n_tokens) else 0.0

        # Record in the rolling telemetry so /healthz reflects stream stats too
        if ttft_ms:
            self._ttft_samples.append(ttft_ms)
        self._lat_samples.append(lat_ms)
        _trim(self._ttft_samples)
        _trim(self._lat_samples)

        yield {
            "__meta__": True,
            "backend": self._backend_name,
            "model": model_used,
            "domain": req.domain_name,
            "ttft_ms": ttft_ms or (lat_ms / max(n_tokens, 1)),
            "latency_ms": lat_ms,
            "tokens_generated": n_tokens,
            "tokens_per_second": tps,
        }

    def register_adapter(self, domain_name: str, adapter_path: str) -> None:
        self.lora_registry[domain_name] = adapter_path
        if hasattr(self._backend, "register_adapter"):
            self._backend.register_adapter(domain_name, adapter_path)
        logger.info(f"📎 Registered adapter: {domain_name} → {adapter_path}")

    @property
    def registered_domains(self) -> List[str]:
        return list(self.lora_registry.keys())

    @property
    def backend(self) -> str:
        return self._backend_name

    def latency_stats(self) -> Dict[str, float]:
        import statistics
        out: Dict[str, float] = {}
        if self._ttft_samples:
            out["ttft_p50"] = statistics.median(self._ttft_samples)
            out["ttft_p90"] = _percentile(self._ttft_samples, 0.9)
            out["ttft_p99"] = _percentile(self._ttft_samples, 0.99)
        if self._lat_samples:
            out["latency_p50"] = statistics.median(self._lat_samples)
            out["latency_p90"] = _percentile(self._lat_samples, 0.9)
            out["latency_p99"] = _percentile(self._lat_samples, 0.99)
        return out

    # ── Private ───────────────────────────────────────────────
    def _scan_adapters(self) -> None:
        outputs_dir = "outputs"
        if not os.path.isdir(outputs_dir):
            return
        for name in os.listdir(outputs_dir):
            path = os.path.join(outputs_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                self.lora_registry[name] = path

    def _init_backend(self, name: str, profile) -> Any:
        if name == "vllm":
            from .vllm_backend import VLLMBackend
            return VLLMBackend(self.base_model_id, profile=profile, lora_registry=self.lora_registry)
        if name == "sglang":
            from .sglang_backend import SGLangBackend
            return SGLangBackend(self.base_model_id, profile=profile, lora_registry=self.lora_registry)
        if name == "mlx":
            from .mlx_backend import MLXBackend
            return MLXBackend(self.base_model_id, profile=profile, lora_registry=self.lora_registry)
        if name == "llamacpp":
            from .llamacpp_backend import LlamaCppBackend
            return LlamaCppBackend(self.base_model_id, profile=profile, lora_registry=self.lora_registry)
        if name == "hf":
            from .hf_backend import HFBackend
            return HFBackend(self.base_model_id, profile=profile, lora_registry=self.lora_registry)
        if name == "ollama":
            from .ollama_backend import OllamaInferenceBackend
            return OllamaInferenceBackend(self.base_model_id, profile=profile, lora_registry=self.lora_registry)
        raise ValueError(f"Unknown backend: {name}")


def _ordered_fallback(preferred: str, hw_tier: str) -> List[str]:
    """
    Ordered list of backends to try — preferred first, then sane defaults.

    Ollama is never auto-appended to the fallback chain: it depends on an
    external daemon (or a cloud API key) that we can't assume is available,
    and its LoRA semantics differ from the other backends. Users opt in via
    `backend="ollama"` or `VALONY_INFERENCE_BACKEND=ollama`.
    """
    order: list[str] = [preferred]
    if hw_tier == "apple_silicon":
        extras = ["mlx", "llamacpp", "hf"]
    elif hw_tier.startswith("cuda"):
        extras = ["vllm", "sglang", "hf", "llamacpp"]
    elif hw_tier == "rocm":
        extras = ["vllm", "hf"]
    else:
        extras = ["llamacpp", "hf"]
    for b in extras:
        if b not in order:
            order.append(b)
    return order


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def _trim(xs: list, max_len: int = 256):
    if len(xs) > max_len:
        del xs[: len(xs) - max_len]


# ──────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────
_engine: Optional[InferenceManager] = None


def get_inference_engine(base_model: str, *, backend: Optional[str] = None) -> InferenceManager:
    global _engine
    if _engine is None or _engine.base_model_id != base_model:
        _engine = InferenceManager(base_model, backend=backend)
    return _engine


def is_engine_loaded() -> bool:
    """True iff the singleton has been initialised (without forcing init).

    Used by /healthz and other read-only endpoints so they can report the
    engine state cheaply, without paying the cost of a 14 GB model
    download just to render a status card.
    """
    return _engine is not None


def get_loaded_engine() -> Optional["InferenceManager"]:
    """Return the singleton if loaded, else None. Never triggers init."""
    return _engine


def reset_engine() -> None:
    global _engine
    _engine = None
