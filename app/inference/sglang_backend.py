"""
SGLang backend.

SGLang's `Engine` class supports RadixAttention prefix caching out of the box
and offers the best throughput on large MoE models (DeepSeek-V3, Qwen3-235B).

We use the low-level `Engine.generate()` for single-prompt inference. For
agentic workflows the user can import `sgl` directly and define `sgl.function`s.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

from .manager import GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)


class SGLangBackend:

    def __init__(self, base_model_id: str, *, profile, lora_registry: Dict[str, str]):
        from sglang import Engine

        self.model_id = base_model_id
        self.lora_registry = lora_registry

        dtype_map = {"bfloat16": "bfloat16", "float16": "float16", "fp8": "fp8_e4m3", "int8": "int8", "int4": "auto"}
        dtype = dtype_map.get(profile.inference_dtype, "auto")

        logger.info(f"⚡ SGLang Engine init: {base_model_id} | dtype={dtype}")
        self._engine = Engine(
            model_path=base_model_id,
            dtype=dtype,
            trust_remote_code=True,
            enable_radix_cache=True,         # RadixAttention prefix reuse
            kv_cache_dtype=profile.kv_cache_dtype if profile.kv_cache_dtype != "auto" else "auto",
            mem_fraction_static=0.85,
        )

    def register_adapter(self, domain_name: str, adapter_path: str):
        # SGLang 0.3+ supports LoRA via `Engine(..., lora_paths=...)`. For runtime hot-swap
        # we keep the adapter registry for the router; hot-swap arrives via re-init.
        self.lora_registry[domain_name] = adapter_path

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        t_start = time.perf_counter()
        sampling = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_new_tokens": req.max_new_tokens,
            "stop": req.stop or [],
        }
        out = self._engine.generate(prompt=req.prompt, sampling_params=sampling)
        t_end = time.perf_counter()

        text = out["text"] if isinstance(out, dict) else str(out)
        lat_ms = (t_end - t_start) * 1000

        meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
        n_tokens = int(meta.get("completion_tokens", 0)) or len(text.split())
        ttft_ms = float(meta.get("first_token_latency", lat_ms / max(n_tokens, 1)))
        tps = (n_tokens / (t_end - t_start)) if t_end > t_start and n_tokens else 0.0

        return GenerationResponse(
            text=text.strip(),
            backend="sglang",
            model=self.model_id,
            domain=req.domain_name,
            ttft_ms=ttft_ms,
            latency_ms=lat_ms,
            tokens_generated=n_tokens,
            tokens_per_second=tps,
        )
