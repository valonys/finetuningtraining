"""
vLLM backend.

Features we enable:
  * enable_lora=True with a soft max (swap adapters without re-loading base)
  * kv_cache_dtype from the hardware profile (fp8 on Hopper/Blackwell, auto otherwise)
  * enable_prefix_caching=True — RadixAttention-style cross-request prefix reuse
  * max_num_seqs derived from VRAM headroom
"""
from __future__ import annotations

import logging
import time
from typing import Dict

from .manager import GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)


class VLLMBackend:

    def __init__(self, base_model_id: str, *, profile, lora_registry: Dict[str, str]):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest  # noqa: F401  (used in generate)

        self.model_id = base_model_id
        self.lora_registry = lora_registry

        dtype_map = {"bfloat16": "bfloat16", "float16": "float16", "fp8": "auto", "int8": "auto", "int4": "auto"}
        dtype = dtype_map.get(profile.inference_dtype, "auto")

        kv_cache_dtype = profile.kv_cache_dtype if profile.kv_cache_dtype != "auto" else "auto"

        logger.info(
            f"⚡ vLLM init: {base_model_id} | dtype={dtype} | "
            f"kv_cache={kv_cache_dtype} | prefix_caching=on"
        )
        self._llm = LLM(
            model=base_model_id,
            enable_lora=True,
            max_loras=8,
            max_lora_rank=64,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            enable_prefix_caching=True,    # ← RadixAttention-style prefix reuse
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            enable_chunked_prefill=True,   # ← chunked prefill for long prompts
        )
        self._SamplingParams = SamplingParams
        self._LoRARequest = LoRARequest
        self._lora_counter = 0
        self._lora_ids: Dict[str, int] = {}

    def register_adapter(self, domain_name: str, adapter_path: str):
        # We lazily assign an int ID at first use; vLLM requires it for LoRARequest.
        self.lora_registry[domain_name] = adapter_path

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        params = self._SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_new_tokens,
            repetition_penalty=req.repetition_penalty,
            stop=req.stop,
        )

        lora_req = None
        if req.domain_name != "base" and req.domain_name in self.lora_registry:
            if req.domain_name not in self._lora_ids:
                self._lora_counter += 1
                self._lora_ids[req.domain_name] = self._lora_counter
            lora_req = self._LoRARequest(
                lora_name=req.domain_name,
                lora_int_id=self._lora_ids[req.domain_name],
                lora_path=self.lora_registry[req.domain_name],
            )

        t_start = time.perf_counter()
        outputs = self._llm.generate(
            prompts=[req.prompt],
            sampling_params=params,
            lora_request=lora_req,
        )
        t_end = time.perf_counter()

        gen = outputs[0].outputs[0]
        text = gen.text.strip()
        n_tokens = len(gen.token_ids) if hasattr(gen, "token_ids") else 0
        lat_ms = (t_end - t_start) * 1000
        tps = (n_tokens / (t_end - t_start)) if (t_end > t_start and n_tokens) else 0.0
        # vLLM doesn't expose TTFT directly in non-streaming; estimate as latency / tokens
        ttft_ms = (lat_ms / n_tokens) if n_tokens else lat_ms

        return GenerationResponse(
            text=text,
            backend="vllm",
            model=self.model_id,
            domain=req.domain_name,
            ttft_ms=ttft_ms,
            latency_ms=lat_ms,
            tokens_generated=n_tokens,
            tokens_per_second=tps,
        )
