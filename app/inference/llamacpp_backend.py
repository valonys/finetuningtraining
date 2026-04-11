"""
llama.cpp backend — GGUF runtime for CPU, Metal, or CUDA without vLLM.

The user is expected to pass a GGUF path (either a local file or a
`repo_id:filename` tuple) as `base_model_id`. Adapter swap uses llama.cpp's
`set_lora` if the python bindings support it; otherwise we no-op with a
warning.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

from .manager import GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)


class LlamaCppBackend:

    def __init__(self, base_model_id: str, *, profile, lora_registry: Dict[str, str]):
        from llama_cpp import Llama

        self.model_id = base_model_id
        self.lora_registry = lora_registry

        n_gpu_layers = 0
        if profile.inference_backend == "llamacpp" and profile.torch_dtype != "float32":
            # Offload everything to GPU/Metal when we have accelerators
            n_gpu_layers = -1

        logger.info(f"🦙 llama.cpp load: {base_model_id} | n_gpu_layers={n_gpu_layers}")
        self._llm = Llama(
            model_path=base_model_id,
            n_ctx=profile.max_seq_length,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            verbose=False,
        )

    def register_adapter(self, domain_name: str, adapter_path: str):
        """Attach a LoRA adapter if the bindings support it."""
        try:
            self._llm.set_lora(adapter_path)
            self.lora_registry[domain_name] = adapter_path
        except Exception as e:
            logger.warning(f"⚠️  llama.cpp adapter load not supported: {e}")

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        t_start = time.perf_counter()
        ttft_ms = 0.0
        first = True
        parts: list[str] = []
        n_tokens = 0

        for chunk in self._llm(
            req.prompt,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop or [],
            stream=True,
            repeat_penalty=req.repetition_penalty,
        ):
            if first:
                ttft_ms = (time.perf_counter() - t_start) * 1000
                first = False
            parts.append(chunk["choices"][0]["text"])
            n_tokens += 1

        t_end = time.perf_counter()
        lat_ms = (t_end - t_start) * 1000
        tps = n_tokens / (t_end - t_start) if t_end > t_start and n_tokens else 0.0

        return GenerationResponse(
            text="".join(parts).strip(),
            backend="llamacpp",
            model=self.model_id,
            domain=req.domain_name,
            ttft_ms=ttft_ms or (lat_ms / max(n_tokens, 1)),
            latency_ms=lat_ms,
            tokens_generated=n_tokens,
            tokens_per_second=tps,
        )
