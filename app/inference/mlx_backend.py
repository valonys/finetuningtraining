"""
MLX-LM backend — native Apple Silicon inference.

Notes:
  * MLX uses 4-bit / 8-bit quantization via `mlx_lm.convert`. The user should
    point `base_model_id` at an `mlx-community/*-4bit` repo for best latency.
  * LoRA adapters are merged into the base model at registration time — MLX-LM
    does not support runtime LoRA swap.
  * `mlx_lm.generate()` returns the full string in one go; we measure wall-clock
    TTFT via the `stream_generate()` helper for accurate telemetry.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

from .manager import GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)


class MLXBackend:

    def __init__(self, base_model_id: str, *, profile, lora_registry: Dict[str, str]):
        from mlx_lm import load

        self.model_id = base_model_id
        self.lora_registry = lora_registry
        self._active_domain: str = "base"

        logger.info(f"🍎 MLX load: {base_model_id}")
        self._model, self._tokenizer = load(base_model_id)

    def register_adapter(self, domain_name: str, adapter_path: str):
        """Merge an MLX adapter into the active model."""
        try:
            from mlx_lm import load as _load
            logger.info(f"🍎 Loading MLX adapter {domain_name}")
            self._model, self._tokenizer = _load(self.model_id, adapter_path=adapter_path)
            self._active_domain = domain_name
        except Exception as e:
            logger.warning(f"⚠️  MLX adapter load failed: {e}")
        self.lora_registry[domain_name] = adapter_path

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        # Swap adapter lazily
        if req.domain_name != "base" and req.domain_name != self._active_domain:
            if req.domain_name in self.lora_registry:
                self.register_adapter(req.domain_name, self.lora_registry[req.domain_name])

        try:
            from mlx_lm import stream_generate
        except ImportError:
            stream_generate = None

        t_start = time.perf_counter()
        ttft_ms = 0.0
        n_tokens = 0
        out_parts: list[str] = []

        if stream_generate is not None:
            for token in stream_generate(
                self._model,
                self._tokenizer,
                prompt=req.prompt,
                max_tokens=req.max_new_tokens,
                temp=req.temperature,
            ):
                if n_tokens == 0:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                out_parts.append(token if isinstance(token, str) else token.text)
                n_tokens += 1
        else:
            from mlx_lm import generate
            text = generate(
                self._model,
                self._tokenizer,
                prompt=req.prompt,
                max_tokens=req.max_new_tokens,
                temp=req.temperature,
            )
            out_parts.append(text)
            n_tokens = len(text.split())

        t_end = time.perf_counter()
        lat_ms = (t_end - t_start) * 1000
        tps = n_tokens / (t_end - t_start) if t_end > t_start and n_tokens else 0.0

        return GenerationResponse(
            text="".join(out_parts).strip(),
            backend="mlx",
            model=self.model_id,
            domain=req.domain_name,
            ttft_ms=ttft_ms or lat_ms / max(n_tokens, 1),
            latency_ms=lat_ms,
            tokens_generated=n_tokens,
            tokens_per_second=tps,
        )
