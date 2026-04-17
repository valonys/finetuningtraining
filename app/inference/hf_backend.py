"""
HuggingFace Transformers backend — always-available fallback.

Features:
  * 4-bit / bf16 loading via the hardware profile
  * LoRA hot-swap via `PeftModel.from_pretrained`
  * Streaming token output via `TextIteratorStreamer` to measure TTFT
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

from .manager import GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)


def _patch_isin_mps_friendly():
    """Monkey-patch transformers 4.45.x bug in `isin_mps_friendly`.

    The function does `test_elements.shape[0]` which crashes with
    `IndexError: tuple index out of range` when the input is a 0-dim
    tensor. This happens when `pad_token_id` is a scalar int
    (which torch.tensor() converts to a 0-dim tensor).

    Fixed in transformers 4.46+ but we're pinned to 4.45 for torch
    2.2.2 compatibility. The patch is safe for all versions — if the
    original works, it's never called.
    """
    try:
        import transformers.pytorch_utils as _pu
        _orig = _pu.isin_mps_friendly
        def _safe_isin(elements, test_elements):
            if test_elements.dim() == 0:
                test_elements = test_elements.unsqueeze(0)
            if elements.dim() == 0:
                elements = elements.unsqueeze(0)
            return _orig(elements, test_elements)
        _pu.isin_mps_friendly = _safe_isin
    except Exception:
        pass

_patch_isin_mps_friendly()


class HFBackend:

    def __init__(self, base_model_id: str, *, profile, lora_registry: Dict[str, str]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = base_model_id
        self.lora_registry = lora_registry
        self._active_domain: str = "base"
        self._peft_model = None

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype.get(profile.torch_dtype, torch.float16)

        quant = None
        if profile.load_in_4bit:
            try:
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                quant = None

        device_map = "auto"
        if profile.training_backend == "mlx":
            device_map = {"": "mps"}

        logger.info(f"🤗 HF load: {base_model_id} | 4bit={bool(quant)} | dtype={torch_dtype}")
        self._tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quant,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self._base_model.eval()

    def register_adapter(self, domain_name: str, adapter_path: str):
        self.lora_registry[domain_name] = adapter_path

    def _ensure_adapter(self, domain_name: str):
        if domain_name == "base":
            self._peft_model = None
            self._active_domain = "base"
            return
        if self._active_domain == domain_name:
            return
        from peft import PeftModel
        adapter_path = self.lora_registry.get(domain_name)
        if not adapter_path:
            raise ValueError(f"HF: adapter '{domain_name}' not registered")
        logger.info(f"🔄 HF adapter swap → {domain_name}")
        self._peft_model = PeftModel.from_pretrained(self._base_model, adapter_path)
        self._peft_model.eval()
        self._active_domain = domain_name

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        import torch
        from transformers import TextIteratorStreamer

        self._ensure_adapter(req.domain_name)
        model = self._peft_model or self._base_model
        device = next(model.parameters()).device

        inputs = self._tokenizer(req.prompt, return_tensors="pt").to(device)
        # CPU inference on an Intel i7 can take 90-120s for the first
        # token on a 0.5B model — keep the timeout generous so we don't
        # time out during legitimate (but slow) CPU generation.
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300
        )

        # Ensure pad_token_id is set on the model config rather than
        # passed as a generate() kwarg — transformers 4.45's
        # _prepare_special_tokens has a bug where a scalar pad_token_id
        # tensor causes `isin_mps_friendly` to crash with IndexError.
        if hasattr(model, "generation_config"):
            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = self._tokenizer.eos_token_id
        elif hasattr(model, "config"):
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = self._tokenizer.eos_token_id

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            repetition_penalty=req.repetition_penalty,
        )

        # Wrap generate in a closure that surfaces exceptions — threads
        # swallow them by default, causing the streamer to timeout with
        # no indication of what went wrong.
        gen_error: list = []   # mutable so the thread can write to it
        def _generate():
            try:
                model.generate(**gen_kwargs)
            except Exception as e:
                import logging
                logging.getLogger(__name__).exception(f"⚠️  HF generate thread error: {e}")
                gen_error.append(e)
                streamer.text_queue.put(streamer.stop_signal)  # unblock the reader

        t_start = time.perf_counter()
        thread = threading.Thread(target=_generate)
        thread.start()

        ttft_ms: Optional[float] = None
        parts: list[str] = []
        n_tokens = 0
        for new_text in streamer:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            parts.append(new_text)
            n_tokens += 1

        thread.join()
        if gen_error:
            raise RuntimeError(f"HF generation failed: {gen_error[0]}") from gen_error[0]
        t_end = time.perf_counter()
        lat_ms = (t_end - t_start) * 1000
        tps = n_tokens / (t_end - t_start) if t_end > t_start and n_tokens else 0.0

        return GenerationResponse(
            text="".join(parts).strip(),
            backend="hf",
            model=self.model_id,
            domain=req.domain_name,
            ttft_ms=ttft_ms or lat_ms / max(n_tokens, 1),
            latency_ms=lat_ms,
            tokens_generated=n_tokens,
            tokens_per_second=tps,
        )

    # ── Streaming interface (SSE / /v1/chat/stream) ──────────
    def stream(self, req: GenerationRequest):
        """Yield text deltas as they arrive from the model.

        Uses the same TextIteratorStreamer as `generate()` but yields
        each chunk immediately so the SSE endpoint can push it to the
        browser for a typewriter effect.
        """
        import torch
        from transformers import TextIteratorStreamer

        self._ensure_adapter(req.domain_name)
        model = self._peft_model or self._base_model
        device = next(model.parameters()).device

        inputs = self._tokenizer(req.prompt, return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300
        )

        if hasattr(model, "generation_config"):
            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = self._tokenizer.eos_token_id
        elif hasattr(model, "config"):
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = self._tokenizer.eos_token_id

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            repetition_penalty=req.repetition_penalty,
        )

        gen_error: list = []
        def _generate():
            try:
                model.generate(**gen_kwargs)
            except Exception as e:
                logger.exception(f"⚠️  HF stream generate error: {e}")
                gen_error.append(e)
                streamer.text_queue.put(streamer.stop_signal)

        thread = threading.Thread(target=_generate)
        thread.start()

        for new_text in streamer:
            if new_text:
                yield new_text

        thread.join()
        if gen_error:
            raise RuntimeError(f"HF generation failed: {gen_error[0]}") from gen_error[0]
