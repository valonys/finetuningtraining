"""
Ollama inference backend — serves chat generation through a local Ollama
daemon or Ollama Cloud using the same OpenAI-compatible wire format as the
synth provider (`app/providers/ollama.py`). In fact the two share that
client class — this backend is a thin wrapper that:

  1. Streams tokens via `OllamaProvider.stream_chat()` so we can measure
     p50/p90/p99 TTFT the same way the other backends do.
  2. Implements the `register_adapter(name, path) / generate(req)` shape
     the inference manager expects.

────────────────────────────────────────────────────────────────────────
IMPORTANT: LoRA semantics are different on Ollama
────────────────────────────────────────────────────────────────────────
Ollama does NOT hot-swap HuggingFace-format LoRA adapters on top of a
shared base model the way vLLM / SGLang / MLX / llama.cpp / HF do. Ollama
serves pre-packaged models from its own catalog (native or imported
GGUF). To use a LoRA trained in this Studio with Ollama you have to:

    (a) Merge the adapter into the base weights
    (b) Convert the merged model to GGUF
    (c) `ollama create mymodel -f Modelfile` to import it
    (d) Serve with `ollama run mymodel` or via the daemon

None of that happens in this backend. What IS supported is **model
routing by domain name**: you tell the backend which Ollama model tag to
use for each domain, and the `domain_name` argument to `generate()`
selects between them. That gives you multi-domain serving without
touching LoRA at all — you just pre-pull the Ollama models you need.

Example::

    engine = get_inference_engine("llama3.1", backend="ollama")
    engine.register_adapter("customer_grasps", "llama3.1")     # friendly tone
    engine.register_adapter("asset_integrity", "qwen2.5:7b")  # technical
    engine.register_adapter("ai_llm",          "nemotron")    # Ollama Cloud
    engine.generate(prompt, domain_name="ai_llm")             # → nemotron

If a request arrives with a `domain_name` that isn't registered (or the
literal string `"base"`), the backend falls back to the base model
passed in the constructor.
────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List

from .manager import GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Best-effort HF-id → Ollama-tag hints
# ──────────────────────────────────────────────────────────────
_HF_TO_OLLAMA_HINTS: Dict[str, str] = {
    # Qwen
    "qwen/qwen2.5-0.5b-instruct":  "qwen2.5:0.5b",
    "qwen/qwen2.5-1.5b-instruct":  "qwen2.5:1.5b",
    "qwen/qwen2.5-3b-instruct":    "qwen2.5:3b",
    "qwen/qwen2.5-7b-instruct":    "qwen2.5:7b",
    "qwen/qwen2.5-14b-instruct":   "qwen2.5:14b",
    "qwen/qwen2.5-32b-instruct":   "qwen2.5:32b",
    "qwen/qwen2.5-72b-instruct":   "qwen2.5:72b",
    # Llama
    "meta-llama/llama-3.2-1b-instruct":       "llama3.2:1b",
    "meta-llama/llama-3.2-3b-instruct":       "llama3.2:3b",
    "meta-llama/meta-llama-3.1-8b-instruct":  "llama3.1:8b",
    "meta-llama/llama-3.3-70b-instruct":      "llama3.3:70b",
    # Mistral
    "mistralai/mistral-7b-instruct-v0.3":     "mistral:7b",
    "mistralai/mistral-nemo-instruct-2407":   "mistral-nemo",
    # Gemma
    "google/gemma-2-2b-it":                   "gemma2:2b",
    "google/gemma-2-9b-it":                   "gemma2:9b",
    # Phi
    "microsoft/phi-3.5-mini-instruct":        "phi3.5:latest",
    "microsoft/phi-4":                        "phi4",
    # DeepSeek
    "deepseek-ai/deepseek-r1-distill-qwen-7b":  "deepseek-r1:7b",
    "deepseek-ai/deepseek-r1-distill-llama-8b": "deepseek-r1:8b",
    # NVIDIA Nemotron on Ollama Cloud
    "nvidia/llama-3.1-nemotron-70b-instruct": "nemotron",
}


def resolve_ollama_model(model_id: str) -> str:
    """
    Best-effort map a HuggingFace-style id to an Ollama tag.

    If the input is already a tag (no `/`), pass it through unchanged.
    If the input matches a known HF id, return the mapped Ollama tag.
    Otherwise, return the org-stripped lowercase name as a guess.
    """
    if "/" not in model_id:
        return model_id
    lower = model_id.lower()
    if lower in _HF_TO_OLLAMA_HINTS:
        return _HF_TO_OLLAMA_HINTS[lower]
    # Fallback: strip org prefix, lowercase, drop -instruct/-chat suffixes
    name = model_id.split("/", 1)[1].lower()
    for suffix in ("-instruct", "-chat", "-it"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


# ──────────────────────────────────────────────────────────────
# Backend
# ──────────────────────────────────────────────────────────────
class OllamaInferenceBackend:
    """Serves generation via Ollama (local daemon or Ollama Cloud)."""

    def __init__(self, base_model_id: str, *, profile, lora_registry: Dict[str, str]):
        from app.providers.ollama import OllamaProvider

        if "/" in base_model_id:
            resolved = resolve_ollama_model(base_model_id)
            logger.warning(
                f"⚠️  Ollama backend received HF-style id '{base_model_id}'. "
                f"Ollama uses its own tags — mapped to '{resolved}'. "
                f"To be explicit, set VALONY_BASE_MODEL to an Ollama tag "
                f"(e.g. 'llama3.1', 'nemotron', 'qwen2.5:7b')."
            )
            base_model_id = resolved

        self.model_id = base_model_id
        # The inference manager passes in an empty dict which we mutate — the
        # router keeps a reference so register_adapter() is visible here too.
        self.lora_registry = lora_registry

        # Reuse the synth provider class for the wire layer — same URL
        # resolution, same auth handling, one code path to maintain.
        self._provider = OllamaProvider(model=base_model_id)
        self._base_provider_model = self._provider.model

        logger.info(
            f"🦙 Ollama inference backend → {self._provider.base_url} | "
            f"base={self._provider.model} | "
            f"mode={'cloud' if self._provider.is_cloud else 'local'}"
        )

    # ── Public API ────────────────────────────────────────────
    def register_adapter(self, domain_name: str, adapter_path: str) -> None:
        """
        On Ollama, `adapter_path` is interpreted as an Ollama model tag
        (not a filesystem path to a LoRA). See the module docstring for
        the full story.
        """
        resolved = resolve_ollama_model(adapter_path)
        if resolved != adapter_path:
            logger.info(f"🦙 '{adapter_path}' → Ollama tag '{resolved}' for domain {domain_name}")
        self.lora_registry[domain_name] = resolved

    def stream(self, req: GenerationRequest):
        """
        Yield response deltas as strings, live from Ollama's SSE stream.

        Used by `InferenceManager.generate_stream()` → FastAPI's
        `/v1/chat/stream` endpoint for Server-Sent Events so the UI shows
        the typewriter effect as Nemotron emits tokens, rather than a
        big reveal after generation completes.
        """
        model = self.lora_registry.get(req.domain_name) if req.domain_name != "base" else None
        model = model or self._base_provider_model

        original_model = self._provider.model
        self._provider.model = model
        try:
            for delta in self._provider.stream_chat(
                messages=[{"role": "user", "content": req.prompt}],
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_new_tokens,
                stop=req.stop,
            ):
                yield delta
        finally:
            self._provider.model = original_model

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        # Select the model: domain mapping takes precedence, else the base
        model = self.lora_registry.get(req.domain_name) if req.domain_name != "base" else None
        model = model or self._base_provider_model

        # Temporarily swap the provider's model for this request
        original_model = self._provider.model
        self._provider.model = model

        messages = [{"role": "user", "content": req.prompt}]

        t_start = time.perf_counter()
        ttft_ms: float = 0.0
        n_tokens = 0
        chunks: List[str] = []

        try:
            for delta in self._provider.stream_chat(
                messages=messages,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_new_tokens,
                stop=req.stop,
            ):
                if n_tokens == 0:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                chunks.append(delta)
                n_tokens += 1
        finally:
            self._provider.model = original_model

        t_end = time.perf_counter()
        lat_ms = (t_end - t_start) * 1000
        tps = (n_tokens / (t_end - t_start)) if (t_end > t_start and n_tokens) else 0.0

        return GenerationResponse(
            text="".join(chunks).strip(),
            backend="ollama",
            model=model,
            domain=req.domain_name,
            ttft_ms=ttft_ms or (lat_ms / max(n_tokens, 1)),
            latency_ms=lat_ms,
            tokens_generated=n_tokens,
            tokens_per_second=tps,
        )
