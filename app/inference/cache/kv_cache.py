"""
Thin wrapper around HF's per-layer KV cache tensors, used by the HF backend
when we want to reuse prior context across turns of a multi-turn chat.

Not used by vLLM / SGLang / MLX (they manage their own KV cache internally).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class KVCache:
    key_values: Any = None                  # HF `past_key_values` tuple
    cached_prompt_ids: list[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.key_values = None
        self.cached_prompt_ids = []
        self.metadata.clear()

    def is_empty(self) -> bool:
        return self.key_values is None
