"""
PrefixCache — cross-request prefix matching via a radix trie.

Inspired by SGLang's RadixAttention: when a new request arrives, we walk the
trie to find the longest cached prompt prefix, then the HF backend skips
recomputing that prefix (via `past_key_values`). This is a best-effort speedup
for the HF backend — on vLLM/SGLang the engine handles it natively.

The data structure stores:
    prefix token-ID sequence → {"kv": KVCache, "hits": int, "last_used": float}

Eviction: LRU by `last_used`, capped at `max_entries`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .kv_cache import KVCache


@dataclass
class _TrieNode:
    children: Dict[int, "_TrieNode"] = field(default_factory=dict)
    kv: Optional[KVCache] = None
    last_used: float = 0.0
    hits: int = 0


class PrefixCache:

    def __init__(self, max_entries: int = 64):
        self.root = _TrieNode()
        self.max_entries = max_entries
        self._entries: List[_TrieNode] = []

    def put(self, token_ids: List[int], kv: KVCache) -> None:
        node = self.root
        for tid in token_ids:
            node = node.children.setdefault(tid, _TrieNode())
        node.kv = kv
        node.last_used = time.time()
        if node not in self._entries:
            self._entries.append(node)
        self._evict()

    def get_longest(self, token_ids: List[int]) -> tuple[int, Optional[KVCache]]:
        """Return (prefix_len, kv) for the longest cached prefix, or (0, None)."""
        node = self.root
        best_len = 0
        best_kv: Optional[KVCache] = None
        for i, tid in enumerate(token_ids):
            if tid not in node.children:
                break
            node = node.children[tid]
            if node.kv is not None:
                best_len = i + 1
                best_kv = node.kv
                node.last_used = time.time()
                node.hits += 1
        return best_len, best_kv

    def _evict(self):
        if len(self._entries) <= self.max_entries:
            return
        self._entries.sort(key=lambda n: n.last_used)
        victims = self._entries[: len(self._entries) - self.max_entries]
        for v in victims:
            v.kv = None
        self._entries = [e for e in self._entries if e.kv is not None]
