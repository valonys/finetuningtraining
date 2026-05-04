"""Embedding interfaces and a deterministic local fallback."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol


class Embedder(Protocol):
    """Provider interface for embedding text into a shared vector space."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicHashEmbedder:
    """Dependency-free embedding fallback for dev, tests, and CI.

    This is not intended to beat a real model. It gives stable cosine-search
    behavior without network calls so the pipeline remains pluggable and easy
    to validate before wiring Snowflake Cortex, OpenAI, Bedrock, or local
    sentence-transformers.
    """

    def __init__(self, dim: int = 384):
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        tokens = _tokens(text)
        counts = Counter(tokens)
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign * (1.0 + math.log(count))
        return _normalize(vector)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    return sum(x * y for x, y in zip(a, b))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]
