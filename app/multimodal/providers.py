"""Production provider adapters for the multimodal pipeline."""
from __future__ import annotations

import os
from typing import Any

from .embeddings import DeterministicHashEmbedder


class OpenAICompatEmbedder:
    """Embedding adapter for OpenAI-compatible `/v1/embeddings` APIs."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VALONY_EMBED_API_KEY")
        self.base_url = (base_url or os.environ.get("VALONY_EMBED_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("VALONY_EMBED_MODEL") or "text-embedding-3-small"
        self._dim = dim or int(os.environ.get("VALONY_EMBED_DIM", "1536"))
        if not self.api_key:
            raise ValueError("OpenAI-compatible embedder requires an API key")

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import requests

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"embedding provider rejected request: {response.text[:500]}")
        data = response.json()
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


class SnowflakeCortexEmbedder:
    """Snowflake Cortex embedding adapter.

    Pass an active Snowpark session. This class deliberately does not create
    the connection because enterprises typically inject governed sessions.
    """

    def __init__(self, session: Any, *, model: str = "snowflake-arctic-embed-m", dim: int = 768):
        self.session = session
        self.model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            escaped = text.replace("'", "''")
            model = self.model.replace("'", "''")
            row = self.session.sql(
                f"SELECT AI_EMBED('{model}', '{escaped}') AS embedding"
            ).collect()[0]
            vectors.append(list(row["EMBEDDING"]))
        return vectors


def resolve_embedder(provider: str | None, *, dim: int):
    """Build an embedder from env-friendly provider names."""
    name = (provider or os.environ.get("VALONY_MM_EMBED_PROVIDER") or "hash").lower()
    if name in {"hash", "local", "deterministic"}:
        return DeterministicHashEmbedder(dim=dim)
    if name in {"openai", "openai_compat", "compatible"}:
        return OpenAICompatEmbedder(dim=dim)
    raise ValueError(
        f"unsupported multimodal embed provider '{name}'. "
        "Use 'hash' or 'openai_compat', or inject a custom Embedder."
    )
