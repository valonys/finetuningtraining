"""End-to-end multimodal indexing pipeline."""
from __future__ import annotations

from collections.abc import Iterable

from .chunking import chunk_records
from .embeddings import DeterministicHashEmbedder, Embedder
from .schemas import ContentChunk, ContentRecord, Modality, PipelineConfig, RetrievalResult
from .vector_store import SQLiteVectorStore, VectorStore


class MultimodalPipeline:
    """Normalize, chunk, embed, index, and retrieve multimodal content."""

    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.config = config or PipelineConfig()
        self.config.validate()
        self.embedder = embedder or DeterministicHashEmbedder(dim=self.config.embedding_dim)
        self.vector_store = vector_store or SQLiteVectorStore()

    def index_records(self, records: Iterable[ContentRecord]) -> list[ContentChunk]:
        chunks = chunk_records(
            records,
            target_chars=self.config.chunk_target_chars,
            overlap_chars=self.config.chunk_overlap_chars,
        )
        if not chunks:
            return []
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        self.vector_store.upsert(chunks, vectors)
        return chunks

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source_type: Modality | None = None,
        tenant_id: str | None = None,
        collection: str | None = None,
    ) -> list[RetrievalResult]:
        query_vector = self.embedder.embed([query])[0]
        return self.vector_store.search(
            query_vector,
            tenant_id=tenant_id or self.config.tenant_id,
            collection=collection or self.config.collection,
            top_k=top_k or self.config.default_top_k,
            source_type=source_type,
        )
