"""Provider-neutral multimodal ingestion, retrieval, and RAG pipeline."""
from .embeddings import DeterministicHashEmbedder, Embedder
from .pipeline import MultimodalPipeline
from .rag import ContextBuilder, RAGAnswer, RAGEngine
from .schemas import (
    ContentChunk,
    ContentRecord,
    Modality,
    PipelineConfig,
    RetrievalResult,
    SourceRef,
)
from .vector_store import SQLiteVectorStore, VectorStore

__all__ = [
    "ContentChunk",
    "ContentRecord",
    "ContextBuilder",
    "DeterministicHashEmbedder",
    "Embedder",
    "Modality",
    "MultimodalPipeline",
    "PipelineConfig",
    "RAGAnswer",
    "RAGEngine",
    "RetrievalResult",
    "SQLiteVectorStore",
    "SourceRef",
    "VectorStore",
]
