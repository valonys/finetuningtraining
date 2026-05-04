"""SQLite vector store for portable multimodal retrieval."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Protocol

from .embeddings import cosine
from .schemas import ContentChunk, Modality, RetrievalResult, SourceRef


class VectorStore(Protocol):
    def upsert(self, chunks: list[ContentChunk], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        *,
        tenant_id: str,
        collection: str,
        top_k: int,
        source_type: Modality | None = None,
    ) -> list[RetrievalResult]: ...

    def stats(self, *, tenant_id: str, collection: str) -> dict: ...


class SQLiteVectorStore:
    """Durable local vector index.

    This is intentionally simple: vectors are stored as JSON and scored in
    Python. For high-volume production, keep this interface and swap in
    pgvector, Snowflake VECTOR, OpenSearch, Pinecone, Weaviate, or Milvus.
    """

    def __init__(self, db_path: str | Path = "outputs/.multimodal/index.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id     TEXT PRIMARY KEY,
                    tenant_id    TEXT NOT NULL,
                    collection   TEXT NOT NULL,
                    source_type  TEXT NOT NULL,
                    record_id    TEXT NOT NULL,
                    chunk_index  INTEGER NOT NULL,
                    text         TEXT NOT NULL,
                    source_json  TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    vector_json  TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_scope "
                "ON chunks(tenant_id, collection, source_type)"
            )
            self._conn.commit()

    def upsert(self, chunks: list[ContentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        rows = []
        for chunk, vector in zip(chunks, vectors):
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.source.tenant_id,
                    chunk.source.collection,
                    chunk.source.source_type.value,
                    chunk.record_id,
                    chunk.chunk_index,
                    chunk.text,
                    json.dumps(_source_to_dict(chunk.source), sort_keys=True),
                    json.dumps(chunk.metadata, sort_keys=True, default=str),
                    json.dumps(vector),
                )
            )
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, tenant_id, collection, source_type, record_id,
                    chunk_index, text, source_json, metadata_json, vector_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    tenant_id=excluded.tenant_id,
                    collection=excluded.collection,
                    source_type=excluded.source_type,
                    record_id=excluded.record_id,
                    chunk_index=excluded.chunk_index,
                    text=excluded.text,
                    source_json=excluded.source_json,
                    metadata_json=excluded.metadata_json,
                    vector_json=excluded.vector_json
                """,
                rows,
            )
            self._conn.commit()

    def search(
        self,
        query_vector: list[float],
        *,
        tenant_id: str,
        collection: str,
        top_k: int,
        source_type: Modality | None = None,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        sql = (
            "SELECT * FROM chunks WHERE tenant_id = ? AND collection = ?"
            + (" AND source_type = ?" if source_type else "")
        )
        params: tuple[str, ...] = (
            (tenant_id, collection, source_type.value) if source_type else (tenant_id, collection)
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        scored: list[RetrievalResult] = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            score = cosine(query_vector, vector)
            scored.append(RetrievalResult(chunk=_row_to_chunk(row), score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def stats(self, *, tenant_id: str, collection: str) -> dict:
        """Return lightweight collection stats for API/UI surfaces."""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE tenant_id = ? AND collection = ?",
                (tenant_id, collection),
            ).fetchone()["n"]
            rows = self._conn.execute(
                """
                SELECT source_type, COUNT(*) AS n
                FROM chunks
                WHERE tenant_id = ? AND collection = ?
                GROUP BY source_type
                ORDER BY source_type
                """,
                (tenant_id, collection),
            ).fetchall()
        return {
            "tenant_id": tenant_id,
            "collection": collection,
            "chunk_count": int(total),
            "by_modality": {row["source_type"]: int(row["n"]) for row in rows},
        }


def _source_to_dict(source: SourceRef) -> dict:
    return {
        "source_uri": source.source_uri,
        "source_type": source.source_type.value,
        "tenant_id": source.tenant_id,
        "collection": source.collection,
        "title": source.title,
        "start_time_s": source.start_time_s,
        "end_time_s": source.end_time_s,
        "page": source.page,
        "frame": source.frame,
        "extra": source.extra,
    }


def _source_from_dict(data: dict) -> SourceRef:
    return SourceRef(
        source_uri=data["source_uri"],
        source_type=Modality(data["source_type"]),
        tenant_id=data.get("tenant_id", "public"),
        collection=data.get("collection", "default"),
        title=data.get("title"),
        start_time_s=data.get("start_time_s"),
        end_time_s=data.get("end_time_s"),
        page=data.get("page"),
        frame=data.get("frame"),
        extra=data.get("extra") or {},
    )


def _row_to_chunk(row: sqlite3.Row) -> ContentChunk:
    return ContentChunk(
        chunk_id=row["chunk_id"],
        record_id=row["record_id"],
        text=row["text"],
        source=_source_from_dict(json.loads(row["source_json"])),
        chunk_index=row["chunk_index"],
        metadata=json.loads(row["metadata_json"]),
    )
