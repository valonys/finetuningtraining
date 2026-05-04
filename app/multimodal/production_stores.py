"""Optional production vector-store adapters.

These classes keep the same interface as `SQLiteVectorStore` but depend on
external infrastructure. They are intentionally imported only when selected by
deployment code so local dev and tests stay dependency-light.
"""
from __future__ import annotations

import json
from typing import Any

from .schemas import ContentChunk, Modality, RetrievalResult
from .vector_store import _source_from_dict, _source_to_dict


class PostgresPGVectorStore:
    """pgvector-backed store using a DB-API/psycopg connection."""

    def __init__(self, conn):
        self.conn = conn
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS multimodal_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    source_json JSONB NOT NULL,
                    metadata_json JSONB NOT NULL,
                    vector vector NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_mm_chunks_scope "
                "ON multimodal_chunks(tenant_id, collection, source_type)"
            )
        self.conn.commit()

    def upsert(self, chunks: list[ContentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        with self.conn.cursor() as cur:
            for chunk, vector in zip(chunks, vectors):
                cur.execute(
                    """
                    INSERT INTO multimodal_chunks (
                        chunk_id, tenant_id, collection, source_type, record_id,
                        chunk_index, text, source_json, metadata_json, vector
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        tenant_id=excluded.tenant_id,
                        collection=excluded.collection,
                        source_type=excluded.source_type,
                        record_id=excluded.record_id,
                        chunk_index=excluded.chunk_index,
                        text=excluded.text,
                        source_json=excluded.source_json,
                        metadata_json=excluded.metadata_json,
                        vector=excluded.vector
                    """,
                    (
                        chunk.chunk_id,
                        chunk.source.tenant_id,
                        chunk.source.collection,
                        chunk.source.source_type.value,
                        chunk.record_id,
                        chunk.chunk_index,
                        chunk.text,
                        json.dumps(_source_to_dict(chunk.source)),
                        json.dumps(chunk.metadata, default=str),
                        _vector_literal(vector),
                    ),
                )
        self.conn.commit()

    def search(
        self,
        query_vector: list[float],
        *,
        tenant_id: str,
        collection: str,
        top_k: int,
        source_type: Modality | None = None,
    ) -> list[RetrievalResult]:
        where = "tenant_id = %s AND collection = %s"
        params: list[Any] = [tenant_id, collection]
        if source_type:
            where += " AND source_type = %s"
            params.append(source_type.value)
        query_literal = _vector_literal(query_vector)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT *, 1 - (vector <=> %s::vector) AS score
                FROM multimodal_chunks
                WHERE {where}
                ORDER BY vector <=> %s::vector
                LIMIT %s
                """,
                [query_literal, *params, query_literal, top_k],
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
        return [_pg_row_to_result(_row_to_mapping(row, columns)) for row in rows]

    def stats(self, *, tenant_id: str, collection: str) -> dict:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_type, COUNT(*)
                FROM multimodal_chunks
                WHERE tenant_id = %s AND collection = %s
                GROUP BY source_type
                """,
                (tenant_id, collection),
            )
            rows = cur.fetchall()
        by_modality = {r[0]: int(r[1]) for r in rows}
        return {
            "tenant_id": tenant_id,
            "collection": collection,
            "chunk_count": sum(by_modality.values()),
            "by_modality": by_modality,
        }


class SnowflakeVectorStore:
    """Snowflake VECTOR table adapter.

    This adapter is a production integration target. Use it with
    `SnowflakeCortexEmbedder` so ingestion and search stay inside Snowflake.
    """

    def __init__(self, session, *, table_name: str = "MULTIMODAL_CHUNKS"):
        self.session = session
        self.table_name = table_name

    def upsert(self, chunks: list[ContentChunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError(
            "SnowflakeVectorStore requires deployment-specific staging/merge SQL. "
            "Use the VectorStore protocol and docs/MULTIMODAL_PIPELINE.md as the contract."
        )

    def search(self, query_vector: list[float], *, tenant_id: str, collection: str, top_k: int, source_type: Modality | None = None) -> list[RetrievalResult]:
        raise NotImplementedError("SnowflakeVectorStore search SQL is deployment-specific.")

    def stats(self, *, tenant_id: str, collection: str) -> dict:
        rows = self.session.sql(
            f"""
            SELECT source_type, COUNT(*) AS n
            FROM {self.table_name}
            WHERE tenant_id = '{tenant_id}' AND collection = '{collection}'
            GROUP BY source_type
            """
        ).collect()
        by_modality = {r["SOURCE_TYPE"]: int(r["N"]) for r in rows}
        return {
            "tenant_id": tenant_id,
            "collection": collection,
            "chunk_count": sum(by_modality.values()),
            "by_modality": by_modality,
        }


def _pg_row_to_result(row: dict[str, Any]) -> RetrievalResult:
    source = row["source_json"]
    metadata = row["metadata_json"]
    if isinstance(source, str):
        source = json.loads(source)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    chunk = ContentChunk(
        chunk_id=row["chunk_id"],
        record_id=row["record_id"],
        text=row["text"],
        source=_source_from_dict(source),
        chunk_index=int(row["chunk_index"]),
        metadata=metadata,
    )
    return RetrievalResult(chunk=chunk, score=float(row["score"]))


def _row_to_mapping(row, columns: list[str]) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    return dict(zip(columns, row))


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.12g}" for v in vector) + "]"
