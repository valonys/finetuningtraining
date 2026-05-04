# Enterprise Multimodal Pipeline

This module turns the Snowflake course pattern into a provider-neutral pipeline
that can plug into ValonyLabs Studio or another stack with minimal integration.

## Domain Pattern

The course material reduces to five production steps:

1. Extract text-bearing facts from every modality: ASR for audio, OCR for slides/images, VLM segment descriptions for video, and native parsing for documents.
2. Preserve source provenance: tenant, collection, URI, modality, page/frame/timestamp, and raw metadata.
3. Chunk the normalized text into retrieval-sized units.
4. Embed every chunk with the same embedding model so all modalities share one vector space.
5. Retrieve across modalities and build a cited RAG context with strict budget limits.

## Code Entry Points

- `app.multimodal.schemas`: stable content, chunk, source, retrieval, and config contracts.
- `app.multimodal.processors`: provider protocols for ASR/OCR/VLM plus a text-file processor.
- `app.multimodal.pipeline.MultimodalPipeline`: chunk, embed, index, and search.
- `app.multimodal.vector_store.SQLiteVectorStore`: local durable vector store. Swap behind the `VectorStore` protocol for Snowflake VECTOR, pgvector, OpenSearch, Milvus, Pinecone, or Weaviate.
- `app.multimodal.rag.RAGEngine`: retrieval + context assembly + generator callback.

## Minimal Integration

```python
from app.multimodal import MultimodalPipeline, PipelineConfig
from app.multimodal.processors import TextFileProcessor
from app.multimodal.schemas import Modality

pipeline = MultimodalPipeline(
    config=PipelineConfig(tenant_id="acme", collection="meeting-q2")
)

records = TextFileProcessor().process(
    "data/uploads/transcript.txt",
    tenant_id="acme",
    collection="meeting-q2",
    source_type=Modality.AUDIO,
)
pipeline.index_records(records)

results = pipeline.search("budget and remote control design", top_k=5)
```

## API Surface

- `POST /v1/multimodal/index`: validates allowed paths, parses them with Data Forge, normalizes records, chunks, embeds, and indexes.
- `POST /v1/multimodal/search`: semantic search across the tenant/collection, optionally filtered by modality.
- `POST /v1/multimodal/rag`: builds a cited RAG context and can optionally call the configured inference backend.
- `GET /v1/multimodal/{collection}/stats`: returns chunk counts by modality.

The default API path uses deterministic local embeddings and SQLite vector
storage. Set `embed_provider=openai_compat` with `VALONY_EMBED_API_KEY`,
`VALONY_EMBED_BASE_URL`, and `VALONY_EMBED_MODEL` to use a production embedding
gateway. For Snowflake Cortex or pgvector, inject the corresponding adapters
behind the same `Embedder` / `VectorStore` protocols.

## Course Demo

```bash
python scripts/multimodal_course_demo.py \
  --source-dir ~/Documents/"Building Multimodal Data Pipelines" \
  --collection multimodal-course
```

This indexes the course transcript files as normalized multimodal records and
runs a sample cross-modal query.

## Production Provider Swaps

- ASR: implement `AudioTranscriber` with Snowflake `AI_TRANSCRIBE`, Whisper, AWS Transcribe, Deepgram, or Azure Speech.
- OCR: implement `ImageTextExtractor` with existing Data Forge OCR engines, Tesseract, PaddleOCR, Docling, or Azure Document Intelligence.
- VLM: implement `VideoAnalyzer` with Qwen-VL, Gemini, Claude vision, Bedrock, or a Snowpark Container Service.
- Embeddings: implement `Embedder` with Snowflake `AI_EMBED`, OpenAI-compatible embeddings, Bedrock Titan, or sentence-transformers.
- Storage: implement `VectorStore` with Snowflake VECTOR tables, pgvector, OpenSearch, Milvus, Pinecone, or Weaviate.
- Generation: pass the existing inference gateway to `RAGEngine(generator=...)`.

## Enterprise Guardrails

- Tenant and collection are first-class fields on every source and query.
- Retrieval supports modality filters for audio-only, slide-only, video-only, or blended search.
- Context building enforces a hard character budget before calling the LLM.
- Every context block includes citation metadata and modality labels.
- Default local embeddings are deterministic and dependency-free for CI; production should replace them with a real embedding model.
