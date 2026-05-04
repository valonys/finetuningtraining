from pathlib import Path

from app.multimodal import (
    ContextBuilder,
    DeterministicHashEmbedder,
    Modality,
    MultimodalPipeline,
    PipelineConfig,
    RAGEngine,
    SQLiteVectorStore,
)
from app.multimodal.processors import TextFileProcessor


def test_multimodal_pipeline_indexes_and_searches_across_modalities(tmp_path: Path):
    db = tmp_path / "mm.db"
    pipeline = MultimodalPipeline(
        config=PipelineConfig(
            tenant_id="tenant_a",
            collection="meeting_1",
            chunk_target_chars=240,
            chunk_overlap_chars=20,
            embedding_dim=64,
        ),
        embedder=DeterministicHashEmbedder(dim=64),
        vector_store=SQLiteVectorStore(db),
    )

    audio = tmp_path / "audio.txt"
    audio.write_text("The team discussed budget, pricing, and remote control design.")
    slide = tmp_path / "slide.txt"
    slide.write_text("The slide listed product features and remote control buttons.")

    processor = TextFileProcessor()
    records = []
    records += processor.process(
        str(audio),
        tenant_id="tenant_a",
        collection="meeting_1",
        source_type=Modality.AUDIO,
    )
    records += processor.process(
        str(slide),
        tenant_id="tenant_a",
        collection="meeting_1",
        source_type=Modality.SLIDE,
    )

    chunks = pipeline.index_records(records)
    assert len(chunks) == 2

    all_results = pipeline.search("remote control product features", top_k=2)
    assert len(all_results) == 2
    assert {r.chunk.source.source_type for r in all_results} == {Modality.AUDIO, Modality.SLIDE}

    slide_results = pipeline.search(
        "remote control product features",
        top_k=2,
        source_type=Modality.SLIDE,
    )
    assert len(slide_results) == 1
    assert slide_results[0].chunk.source.source_type == Modality.SLIDE


def test_rag_engine_builds_cited_prompt(tmp_path: Path):
    pipeline = MultimodalPipeline(
        config=PipelineConfig(
            tenant_id="tenant_a",
            collection="meeting_2",
            chunk_target_chars=240,
            chunk_overlap_chars=20,
            embedding_dim=64,
            max_context_chars=2000,
        ),
        embedder=DeterministicHashEmbedder(dim=64),
        vector_store=SQLiteVectorStore(tmp_path / "rag.db"),
    )
    transcript = tmp_path / "video_segment.txt"
    transcript.write_text("The video segment shows consensus on next steps and action items.")
    records = TextFileProcessor().process(
        str(transcript),
        tenant_id="tenant_a",
        collection="meeting_2",
        source_type=Modality.VIDEO,
    )
    pipeline.index_records(records)

    prompts = []

    def fake_generator(prompt: str) -> str:
        prompts.append(prompt)
        return "The team reached consensus on next steps [1]."

    rag = RAGEngine(
        pipeline,
        generator=fake_generator,
        context_builder=ContextBuilder(max_chars=2000),
    )
    answer = rag.answer("How did the team reach consensus?", top_k=3)

    assert answer.sources
    assert "[1]" in answer.answer
    assert "modality=video" in prompts[0]
    assert "Use only the provided context" in prompts[0] or "provided context" in prompts[0]
