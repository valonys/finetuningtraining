"""Retrieval-augmented generation assembly for multimodal content."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .pipeline import MultimodalPipeline
from .schemas import Modality, RetrievalResult


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[str]
    context: str
    results: list[RetrievalResult]


class ContextBuilder:
    """Build cited context while enforcing a token/character budget."""

    def __init__(self, max_chars: int = 12000):
        if max_chars < 1000:
            raise ValueError("max_chars must be >= 1000")
        self.max_chars = max_chars

    def build(self, results: list[RetrievalResult]) -> tuple[str, list[str]]:
        context_parts: list[str] = []
        sources: list[str] = []
        used = 0
        for idx, result in enumerate(results, start=1):
            chunk = result.chunk
            source = _format_source(chunk.source)
            part = (
                f"[{idx}] {source}\n"
                f"modality={chunk.source.source_type.value} score={result.score:.4f}\n"
                f"{chunk.text.strip()}"
            )
            if used + len(part) > self.max_chars:
                break
            context_parts.append(part)
            sources.append(f"[{idx}] {source}")
            used += len(part)
        return "\n\n".join(context_parts), sources


class RAGEngine:
    """Provider-neutral RAG facade.

    The generator is a callable so deployments can pass existing inference
    infrastructure: ValonyLabs `/v1/chat`, OpenAI-compatible clients,
    Snowflake Cortex COMPLETE, Bedrock, or an internal gateway.
    """

    def __init__(
        self,
        pipeline: MultimodalPipeline,
        *,
        generator: Callable[[str], str],
        context_builder: ContextBuilder | None = None,
    ):
        self.pipeline = pipeline
        self.generator = generator
        self.context_builder = context_builder or ContextBuilder(
            max_chars=pipeline.config.max_context_chars
        )

    def answer(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source_type: Modality | None = None,
    ) -> RAGAnswer:
        results = self.pipeline.search(query, top_k=top_k, source_type=source_type)
        context, sources = self.context_builder.build(results)
        if not context:
            return RAGAnswer(
                answer="I do not have enough indexed multimodal context to answer.",
                sources=[],
                context="",
                results=[],
            )
        prompt = _prompt(query=query, context=context, sources=sources)
        return RAGAnswer(
            answer=self.generator(prompt),
            sources=sources,
            context=context,
            results=results,
        )


def _prompt(*, query: str, context: str, sources: list[str]) -> str:
    source_list = "\n".join(sources)
    return (
        "You are an enterprise multimodal RAG assistant. Answer using only the "
        "provided context. Cite sources with bracket numbers. If the context "
        "does not support an answer, say so.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        f"Available sources:\n{source_list}\n\n"
        "Answer:"
    )


def _format_source(source) -> str:
    label = source.title or source.source_uri
    locator = ""
    if source.start_time_s is not None:
        end = "" if source.end_time_s is None else f"-{source.end_time_s:.1f}s"
        locator = f" @{source.start_time_s:.1f}s{end}"
    elif source.page is not None:
        locator = f" page {source.page}"
    elif source.frame is not None:
        locator = f" frame {source.frame}"
    return f"{label}{locator}"
