"""
app/rag/
────────
RAG (Retrieval-Augmented Generation) for the in-app Docs section.

Pragmatic design — no vector DB, no embedding model. The Docs corpus is
small (~15 short articles, ~12 KB total), so BM25 is the right answer:

  * Pure-Python (`rank_bm25`), no compiled deps, no model download
  * Deterministic, debuggable, fast (<1 ms per query)
  * Trivially upgradable to vector embeddings later if the corpus grows

The corpus is the **server-side authoritative source** for what the RAG
knows. The frontend's `frontend/src/docs.tsx` is the rendering layer
(JSX components for the docs sidebar). They're maintained in sync by
hand for now; a build-time codegen step is a good follow-up.

Public surface:
    from app.rag import (
        ARTICLES,           # list[DocArticle]
        get_retriever,      # cached singleton DocsRetriever
        build_rag_prompt,   # produce the augmented system prompt
    )
"""
from .corpus import ARTICLES, DocArticle
from .retriever import DocsRetriever, get_retriever
from .prompts import build_rag_prompt, DOCS_SYSTEM_PROMPT

__all__ = [
    "ARTICLES",
    "DocArticle",
    "DocsRetriever",
    "get_retriever",
    "build_rag_prompt",
    "DOCS_SYSTEM_PROMPT",
]
