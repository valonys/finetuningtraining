"""
app/rag/retriever.py
────────────────────
BM25 over the Docs corpus.

Why BM25 and not embeddings:

* Corpus is 17 short articles (~12 KB total). At this size, lexical
  retrieval with proper tokenization is competitive with — often
  better than — neural embeddings on out-of-distribution queries.
* No model download, no GPU, no embedding-server dependency. ~1 ms
  per query, deterministic.
* Trivially upgradable later: swap `BM25Retriever` for an embedding
  retriever and the rest of the RAG pipeline (prompt builder, chat
  endpoint) stays unchanged.

The tokenizer is intentionally permissive: lowercase, alphanumeric +
underscore, drop standard English stopwords. Good enough for a Studio
Docs corpus where users ask things like *"how do I upload a pdf?"*.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .corpus import ARTICLES, DocArticle

logger = logging.getLogger(__name__)


# Small English stopword list — kept inline so we don't add yet another
# dep just for a 50-word lookup table.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "i", "if",
    "in", "into", "is", "it", "its", "may", "might", "must", "no", "not",
    "of", "on", "or", "should", "so", "such", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "too", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your",
})

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric word split, stopwords removed."""
    return [
        tok.lower()
        for tok in _TOKEN_RE.findall(text)
        if tok.lower() not in _STOPWORDS and len(tok) > 1
    ]


@dataclass
class RetrievalHit:
    article: DocArticle
    score: float


class DocsRetriever:
    """BM25 retriever over the docs corpus."""

    def __init__(self, articles: Optional[list[DocArticle]] = None):
        self.articles = list(articles) if articles is not None else list(ARTICLES)
        self._build_index()

    def _build_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RuntimeError(
                "rank_bm25 is required for the Docs RAG. "
                "Install it: `pip install rank_bm25`"
            ) from e

        # Index combines title + section + body, with title weighted by
        # repetition (cheap way to give title hits a boost in BM25).
        self._tokenized_corpus: list[list[str]] = []
        for art in self.articles:
            doc = (
                f"{art.title} {art.title} "        # title boost (×2)
                f"{art.section} "
                f"{art.body}"
            )
            self._tokenized_corpus.append(tokenize(doc))
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.0) -> list[RetrievalHit]:
        """Return the top-k articles for `query`, sorted by score desc."""
        if not query.strip():
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(
            zip(self.articles, scores), key=lambda p: p[1], reverse=True
        )
        out: list[RetrievalHit] = []
        for art, sc in ranked[:k]:
            if sc <= min_score:
                continue
            out.append(RetrievalHit(article=art, score=float(sc)))
        return out


# ── Cached singleton ────────────────────────────────────────────────
_retriever: Optional[DocsRetriever] = None


def get_retriever() -> DocsRetriever:
    """Module-level cached retriever — index is built once on first use."""
    global _retriever
    if _retriever is None:
        logger.info(f"📚 Building Docs RAG index over {len(ARTICLES)} articles")
        _retriever = DocsRetriever()
    return _retriever


def reset_retriever() -> None:
    """Drop the cached retriever (for tests)."""
    global _retriever
    _retriever = None
