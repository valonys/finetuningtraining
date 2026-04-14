"""
Unit tests for the Docs RAG layer.

Covers:
  - Tokenizer behaviour (lowercase, alnum split, stopword removal)
  - Retriever correctness on representative help queries
  - hits_to_sources projection
  - Prompt builder structure (heading present, citation rules included,
    snippets injected)
"""
from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────
def test_tokenize_lowercases_and_splits():
    from app.rag.retriever import tokenize
    assert tokenize("Hello World!") == ["hello", "world"]


def test_tokenize_drops_stopwords():
    from app.rag.retriever import tokenize
    out = tokenize("The cat is on the mat")
    assert "cat" in out and "mat" in out
    assert "the" not in out and "is" not in out and "on" not in out


def test_tokenize_keeps_underscored_identifiers():
    from app.rag.retriever import tokenize
    out = tokenize("Set OLLAMA_API_KEY in the env")
    assert "ollama_api_key" in out


def test_tokenize_drops_single_char_tokens():
    from app.rag.retriever import tokenize
    assert "a" not in tokenize("a b cat")


# ──────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch):
    """Ensure each test gets a fresh retriever singleton."""
    from app.rag import retriever as r
    r.reset_retriever()
    yield
    r.reset_retriever()


def test_retriever_finds_install_article_for_install_query():
    from app.rag import get_retriever
    hits = get_retriever().retrieve("how do I install the backend?", k=3)
    titles = [h.article.title for h in hits]
    assert any("Installation" in t for t in titles), \
        f"Expected an install-related article in top-3, got {titles}"


def test_retriever_finds_ocr_article_for_ocr_query():
    from app.rag import get_retriever
    hits = get_retriever().retrieve("which OCR engine should I use for PDFs?", k=3)
    titles = [h.article.title for h in hits]
    assert any("OCR" in t for t in titles), titles


def test_retriever_finds_pretraining_article_for_scratch_query():
    from app.rag import get_retriever
    hits = get_retriever().retrieve("can I train a model from scratch?", k=3)
    titles = [h.article.title for h in hits]
    assert any("Pre-training" in t or "pretraining" in t.lower() for t in titles), titles


def test_retriever_finds_grpo_article_for_reward_query():
    from app.rag import get_retriever
    hits = get_retriever().retrieve("how does reward shaping work for math?", k=3)
    titles = [h.article.title for h in hits]
    assert any("GRPO" in t for t in titles), titles


def test_retriever_returns_empty_for_empty_query():
    from app.rag import get_retriever
    assert get_retriever().retrieve("", k=5) == []
    assert get_retriever().retrieve("   ", k=5) == []


def test_retriever_respects_k():
    from app.rag import get_retriever
    hits = get_retriever().retrieve("training", k=2)
    assert len(hits) <= 2


def test_retriever_filters_min_score():
    from app.rag import get_retriever
    # Junk query should not score high enough on any article
    hits = get_retriever().retrieve("xqzpwvk gibberish", k=5, min_score=0.5)
    assert hits == []


def test_retriever_orders_by_score_desc():
    from app.rag import get_retriever
    hits = get_retriever().retrieve("install backend uvicorn", k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


# ──────────────────────────────────────────────────────────────
# Sources projection
# ──────────────────────────────────────────────────────────────
def test_hits_to_sources_shape():
    from app.rag import get_retriever
    from app.rag.prompts import hits_to_sources
    hits = get_retriever().retrieve("installation", k=2)
    sources = hits_to_sources(hits)
    for s in sources:
        assert set(s.keys()) == {"title", "section", "article_id", "score"}
        assert isinstance(s["score"], float)
        assert isinstance(s["title"], str) and s["title"]


# ──────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────
def test_build_rag_prompt_includes_grounding_and_format_rules():
    from app.rag import build_rag_prompt
    prompt, hits = build_rag_prompt("how do I upload a PDF?", k=3)
    assert "ValonyLabs Studio Assistant" in prompt
    assert "GROUNDING RULES" in prompt
    assert "RESPONSE FORMAT" in prompt
    assert "## Level-2 headings" in prompt
    assert "Numbered lists" in prompt
    # Snippets injected
    assert "Article" in prompt
    assert len(hits) > 0


def test_build_rag_prompt_handles_no_match():
    from app.rag import build_rag_prompt
    prompt, hits = build_rag_prompt("xqzpwvk nonsense word salad", k=5, min_score=10.0)
    assert hits == []
    assert "No relevant articles matched" in prompt


def test_build_rag_prompt_k_truncates_to_actual_hits():
    from app.rag import build_rag_prompt
    prompt, hits = build_rag_prompt("install", k=2)
    # The prompt header references the *actual* number of returned hits,
    # not the requested k (so the model isn't told "top 5" when only 2 matched).
    assert f"top {len(hits)}" in prompt


# ──────────────────────────────────────────────────────────────
# Corpus integrity
# ──────────────────────────────────────────────────────────────
def test_corpus_articles_have_unique_ids():
    from app.rag import ARTICLES
    ids = [a.article_id for a in ARTICLES]
    assert len(ids) == len(set(ids)), f"Duplicate article ids: {ids}"


def test_corpus_articles_have_nonempty_bodies():
    from app.rag import ARTICLES
    for a in ARTICLES:
        assert a.title and a.section and a.article_id and a.body.strip(), \
            f"Empty field on article {a.article_id}"


def test_find_article():
    from app.rag.corpus import find_article
    assert find_article("install") is not None
    assert find_article("nonexistent_xyz") is None
