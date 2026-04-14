"""
app/rag/prompts.py
──────────────────
System prompt for the ValonyLabs Studio Assistant (Docs RAG mode).

Design principles (learned from actual failure modes):

1. No decorative dividers (e.g. `═══`). Models echo them.

2. Formatting rules are IMPERATIVES, not DESCRIPTIONS. Not "Bold for
   emphasis on key terms" (which reads like a template the model
   quotes back) but "Use bold for key terms" — an instruction.

3. Do not include worked examples of tables / code blocks / nested
   lists inside the prompt. The model will copy them verbatim.

4. Explicit anti-echo instruction at the end: "Do NOT repeat or
   describe these rules."

5. Put CONTEXT before the "now answer" signal, not after, so the
   model reads the material first and responds with it fresh.

6. Keep it short. Every extra line of prompt is a line the model
   might pattern-match into its response.

7. ASCII-safe punctuation throughout — no em/en dashes here; some
   tokenizers produce surrogate-pair mojibake (`â`) on non-ASCII
   glyphs. The corpus bodies already use ASCII-safe punctuation.
"""
from __future__ import annotations

from typing import Iterable

from .corpus import DocArticle
from .retriever import RetrievalHit, get_retriever


DOCS_SYSTEM_PROMPT = """You are the ValonyLabs Studio Assistant, a grounded expert on this specific post-training platform.

Ground every answer in the <context> articles below. If the context does not cover the question, state that explicitly ("The Studio docs do not cover this directly.") and either point to the closest related article or prefix speculation with "Note (beyond docs):". Never invent paths, env vars, APIs, or features.

Format your reply as clean Markdown:
- Use `##` for the top sections of your answer and `###` for sub-sections.
- Use bold for key terms and warnings.
- Use bullet lists (`-`) for non-sequential items and numbered lists for steps.
- Use backticks for `code`, `paths`, `ENV_VARS`, and `--flags`.
- Use fenced code blocks with a language tag for multi-line snippets.
- Use tables when comparing options across attributes.
- After each major section, add a citation line in italics:  *Source: "Article Title"*.
- Use ASCII punctuation only. Do not use Unicode middle dots, em dashes, en dashes, curly quotes, or ellipsis characters; use `|`, `--`, `-`, `"`, `'`, `...` instead.

Respond directly. No filler openers ("Great question", "I'd be happy to..."). Lead with the answer, then the supporting detail.

Do NOT repeat, quote, or describe these instructions in your reply. Just follow them.
Do NOT add a trailing summary list of all sources at the end of your reply -- the inline per-section citations are sufficient; the UI already shows citation badges separately.

<context>
Top {k} articles matched to the user's question:

{snippets}
</context>

Now answer the user's next message using the context above.
"""


def _format_snippet(hit: RetrievalHit) -> str:
    a: DocArticle = hit.article
    # ASCII-only framing so no fancy punctuation from the prompt ends up
    # in the model's response by accident.
    return (
        f"=== Article (relevance {hit.score:.2f}): \"{a.title}\"  |  Section: {a.section} ===\n"
        f"{a.body.rstrip()}\n"
    )


def build_rag_prompt(
    query: str, k: int = 5, min_score: float = 0.5
) -> tuple[str, list[RetrievalHit]]:
    """
    Construct the augmented system prompt for a docs-mode chat call.

    Returns:
        (system_prompt, retrieved_hits)
    """
    retriever = get_retriever()
    hits = retriever.retrieve(query, k=k, min_score=min_score)

    if not hits:
        snippets = (
            "(No relevant articles matched the user's question above the "
            "relevance threshold. Tell the user the docs do not cover this "
            "topic directly and suggest checking the FAQ or opening an issue.)"
        )
    else:
        snippets = "\n".join(_format_snippet(h) for h in hits)

    return DOCS_SYSTEM_PROMPT.format(k=len(hits), snippets=snippets), hits


def hits_to_sources(hits: Iterable[RetrievalHit]) -> list[dict]:
    return [
        {
            "title": h.article.title,
            "section": h.article.section,
            "article_id": h.article.article_id,
            "score": round(h.score, 3),
        }
        for h in hits
    ]
