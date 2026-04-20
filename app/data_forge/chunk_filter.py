"""
app/data_forge/chunk_filter.py
───────────────────────────────
Heuristic quality filter over chunks produced by the semantic chunker.

The symptom without this filter: a book PDF goes through ingest + chunk
+ Q/A synthesis, and the output dataset is littered with trivia like:

    {"instruction": "Who is the author of this article?",
     "response":    "The author is ..."}
    {"instruction": "How many chapters does this book have?",
     "response":    "Twelve."}
    {"instruction": "What is the ISBN?",
     "response":    "978-..."}

...because the synth pipeline saw cover pages, tables of contents, and
bibliography fragments as inputs and produced the only questions those
passages could answer.

The fix is to **drop those passages before synthesis**. A chunk is noise
if any of the following are true:

  * It is too short to contain real content (< MIN_CHARS).
  * It is too digit-dense (page numbers, ISBNs, call numbers).
  * It is too caps-dense (title pages, running headers).
  * Its sentence completeness is too low (TOC lines, bibliography entries).
  * It matches bibliography / citation / TOC / front-matter signatures.
  * It's essentially a list of short lines with no connecting prose.

Tuning rationale: these thresholds were set from inspecting rejected
vs. kept chunks on a corpus of technical books, reports, and articles.
They err toward *keeping* ambiguous content (the Q/A synth has its own
post-filter for trivial questions) rather than rejecting too aggressively.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Tuple


# ── Tunables ────────────────────────────────────────────────────────
MIN_CHARS = 200            # chunks shorter than this can't contain real knowledge
MAX_DIGIT_RATIO = 0.22     # above this ratio, it's likely a page-number / bibliography fragment
MAX_CAPS_RATIO = 0.45      # above this ratio, likely a title page or running header
MIN_SENTENCE_LEN = 5       # mean sentence length in words; TOC entries are 2-4 words
MIN_SENTENCE_COUNT = 2     # a real chunk has at least a couple sentences

# Regexes for common front-matter / bibliography / TOC signatures.
# A chunk matching ANY of these with high density is rejected.
_PATTERNS_HARD_REJECT = [
    # Bibliography entries: "Smith, J. (2020). Title. Journal, 1(2), 3-4."
    re.compile(r"^[A-Z][a-z]+,\s+[A-Z]\.\s*.*\(\d{4}\)\.", re.MULTILINE),
    # ISBN blocks
    re.compile(r"\bISBN[:\s-]+[\d\-Xx]{10,17}", re.IGNORECASE),
    # "Copyright © 20XX" blurbs
    re.compile(r"copyright\s*(?:©|\(c\))\s*\d{4}", re.IGNORECASE),
    # Multi-line "Chapter X ..... NNN" TOC entries
    re.compile(r"(?:Chapter|Section|Part)\s+\d+[\s\.]+\d{1,4}\s*$", re.IGNORECASE | re.MULTILINE),
    # Pure "Page N of M" strips
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
    # "Table of Contents" / "Index" / "Bibliography" / "References" as the whole heading
    re.compile(r"^\s*#+\s*(?:Table of Contents|Index|Bibliography|References|Acknowledgements?|Dedication|Foreword|Preface|Glossary)\s*$", re.IGNORECASE | re.MULTILINE),
]

# Words that should NOT dominate a chunk (they're front-matter markers).
_FRONT_MATTER_WORDS = frozenset({
    "copyright", "isbn", "printed", "published", "publisher", "edition",
    "foreword", "preface", "acknowledgements", "acknowledgments",
    "dedication", "author", "illustrator",
})

# Fuzzy TOC signature: many lines, each ending in digits (page number),
# each line short, little connecting prose.
_TOC_LINE_RE = re.compile(r"^\s*.{0,80}\s+\d{1,4}\s*$", re.MULTILINE)

# ── New v2 patterns (arXiv / academic PDF focused) ─────────────────

# Numbered reference entries: "[1] Author et al. Title..." clusters
_NUMBERED_REF_RE = re.compile(r"^\s*\[\d{1,3}\]\s+[A-Z]", re.MULTILINE)

# Figure/Table captions: "Figure 3: ..." or "Table 2.1 –"
_FIGURE_TABLE_RE = re.compile(
    r"^\s*(?:Fig(?:ure)?|Table)\s+\d+[\.:–\-]",
    re.IGNORECASE | re.MULTILINE,
)

# Citation clusters: text that's mostly [1,2,3] or [Author, Year] references
_CITE_BRACKET_RE = re.compile(r"\[\d{1,3}(?:,\s*\d{1,3})*\]")

# LaTeX artifacts left over from PDF extraction
_LATEX_CMD_RE = re.compile(r"\\(?:begin|end|textbf|textit|cite|ref|label|section|subsection)\b")

# Index entries: "algorithm, 42, 105, 312" or "backpropagation  see gradient"
_INDEX_ENTRY_RE = re.compile(
    r"^\s*[a-zA-Z][a-zA-Z\s,-]+(?:\d{1,4}(?:\s*[,–-]\s*\d{1,4})*|see\s+[a-zA-Z])",
    re.MULTILINE,
)


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    dropped_count: int = 0
    reasons: Counter[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = Counter()

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped_count": self.dropped_count,
            "reasons": dict(self.reasons),
        }


# ── Individual checks ───────────────────────────────────────────────
def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    alnum = sum(c.isalnum() for c in text)
    if alnum == 0:
        return 0.0
    digits = sum(c.isdigit() for c in text)
    return digits / alnum


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def _sentence_lengths(text: str) -> list[int]:
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return [len(s.split()) for s in sents]


def _toc_density(text: str) -> float:
    """Fraction of lines that look like 'Short Title ................  123'."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    toc_like = sum(1 for ln in lines if _TOC_LINE_RE.match(ln))
    return toc_like / len(lines)


def _front_matter_word_density(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    matches = sum(1 for w in words if w in _FRONT_MATTER_WORDS)
    return matches / len(words)


def _bibliography_density(text: str) -> float:
    """Fraction of lines that look like numbered reference entries [1], [2]..."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return 0.0
    ref_lines = sum(1 for ln in lines if _NUMBERED_REF_RE.match(ln))
    return ref_lines / len(lines)


def _citation_bracket_density(text: str) -> float:
    """Fraction of tokens that are citation brackets like [1,2,3]."""
    words = text.split()
    if len(words) < 10:
        return 0.0
    cite_chars = sum(len(m.group()) for m in _CITE_BRACKET_RE.finditer(text))
    return cite_chars / max(len(text), 1)


def _latex_artifact_density(text: str) -> float:
    """Fraction of lines containing LaTeX commands."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    latex_lines = sum(1 for ln in lines if _LATEX_CMD_RE.search(ln))
    return latex_lines / len(lines)


def _figure_caption_only(text: str) -> bool:
    """True if the chunk is essentially just figure/table captions."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 1 or len(lines) > 5:
        return False
    return all(
        _FIGURE_TABLE_RE.match(ln) or len(ln.strip()) < 20
        for ln in lines
    )


def _index_page_density(text: str) -> float:
    """Fraction of lines that look like book-index entries."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 5:
        return 0.0
    idx_lines = sum(1 for ln in lines if _INDEX_ENTRY_RE.match(ln))
    return idx_lines / len(lines)


def _is_acknowledgements_section(text: str) -> bool:
    """Detect acknowledgements sections (short, starts with the heading)."""
    stripped = text.strip()
    first_line = stripped.split("\n", 1)[0].lower()
    if not re.search(r"acknowledg[ei]?ments?", first_line):
        return False
    # Acknowledgements sections are typically short (< 1500 chars)
    return len(stripped) < 1500


# ── Public API ──────────────────────────────────────────────────────
def is_noise(chunk_text: str) -> Tuple[bool, str]:
    """
    Return (True, reason) if the chunk should be rejected.
    Return (False, "") if it's worth keeping.
    """
    if len(chunk_text.strip()) < MIN_CHARS:
        return True, "too_short"

    if _digit_ratio(chunk_text) > MAX_DIGIT_RATIO:
        return True, "digit_dense"

    if _caps_ratio(chunk_text) > MAX_CAPS_RATIO:
        return True, "all_caps"

    sent_lens = _sentence_lengths(chunk_text)
    if len(sent_lens) < MIN_SENTENCE_COUNT:
        return True, "too_few_sentences"
    mean_sent_len = sum(sent_lens) / len(sent_lens)
    if mean_sent_len < MIN_SENTENCE_LEN:
        return True, "short_sentences"

    if _toc_density(chunk_text) > 0.5:
        return True, "toc_like"

    if _front_matter_word_density(chunk_text) > 0.06:
        return True, "front_matter"

    for pattern in _PATTERNS_HARD_REJECT:
        if pattern.search(chunk_text):
            return True, "front_matter_signature"

    # ── v2 academic / book noise rules ───────────────────────
    if _bibliography_density(chunk_text) > 0.4:
        return True, "bibliography"

    if _is_acknowledgements_section(chunk_text):
        return True, "acknowledgements"

    if _figure_caption_only(chunk_text):
        return True, "figure_caption"

    if _index_page_density(chunk_text) > 0.5:
        return True, "index_page"

    if _citation_bracket_density(chunk_text) > 0.08:
        return True, "citation_cluster"

    if _latex_artifact_density(chunk_text) > 0.3:
        return True, "latex_artifacts"

    return False, ""


def filter_chunks(chunks: list[dict]) -> Tuple[list[dict], dict]:
    """
    Filter a list of chunk dicts (as produced by `chunker.chunk_records`)
    and return (kept, stats_dict).

    Each chunk dict must have a `chunk` key with the text content.
    """
    stats = FilterStats()
    stats.total = len(chunks)

    kept: list[dict] = []
    for ch in chunks:
        text = ch.get("chunk", "")
        noise, reason = is_noise(text)
        if noise:
            stats.dropped_count += 1
            stats.reasons[reason] += 1
        else:
            kept.append(ch)
            stats.kept += 1

    return kept, stats.as_dict()
