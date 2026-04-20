"""
app/harvesters/arxiv.py
───────────────────────
Keyword -> Top-N arXiv papers -> Abstracts (or PDFs) -> files in ./data/uploads/.

Design (why HTTP + stdlib XML, not the `arxiv` PyPI package):

  The `arxiv` package (lukasschwab/python-arxiv-wrapper) is convenient
  but introduces a transitive dependency and pins its own `requests`
  range, which can conflict with the pinned version in this project's
  requirements.

  The arXiv API is a simple Atom/XML endpoint at
  ``http://export.arxiv.org/api/query``. One GET with ``search_query``
  and ``max_results`` returns everything we need: title, authors,
  abstract (``<summary>``), categories, PDF link, and publication date.
  Parsing ~20 entries with ``xml.etree.ElementTree`` takes < 1 ms, no
  extra wheels required.

Two harvest modes:

  ``mode="abstract"``  (default)
      Writes a plain-text file per paper with a provenance header
      (title, authors, arXiv ID, categories, date) followed by the
      abstract text.  These are small (~2 KB each) and feed directly
      into the Data Forge chunker + Q/A synth pipeline.

  ``mode="full"``
      Downloads the PDF from ``arxiv.org/pdf/<id>.pdf`` and saves it
      to ``data/uploads/``. The existing Data Forge PDF parser handles
      extraction downstream.

Filename convention:
    ``{arxiv_id_sanitized}_{title_slug}.txt``  (abstract mode)
    ``{arxiv_id_sanitized}_{title_slug}.pdf``  (full mode)

Typical use:

    h = ArxivHarvester()
    report = h.harvest("transformer attention mechanism", max_results=10)
    for rec in report.harvested:
        print(rec.title, "->", rec.file_path)
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Atom XML namespace used in arXiv API responses
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_API_URL = "http://export.arxiv.org/api/query"


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class ArxivSearchResult:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    pdf_url: str
    categories: str
    published: str


@dataclass
class ArxivHarvestedPaper:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    categories: str
    published: str
    text: str
    file_path: str
    char_count: int = 0
    mode: str = "abstract"


@dataclass
class ArxivHarvestReport:
    harvested: list[ArxivHarvestedPaper] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    query: str = ""
    max_requested: int = 0


# ── Filename sanitisation ──────────────────────────────────────────
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_slug(title: str, maxlen: int = 80) -> str:
    """Convert an arbitrary title string into a filesystem-safe slug."""
    cleaned = _SAFE_NAME_RE.sub("_", title).strip("._")
    return (cleaned or "paper")[:maxlen]


def _sanitize_arxiv_id(arxiv_id: str) -> str:
    """Replace dots and slashes in an arXiv ID so it is filename-safe.

    Examples:
        "2301.12345" -> "2301_12345"
        "cs/0601001"  -> "cs_0601001"
    """
    return arxiv_id.replace("/", "_").replace(".", "_")


# ── XML parsing helpers ────────────────────────────────────────────

def _text(element: Optional[ET.Element]) -> str:
    """Return stripped text of an XML element, or empty string."""
    if element is None:
        return ""
    return (element.text or "").strip()


def _extract_arxiv_id(id_text: str) -> str:
    """Extract the bare arXiv identifier from the <id> URL.

    The <id> element contains a full URL like:
        http://arxiv.org/abs/2301.12345v2
    We want just "2301.12345v2" (or "2301.12345" without the version
    suffix for cleaner filenames, but we keep the version for
    uniqueness).
    """
    # Strip trailing version for a cleaner ID; keep the full form available
    # via the raw URL if needed.
    match = re.search(r"arxiv\.org/abs/(.+)$", id_text)
    if match:
        return match.group(1).strip()
    # Fallback: return whatever is after the last slash
    return id_text.rsplit("/", 1)[-1].strip()


def _parse_entry(entry: ET.Element) -> Optional[ArxivSearchResult]:
    """Parse one <entry> element from the arXiv Atom feed."""
    id_elem = entry.find(f"{{{_ATOM_NS}}}id")
    if id_elem is None:
        return None
    raw_id = _text(id_elem)
    arxiv_id = _extract_arxiv_id(raw_id)
    if not arxiv_id:
        return None

    title_raw = _text(entry.find(f"{{{_ATOM_NS}}}title"))
    # arXiv titles may contain newlines; normalise to single spaces
    title = re.sub(r"\s+", " ", title_raw).strip()

    summary_raw = _text(entry.find(f"{{{_ATOM_NS}}}summary"))
    abstract = re.sub(r"\s+", " ", summary_raw).strip()

    # Authors: multiple <author><name>...</name></author>
    author_names = []
    for author_el in entry.findall(f"{{{_ATOM_NS}}}author"):
        name = _text(author_el.find(f"{{{_ATOM_NS}}}name"))
        if name:
            author_names.append(name)
    authors = ", ".join(author_names)

    # PDF link: <link title="pdf" href="..." rel="related" type="application/pdf"/>
    # Alternatively: <link rel="alternate" ... href="http://arxiv.org/abs/..."/>
    pdf_url = ""
    for link_el in entry.findall(f"{{{_ATOM_NS}}}link"):
        if link_el.get("title") == "pdf":
            pdf_url = link_el.get("href", "")
            break
    # Fallback: construct from ID
    if not pdf_url:
        pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"

    # Categories: <category term="cs.LG" .../>
    # arXiv uses the Atom namespace for category elements but also its
    # own namespace; the term attribute is what we need regardless.
    cats = []
    for cat_el in entry.findall(f"{{{_ATOM_NS}}}category"):
        term = cat_el.get("term", "")
        if term:
            cats.append(term)
    # Also try without namespace (arXiv sometimes uses the arxiv namespace)
    if not cats:
        for cat_el in entry.findall("category"):
            term = cat_el.get("term", "")
            if term:
                cats.append(term)
    categories = ", ".join(cats)

    published = _text(entry.find(f"{{{_ATOM_NS}}}published"))

    return ArxivSearchResult(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        pdf_url=pdf_url,
        categories=categories,
        published=published,
    )


# ── Harvester ───────────────────────────────────────────────────────

class ArxivHarvester:
    """
    Keyword search + abstract/PDF fetch -> files in data/uploads/.

    Typical use:

        h = ArxivHarvester()
        report = h.harvest("transformer attention mechanism", max_results=10)
        for rec in report.harvested:
            print(rec.title, "->", rec.file_path)

    The files land in ``./data/uploads/`` so the Data Forge picks them
    up via the standard ``ingest(paths)`` / ``forge/build_dataset`` flow.
    """

    def __init__(self, *, timeout: int = 30):
        self.timeout = timeout

    # ── Public API ─────────────────────────────────────────────

    def search(
        self,
        query: str,
        max_results: int = 20,
        categories: list[str] | None = None,
    ) -> list[ArxivSearchResult]:
        """Search arXiv by keyword and optional category filter.

        Parameters
        ----------
        query : str
            Free-text search query (searches title, abstract, comments).
        max_results : int
            Maximum number of results to return (1-100).
        categories : list[str] | None
            Optional list of arXiv category codes (e.g. ``["cs.LG", "cs.AI"]``).
            When provided, results are filtered to papers that belong to at
            least one of the listed categories via the ``cat:`` prefix in
            the arXiv query syntax.

        Returns
        -------
        list[ArxivSearchResult]
        """
        if not query.strip():
            raise ValueError("ArxivHarvester.search: query is empty")
        if max_results < 1 or max_results > 100:
            raise ValueError("max_results must be in [1, 100]")

        # Build the arXiv search_query string
        search_query = f"all:{query}"
        if categories:
            cat_clause = " OR ".join(f"cat:{c}" for c in categories)
            search_query = f"({search_query}) AND ({cat_clause})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        logger.info(f"arXiv search: query={search_query!r}, max_results={max_results}")
        resp = requests.get(_ARXIV_API_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        results: list[ArxivSearchResult] = []
        for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
            parsed = _parse_entry(entry)
            if parsed and parsed.title:
                results.append(parsed)

        logger.info(f"   {len(results)} results returned")
        return results

    def fetch_abstract(self, arxiv_id: str) -> Optional[str]:
        """Fetch the abstract for a single paper by arXiv ID.

        Parameters
        ----------
        arxiv_id : str
            e.g. ``"2301.12345"`` or ``"2301.12345v2"``

        Returns
        -------
        str | None
            The abstract text, or None if the paper could not be found.
        """
        params = {
            "id_list": arxiv_id,
            "max_results": 1,
        }
        try:
            resp = requests.get(_ARXIV_API_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch abstract for {arxiv_id}: {e}")
            return None

        root = ET.fromstring(resp.text)
        entry = root.find(f"{{{_ATOM_NS}}}entry")
        if entry is None:
            return None
        summary = _text(entry.find(f"{{{_ATOM_NS}}}summary"))
        if not summary:
            return None
        return re.sub(r"\s+", " ", summary).strip()

    def harvest(
        self,
        query: str,
        *,
        max_results: int = 20,
        mode: str = "abstract",
        output_dir: str = "./data/uploads",
        min_chars: int = 200,
    ) -> ArxivHarvestReport:
        """End-to-end: search -> fetch abstracts/PDFs -> write files.

        Parameters
        ----------
        query : str
            Free-text search query.
        max_results : int
            Max papers to retrieve.
        mode : str
            ``"abstract"`` writes .txt files with metadata + abstract.
            ``"full"`` downloads the PDF directly.
        output_dir : str
            Directory to write output files into.
        min_chars : int
            Minimum abstract length; papers with shorter abstracts are
            skipped (prevents low-content entries from polluting the
            downstream Q/A synth pipeline).

        Returns
        -------
        ArxivHarvestReport
        """
        if mode not in ("abstract", "full"):
            raise ValueError(f"mode must be 'abstract' or 'full', got {mode!r}")

        report = ArxivHarvestReport(query=query, max_requested=max_results)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        hits = self.search(query, max_results=max_results)

        for paper in hits:
            # -- Abstract mode: filter on abstract length --------
            if mode == "abstract":
                if len(paper.abstract) < min_chars:
                    report.skipped.append({
                        "title": paper.title,
                        "arxiv_id": paper.arxiv_id,
                        "reason": (
                            f"abstract too short "
                            f"({len(paper.abstract)} chars < {min_chars})"
                        ),
                    })
                    continue

                id_safe = _sanitize_arxiv_id(paper.arxiv_id)
                slug = _safe_slug(paper.title)
                fname = f"{id_safe}_{slug}.txt"
                fpath = out_dir / fname

                # Provenance header so downstream ingestion can attribute
                # the content in citations.
                header = (
                    f"# {paper.title}\n"
                    f"# Authors: {paper.authors}\n"
                    f"# arXiv ID: {paper.arxiv_id}\n"
                    f"# Categories: {paper.categories}\n"
                    f"# Published: {paper.published}\n"
                    f"# PDF: {paper.pdf_url}\n"
                    f"\n"
                )
                text = paper.abstract
                fpath.write_text(header + text, encoding="utf-8")
                logger.info(
                    f"   -> {paper.title[:60]}  -> {fname}  "
                    f"({len(text)} chars)"
                )

                report.harvested.append(ArxivHarvestedPaper(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    categories=paper.categories,
                    published=paper.published,
                    text=text,
                    file_path=str(fpath),
                    char_count=len(text),
                    mode="abstract",
                ))

            # -- Full mode: download PDF -------------------------
            elif mode == "full":
                id_safe = _sanitize_arxiv_id(paper.arxiv_id)
                slug = _safe_slug(paper.title)
                fname = f"{id_safe}_{slug}.pdf"
                fpath = out_dir / fname

                try:
                    pdf_resp = requests.get(
                        paper.pdf_url, timeout=self.timeout * 3
                    )
                    pdf_resp.raise_for_status()
                except requests.RequestException as e:
                    report.skipped.append({
                        "title": paper.title,
                        "arxiv_id": paper.arxiv_id,
                        "reason": f"PDF download failed: {e}",
                    })
                    continue

                fpath.write_bytes(pdf_resp.content)
                logger.info(
                    f"   -> {paper.title[:60]}  -> {fname}  "
                    f"({len(pdf_resp.content)} bytes)"
                )

                report.harvested.append(ArxivHarvestedPaper(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    categories=paper.categories,
                    published=paper.published,
                    text="",
                    file_path=str(fpath),
                    char_count=len(pdf_resp.content),
                    mode="full",
                ))

        return report
