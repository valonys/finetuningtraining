"""
Unit tests for the arXiv harvester.

All HTTP calls (requests.get to the arXiv API) are mocked out.
We verify:
  * search() parses Atom XML correctly into ArxivSearchResult dataclasses
  * harvest() writes .txt files with provenance headers (abstract mode)
  * Empty query raises ValueError
  * Papers with abstract shorter than min_chars are skipped and reported
  * Filename sanitisation produces safe, deterministic names
  * fetch_abstract() returns the abstract for a single paper by ID
  * harvest() in full mode downloads PDFs
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.harvesters.arxiv import (
    ArxivHarvester,
    ArxivHarvestReport,
    ArxivSearchResult,
    _safe_slug,
    _sanitize_arxiv_id,
)

# ── Sample Atom XML response from arXiv API ────────────────────────
_SAMPLE_ATOM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">ArXiv Query: search_query=all:transformer</title>
  <id>http://arxiv.org/api/query</id>
  <totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">2</totalResults>
  <entry>
    <id>http://arxiv.org/abs/2301.12345v2</id>
    <published>2023-01-15T18:00:00Z</published>
    <title>Attention Is All You Need: A Comprehensive Survey</title>
    <summary>
      We present a comprehensive survey of transformer architectures and their
      applications in natural language processing, computer vision, and
      reinforcement learning.  This paper reviews over 200 recent works and
      identifies key trends in self-attention mechanisms, positional encodings,
      and efficient transformer variants that reduce computational complexity
      from quadratic to linear in sequence length.
    </summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link href="http://arxiv.org/abs/2301.12345v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2301.12345v2" rel="related" type="application/pdf"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2302.67890v1</id>
    <published>2023-02-20T12:00:00Z</published>
    <title>Short Note</title>
    <summary>Brief.</summary>
    <author><name>Carol White</name></author>
    <link href="http://arxiv.org/abs/2302.67890v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2302.67890v1" rel="related" type="application/pdf"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

# A response with only the "Short Note" entry (abstract < 200 chars)
_SHORT_ONLY_ATOM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">ArXiv Query</title>
  <id>http://arxiv.org/api/query</id>
  <entry>
    <id>http://arxiv.org/abs/2302.67890v1</id>
    <published>2023-02-20T12:00:00Z</published>
    <title>Short Note</title>
    <summary>Brief.</summary>
    <author><name>Carol White</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2302.67890v1" rel="related" type="application/pdf"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

_SINGLE_ENTRY_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">ArXiv Query</title>
  <id>http://arxiv.org/api/query</id>
  <entry>
    <id>http://arxiv.org/abs/2301.12345v2</id>
    <published>2023-01-15T18:00:00Z</published>
    <title>Attention Is All You Need: A Comprehensive Survey</title>
    <summary>
      We present a comprehensive survey of transformer architectures and their
      applications in natural language processing, computer vision, and
      reinforcement learning.
    </summary>
    <author><name>Alice Smith</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2301.12345v2" rel="related" type="application/pdf"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


def _mock_response(text: str, status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response with the given text body."""
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.content = text.encode("utf-8")
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ── Filename sanitisation ──────────────────────────────────────────

@pytest.mark.parametrize("raw,expected_contains", [
    ("Attention Is All You Need",       "Attention_Is_All_You_Need"),
    ("Title with / slash",              "Title_with_slash"),
    ("Title: with colon",              "Title_with_colon"),
    ("Title (with parens)",            "Title_with_parens"),
    ("",                               "paper"),
])
def test_safe_slug(raw, expected_contains):
    out = _safe_slug(raw)
    assert "/" not in out and "\\" not in out
    assert 1 <= len(out) <= 80
    assert expected_contains in out


@pytest.mark.parametrize("arxiv_id,expected", [
    ("2301.12345",    "2301_12345"),
    ("2301.12345v2",  "2301_12345v2"),
    ("cs/0601001",    "cs_0601001"),
])
def test_sanitize_arxiv_id(arxiv_id, expected):
    assert _sanitize_arxiv_id(arxiv_id) == expected


# ── search() ───────────────────────────────────────────────────────

@patch("app.harvesters.arxiv.requests.get")
def test_search_parses_xml(mock_get):
    """search() should parse the Atom XML into ArxivSearchResult dataclasses."""
    mock_get.return_value = _mock_response(_SAMPLE_ATOM_XML)

    h = ArxivHarvester()
    results = h.search("transformer", max_results=10)

    assert len(results) == 2

    # First entry
    r0 = results[0]
    assert isinstance(r0, ArxivSearchResult)
    assert r0.arxiv_id == "2301.12345v2"
    assert "Attention Is All You Need" in r0.title
    assert "Alice Smith" in r0.authors
    assert "Bob Jones" in r0.authors
    assert "comprehensive survey" in r0.abstract.lower()
    assert r0.pdf_url == "http://arxiv.org/pdf/2301.12345v2"
    assert "cs.LG" in r0.categories
    assert "cs.CL" in r0.categories
    assert r0.published == "2023-01-15T18:00:00Z"

    # Second entry
    r1 = results[1]
    assert r1.arxiv_id == "2302.67890v1"
    assert r1.title == "Short Note"
    assert r1.authors == "Carol White"
    assert r1.abstract == "Brief."


@patch("app.harvesters.arxiv.requests.get")
def test_search_with_categories(mock_get):
    """search() should include category filter in the query string."""
    mock_get.return_value = _mock_response(_SAMPLE_ATOM_XML)

    h = ArxivHarvester()
    h.search("transformer", max_results=5, categories=["cs.LG", "cs.AI"])

    # Verify the API was called with the category filter in search_query
    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params")
    assert "cat:cs.LG" in params["search_query"]
    assert "cat:cs.AI" in params["search_query"]


def test_search_empty_query_raises():
    """search() should raise ValueError on empty query."""
    h = ArxivHarvester()
    with pytest.raises(ValueError, match="query is empty"):
        h.search("")

    with pytest.raises(ValueError, match="query is empty"):
        h.search("   ")


def test_search_bounds_check():
    """search() should raise ValueError for out-of-range max_results."""
    h = ArxivHarvester()
    with pytest.raises(ValueError, match="max_results"):
        h.search("anything", max_results=0)
    with pytest.raises(ValueError, match="max_results"):
        h.search("anything", max_results=101)


# ── fetch_abstract() ──────────────────────────────────────────────

@patch("app.harvesters.arxiv.requests.get")
def test_fetch_abstract(mock_get):
    """fetch_abstract() returns the abstract text for a known paper."""
    mock_get.return_value = _mock_response(_SINGLE_ENTRY_XML)

    h = ArxivHarvester()
    abstract = h.fetch_abstract("2301.12345v2")

    assert abstract is not None
    assert "comprehensive survey" in abstract.lower()


@patch("app.harvesters.arxiv.requests.get")
def test_fetch_abstract_no_entry(mock_get):
    """fetch_abstract() returns None when the API returns no entry."""
    empty_feed = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<title>ArXiv Query</title>"
        "<id>http://arxiv.org/api/query</id>"
        "</feed>"
    )
    mock_get.return_value = _mock_response(empty_feed)

    h = ArxivHarvester()
    assert h.fetch_abstract("9999.99999") is None


# ── harvest() abstract mode ───────────────────────────────────────

@patch("app.harvesters.arxiv.requests.get")
def test_harvest_abstract_mode(mock_get, tmp_path):
    """harvest() in abstract mode writes .txt files with provenance headers."""
    mock_get.return_value = _mock_response(_SAMPLE_ATOM_XML)

    h = ArxivHarvester()
    report = h.harvest(
        "transformer",
        max_results=10,
        mode="abstract",
        output_dir=str(tmp_path),
        min_chars=20,  # Low threshold so both entries qualify
    )

    assert isinstance(report, ArxivHarvestReport)
    assert report.query == "transformer"
    assert report.max_requested == 10

    # First paper (long abstract) should be harvested
    assert len(report.harvested) >= 1
    rec = report.harvested[0]
    assert rec.arxiv_id == "2301.12345v2"
    assert rec.mode == "abstract"
    assert rec.char_count > 0

    # File should exist and contain provenance header
    fpath = Path(rec.file_path)
    assert fpath.exists()
    content = fpath.read_text(encoding="utf-8")
    assert "# Attention Is All You Need" in content
    assert "# Authors: Alice Smith, Bob Jones" in content
    assert "# arXiv ID: 2301.12345v2" in content
    assert "# Categories:" in content
    assert "# Published:" in content
    assert "# PDF:" in content
    # Body text
    assert "comprehensive survey" in content.lower()
    # Filename convention
    assert fpath.name.startswith("2301_12345v2_")
    assert fpath.suffix == ".txt"


@patch("app.harvesters.arxiv.requests.get")
def test_harvest_skips_short_abstracts(mock_get, tmp_path):
    """Papers with abstract shorter than min_chars are skipped."""
    mock_get.return_value = _mock_response(_SHORT_ONLY_ATOM_XML)

    h = ArxivHarvester()
    report = h.harvest(
        "short note",
        max_results=5,
        mode="abstract",
        output_dir=str(tmp_path),
        min_chars=200,
    )

    assert len(report.harvested) == 0
    assert len(report.skipped) == 1
    assert "too short" in report.skipped[0]["reason"].lower()
    assert report.skipped[0]["arxiv_id"] == "2302.67890v1"


@patch("app.harvesters.arxiv.requests.get")
def test_harvest_mixed_skip_and_harvest(mock_get, tmp_path):
    """With default min_chars=200, the long paper is harvested, the short one skipped."""
    mock_get.return_value = _mock_response(_SAMPLE_ATOM_XML)

    h = ArxivHarvester()
    report = h.harvest(
        "transformer",
        max_results=10,
        mode="abstract",
        output_dir=str(tmp_path),
        min_chars=200,
    )

    assert len(report.harvested) == 1
    assert report.harvested[0].arxiv_id == "2301.12345v2"

    assert len(report.skipped) == 1
    assert report.skipped[0]["arxiv_id"] == "2302.67890v1"
    assert "too short" in report.skipped[0]["reason"].lower()


# ── harvest() full mode ───────────────────────────────────────────

@patch("app.harvesters.arxiv.requests.get")
def test_harvest_full_mode(mock_get, tmp_path):
    """harvest() in full mode downloads PDFs."""
    # First call returns the search XML, subsequent calls return fake PDF bytes
    fake_pdf = b"%PDF-1.4 fake pdf content for testing"
    search_resp = _mock_response(_SAMPLE_ATOM_XML)
    pdf_resp = MagicMock()
    pdf_resp.content = fake_pdf
    pdf_resp.raise_for_status = MagicMock()

    mock_get.side_effect = [search_resp, pdf_resp, pdf_resp]

    h = ArxivHarvester()
    report = h.harvest(
        "transformer",
        max_results=10,
        mode="full",
        output_dir=str(tmp_path),
    )

    assert len(report.harvested) == 2
    for rec in report.harvested:
        assert rec.mode == "full"
        fpath = Path(rec.file_path)
        assert fpath.exists()
        assert fpath.suffix == ".pdf"
        assert fpath.read_bytes() == fake_pdf


def test_harvest_invalid_mode():
    """harvest() should reject unknown modes."""
    h = ArxivHarvester()
    with pytest.raises(ValueError, match="mode must be"):
        h.harvest("anything", mode="unknown")


# ── Filename patterns ─────────────────────────────────────────────

@patch("app.harvesters.arxiv.requests.get")
def test_filename_sanitisation(mock_get, tmp_path):
    """Filenames should not contain dots from the arXiv ID or unsafe chars."""
    mock_get.return_value = _mock_response(_SAMPLE_ATOM_XML)

    h = ArxivHarvester()
    report = h.harvest(
        "transformer",
        max_results=10,
        mode="abstract",
        output_dir=str(tmp_path),
        min_chars=20,
    )

    for rec in report.harvested:
        fname = Path(rec.file_path).name
        # No slashes or path separators in the filename
        assert "/" not in fname and "\\" not in fname
        # arXiv ID dots should be replaced with underscores
        assert not fname.startswith("2301.12345")
        assert fname.startswith("2301_12345")
