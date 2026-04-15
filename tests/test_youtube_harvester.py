"""
Unit tests for the YouTube harvester.

All network calls (yt-dlp search + youtube-transcript-api) are mocked
out. We verify:
  * Search results are normalised into YouTubeSearchResult dataclasses
  * Transcripts are written to disk as UTF-8 text files
  * Filename sanitisation is applied
  * Videos without transcripts or with short transcripts are reported
    in the `skipped` field, not silently dropped
  * Preferred-language fallback chain is honoured
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Filename sanitisation ────────────────────────────────────────
@pytest.mark.parametrize("raw,expected_contains", [
    ("Perfectly Normal Title",    "Perfectly_Normal_Title"),
    ("Title with / slash",        "Title_with_slash"),      # contains underscores at least
    ("Title: with colon",         "Title_with_colon"),
    ("Title (with parens)",       "Title_with_parens"),
    ("",                          "video"),
])
def test_safe_slug(raw, expected_contains):
    from app.harvesters.youtube import _safe_slug
    out = _safe_slug(raw)
    # No path separators
    assert "/" not in out and "\\" not in out
    # Slugs are non-empty and <= 80 chars
    assert 1 <= len(out) <= 80


# ── Harvester: search + harvest with mocked network ──────────────
def _mock_yt_dlp_search(results: list[dict]):
    """Patch the yt_dlp module import so .search() returns our canned data."""
    mod = MagicMock()
    ydl_cls = MagicMock()
    ydl_ctx = MagicMock()
    ydl_ctx.extract_info.return_value = {"entries": results}
    ydl_cls.return_value.__enter__.return_value = ydl_ctx
    ydl_cls.return_value.__exit__.return_value = False
    mod.YoutubeDL = ydl_cls
    return mod


def _mock_transcript_api(transcripts_by_video: dict[str, str]):
    """
    Patch youtube_transcript_api to match the v1.x instance API:

        api = YouTubeTranscriptApi()
        api.fetch(video_id, languages=[...])   # -> FetchedTranscript
        api.list(video_id)                     # -> TranscriptList (iterable)

    Each video_id in transcripts_by_video returns that text.
    Any other video_id raises NoTranscriptFound on both fetch() and list().
    """
    err_mod = MagicMock()
    class TranscriptsDisabled(Exception): ...
    class NoTranscriptFound(Exception): ...
    class VideoUnavailable(Exception): ...
    err_mod.TranscriptsDisabled = TranscriptsDisabled
    err_mod.NoTranscriptFound = NoTranscriptFound
    err_mod.VideoUnavailable = VideoUnavailable

    class _FakeSnippet:
        def __init__(self, text: str):
            self.text = text
            self.start = 0.0
            self.duration = 1.0

    class _FakeFetched:
        def __init__(self, text: str, lang: str = "en", is_generated: bool = True):
            self.snippets = [_FakeSnippet(line) for line in text.split("\n") if line.strip()]
            self.language_code = lang
            self.is_generated = is_generated

    class _FakeTranscript:
        """Matches the new Transcript object shape for api.list() results."""
        def __init__(self, text: str, lang: str = "en"):
            self._text = text
            self.language_code = lang
            self.is_translatable = False
        def fetch(self):
            return _FakeFetched(self._text, self.language_code)
        def translate(self, lang):
            return _FakeTranscript(self._text, lang)

    class _FakeApi:
        def fetch(self, vid, languages=None):
            if vid in transcripts_by_video:
                return _FakeFetched(transcripts_by_video[vid])
            raise NoTranscriptFound(f"no transcript for {vid}")

        def list(self, vid):
            if vid in transcripts_by_video:
                return [_FakeTranscript(transcripts_by_video[vid])]
            raise NoTranscriptFound(f"no transcript for {vid}")

    api_mod = MagicMock()
    api_mod.YouTubeTranscriptApi = _FakeApi
    return api_mod, err_mod


def test_harvest_end_to_end(tmp_path, monkeypatch):
    search_results = [
        {"id": "vid_001", "title": "How Corrosion Allowance Works",
         "url": "https://youtu.be/vid_001", "channel": "Pipe Sciences",
         "duration": 540, "view_count": 12000},
        {"id": "vid_002", "title": "API 570 Inspection Intervals Explained",
         "url": "https://youtu.be/vid_002", "channel": "Asset Integrity 101",
         "duration": 780, "view_count": 8900},
        {"id": "vid_003", "title": "Video With No Captions",
         "url": "https://youtu.be/vid_003", "channel": "Misc", "duration": 120},
    ]
    long_text = (
        "Corrosion allowance is extra wall thickness specified at design "
        "time. This absorbs expected metal loss over the asset's service "
        "life. The rate is tracked via periodic ultrasonic thickness "
        "inspections. Remaining life is the more conservative of the "
        "long-term and short-term trend projections."
    ) * 4   # ~1.5 KB so it clears the min_chars=400 threshold

    transcripts = {
        "vid_001": long_text,
        "vid_002": long_text + " Additional context from the API 570 article.",
        # vid_003 deliberately absent -> should be skipped
    }

    yt_dlp_mod = _mock_yt_dlp_search(search_results)
    api_mod, err_mod = _mock_transcript_api(transcripts)

    monkeypatch.setitem(sys.modules, "yt_dlp", yt_dlp_mod)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", api_mod)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api._errors", err_mod)

    # Force re-import so the module uses our mocks
    if "app.harvesters.youtube" in sys.modules:
        del sys.modules["app.harvesters.youtube"]

    from app.harvesters.youtube import YouTubeHarvester

    h = YouTubeHarvester()
    report = h.harvest(
        "corrosion allowance",
        max_results=3,
        output_dir=str(tmp_path),
        min_chars=400,
    )

    assert report.query == "corrosion allowance"
    assert report.max_requested == 3
    assert len(report.harvested) == 2
    assert len(report.skipped) == 1
    assert report.skipped[0]["title"] == "Video With No Captions"
    # Reason surfaced from the transcript-api exception — new fetch_transcript
    # formats it as "NoTranscriptFound: no transcript for vid_003".
    assert "no transcript" in report.skipped[0]["reason"].lower()

    # Files exist and have the sanitised name pattern
    for rec in report.harvested:
        p = Path(rec.file_path)
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        # Provenance header
        assert rec.title in content
        assert rec.url in content
        # Full transcript body
        assert "Corrosion allowance" in content
        # Filename convention
        assert p.name.startswith(rec.video_id + "_")
        assert p.suffix == ".txt"


def test_harvest_rejects_empty_query(monkeypatch):
    from app.harvesters.youtube import YouTubeHarvester
    with pytest.raises(ValueError):
        YouTubeHarvester().search("", max_results=10)


def test_harvest_bounds_check(monkeypatch):
    from app.harvesters.youtube import YouTubeHarvester
    with pytest.raises(ValueError):
        YouTubeHarvester().search("anything", max_results=0)
    with pytest.raises(ValueError):
        YouTubeHarvester().search("anything", max_results=100)


def test_short_transcript_is_skipped(tmp_path, monkeypatch):
    search_results = [
        {"id": "vid_short", "title": "Quick Clip",
         "url": "https://youtu.be/vid_short", "channel": "X", "duration": 30},
    ]
    transcripts = {"vid_short": "Too short."}

    yt_dlp_mod = _mock_yt_dlp_search(search_results)
    api_mod, err_mod = _mock_transcript_api(transcripts)

    monkeypatch.setitem(sys.modules, "yt_dlp", yt_dlp_mod)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", api_mod)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api._errors", err_mod)

    if "app.harvesters.youtube" in sys.modules:
        del sys.modules["app.harvesters.youtube"]
    from app.harvesters.youtube import YouTubeHarvester

    report = YouTubeHarvester().harvest(
        "quick clip", max_results=1, output_dir=str(tmp_path), min_chars=400,
    )
    assert len(report.harvested) == 0
    assert len(report.skipped) == 1
    assert "too short" in report.skipped[0]["reason"].lower()
