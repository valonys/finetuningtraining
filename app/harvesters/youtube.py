"""
app/harvesters/youtube.py
─────────────────────────
Keyword -> Top-N videos -> Transcripts -> .txt files in ./data/uploads/.

Design (why captions-only, not Whisper STT):

  The upstream `valonys/youTubeClaw` uses yt-dlp + faster-whisper. That's
  the gold-standard path when you need accurate transcripts for videos
  without captions, but it's HEAVY:

      - Downloads the full audio of each video (10-100 MB each)
      - Requires ffmpeg on PATH
      - Requires a Whisper model (~140 MB first download)
      - Transcribes at roughly 1-2x realtime on CPU (so 10 hours of video
        = 5-10 hours of transcription)

  For an interactive Studio demo where users expect "hit the button, get
  training data in under a minute", that's the wrong trade-off. This
  module uses the captions-only path:

      1. yt-dlp for keyword search (same virtual `ytsearch{N}:` URL trick
         upstream uses — no API key, no quota).
      2. youtube-transcript-api to fetch pre-existing captions (auto or
         human) for each video. One HTTPS call per video, no audio.
      3. Skip videos that have no captions — log a warning.

  Result: 10 videos harvested in ~20 seconds instead of ~20 minutes.

  When users need a video without captions, they can fall back to the
  upstream youTubeClaw Whisper path in a future commit; this module's
  interface is designed to accommodate a `strategy="whisper"` flag.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Dataclasses ─────────────────────────────────────────────────────
@dataclass
class YouTubeSearchResult:
    video_id: str
    title: str
    url: str
    channel: str = ""
    duration_s: int = 0
    view_count: int = 0


@dataclass
class HarvestedTranscript:
    video_id: str
    title: str
    url: str
    channel: str
    language: str
    auto_generated: bool
    text: str
    file_path: str
    char_count: int = 0
    duration_s: int = 0


@dataclass
class HarvestReport:
    harvested: list[HarvestedTranscript] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)   # {title, url, reason}
    query: str = ""
    max_requested: int = 0


# ── Filename sanitisation ──────────────────────────────────────────
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_slug(title: str, maxlen: int = 80) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", title).strip("._")
    return (cleaned or "video")[:maxlen]


# ── Harvester ───────────────────────────────────────────────────────
class YouTubeHarvester:
    """
    Keyword search + caption fetch -> plain-text files in data/uploads/.

    Typical use:

        h = YouTubeHarvester()
        report = h.harvest("asset integrity inspection", max_results=10)
        for rec in report.harvested:
            print(rec.title, "->", rec.file_path)

    The files land in `./data/uploads/` so the Data Forge picks them up
    via the standard `ingest(paths)` / `forge/build_dataset` flow.
    Filenames are `<video_id>_<sanitised_title>.txt`.
    """

    def __init__(
        self,
        *,
        preferred_languages: Optional[list[str]] = None,
        user_agent: Optional[str] = None,
    ):
        self.preferred_languages = preferred_languages or ["en", "en-US", "en-GB"]
        self.user_agent = user_agent

    # ── Public API ─────────────────────────────────────────────
    def search(self, query: str, max_results: int = 10) -> list[YouTubeSearchResult]:
        """
        Search YouTube by keyword. Uses yt-dlp's `ytsearch{N}:<query>`
        virtual URL -- no YouTube Data API key required, no quota.
        """
        if not query.strip():
            raise ValueError("YouTubeHarvester.search: query is empty")
        if max_results < 1 or max_results > 50:
            raise ValueError("max_results must be in [1, 50]")

        try:
            import yt_dlp
        except ImportError as e:
            raise RuntimeError(
                "yt-dlp not installed. Add `yt-dlp>=2024.12` to your env."
            ) from e

        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
            "ignoreerrors": True,
        }
        if self.user_agent:
            opts["user_agent"] = self.user_agent

        search_url = f"ytsearch{max_results}:{query}"
        logger.info(f"🔍 YouTube search: {search_url!r}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(search_url, download=False)

        entries = (info or {}).get("entries") or []
        out: list[YouTubeSearchResult] = []
        for e in entries:
            if not e:
                continue
            vid = e.get("id") or e.get("video_id")
            if not vid:
                continue
            out.append(YouTubeSearchResult(
                video_id=vid,
                title=(e.get("title") or vid),
                url=e.get("url") or f"https://www.youtube.com/watch?v={vid}",
                channel=e.get("channel") or e.get("uploader") or "",
                duration_s=int(e.get("duration") or 0),
                view_count=int(e.get("view_count") or 0),
            ))
        logger.info(f"   {len(out)} results")
        return out

    def fetch_transcript(
        self, video_id: str
    ) -> Optional[tuple[str, str, bool]]:
        """
        Return (plain_text, language, auto_generated) for a single video,
        or None if no caption track is available in a preferred language.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import (
                TranscriptsDisabled,
                NoTranscriptFound,
                VideoUnavailable,
            )
        except ImportError as e:
            raise RuntimeError(
                "youtube-transcript-api not installed. "
                "Add `youtube-transcript-api>=0.6` to your env."
            ) from e

        try:
            listing = YouTubeTranscriptApi.list_transcripts(video_id)
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            logger.debug(f"   no transcript for {video_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"   transcript lookup failed for {video_id}: {e}")
            return None

        # Prefer human-edited caption tracks in a preferred language,
        # then auto-generated in a preferred language, then any translatable.
        chosen = None
        auto = False
        for lang in self.preferred_languages:
            try:
                chosen = listing.find_manually_created_transcript([lang])
                auto = False
                break
            except Exception:
                pass
        if chosen is None:
            for lang in self.preferred_languages:
                try:
                    chosen = listing.find_generated_transcript([lang])
                    auto = True
                    break
                except Exception:
                    pass
        if chosen is None:
            # Last resort: any available transcript, translated to English
            try:
                any_transcript = next(iter(listing))
                chosen = any_transcript.translate("en")
                auto = getattr(any_transcript, "is_generated", True)
            except Exception:
                return None

        try:
            entries = chosen.fetch()
        except Exception as e:
            logger.warning(f"   transcript fetch failed for {video_id}: {e}")
            return None

        text_parts = []
        for entry in entries:
            t = (entry.get("text") or "").strip()
            if t and t != "[Music]" and t != "[Applause]":
                text_parts.append(t)
        if not text_parts:
            return None
        text = " ".join(text_parts)
        # Collapse double spaces + stray newline fragments
        text = re.sub(r"\s+", " ", text).strip()
        return text, getattr(chosen, "language_code", "en"), auto

    def harvest(
        self,
        query: str,
        *,
        max_results: int = 10,
        output_dir: str = "./data/uploads",
        min_chars: int = 400,
    ) -> HarvestReport:
        """
        End-to-end: search -> fetch transcripts -> write .txt files.

        `min_chars` filters out toy transcripts (announcements, ads) that
        would pollute the Q/A synth with no-content chunks.
        """
        report = HarvestReport(query=query, max_requested=max_results)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        hits = self.search(query, max_results=max_results)

        for h in hits:
            result = self.fetch_transcript(h.video_id)
            if result is None:
                report.skipped.append({
                    "title": h.title,
                    "url": h.url,
                    "reason": "no transcript available",
                })
                continue
            text, lang, auto = result
            if len(text) < min_chars:
                report.skipped.append({
                    "title": h.title,
                    "url": h.url,
                    "reason": f"transcript too short ({len(text)} chars < {min_chars})",
                })
                continue

            slug = _safe_slug(h.title)
            fname = f"{h.video_id}_{slug}.txt"
            fpath = out_dir / fname

            # Prepend provenance header so downstream ingestion has
            # something to attribute the content to in citations.
            header = (
                f"# {h.title}\n"
                f"# Channel: {h.channel}\n"
                f"# URL: {h.url}\n"
                f"# Language: {lang} ({'auto' if auto else 'human'})\n"
                f"\n"
            )
            fpath.write_text(header + text, encoding="utf-8")
            logger.info(f"   ✓ {h.title[:60]}  -> {fname}  ({len(text)} chars)")

            report.harvested.append(HarvestedTranscript(
                video_id=h.video_id,
                title=h.title,
                url=h.url,
                channel=h.channel,
                language=lang,
                auto_generated=auto,
                text=text,
                file_path=str(fpath),
                char_count=len(text),
                duration_s=h.duration_s,
            ))

        return report
