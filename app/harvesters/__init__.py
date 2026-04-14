"""
app/harvesters/
───────────────
Content harvesters that fetch external material and deposit it into
`./data/uploads/` so the Data Forge can ingest + chunk + Q/A synthesize
it exactly like a file the user drag-dropped.

Currently shipped:
    YouTubeHarvester  — keyword search + caption fetch
                        (credits upstream: valonys/youTubeClaw for the
                        yt-dlp search pattern; transcript path here is
                        captions-only via youtube-transcript-api, not
                        Whisper STT, to keep Studio demos snappy)
"""
from .youtube import (
    HarvestedTranscript,
    YouTubeHarvester,
    YouTubeSearchResult,
)

__all__ = [
    "HarvestedTranscript",
    "YouTubeHarvester",
    "YouTubeSearchResult",
]
