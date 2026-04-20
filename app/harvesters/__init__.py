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
    ArxivHarvester    — keyword search + abstract/PDF fetch from arXiv
                        (uses the arXiv Atom API directly via HTTP, no
                        external arxiv package required)
    CodeHarvester     — directory scanner for .py / .ipynb files
                        AST-based extraction of functions, classes, and
                        notebook cells into SFT instruction/response pairs
"""
from .arxiv import (
    ArxivHarvestedPaper,
    ArxivHarvester,
    ArxivHarvestReport,
    ArxivSearchResult,
)
from .code import (
    CodeHarvester,
    CodeHarvestReport,
    CodeUnit,
)
from .youtube import (
    HarvestedTranscript,
    YouTubeHarvester,
    YouTubeSearchResult,
)

__all__ = [
    "ArxivHarvestedPaper",
    "ArxivHarvester",
    "ArxivHarvestReport",
    "ArxivSearchResult",
    "CodeHarvester",
    "CodeHarvestReport",
    "CodeUnit",
    "HarvestedTranscript",
    "YouTubeHarvester",
    "YouTubeSearchResult",
]
