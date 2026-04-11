"""HTML parser: prefers trafilatura (boilerplate stripping) then BS4 fallback."""
from __future__ import annotations

from ..ingest import IngestedRecord


def parse_html(path: str, **_kw) -> IngestedRecord:
    with open(path, "rb") as f:
        raw = f.read()

    # Trafilatura first — removes navigation, ads, footer, etc.
    text: str | None = None
    try:
        import trafilatura
        text = trafilatura.extract(
            raw.decode("utf-8", errors="replace"),
            include_comments=False,
            include_tables=True,
        )
    except Exception:
        text = None

    if not text:
        # BS4 fallback
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw, "html.parser")
            for bad in soup(["script", "style", "noscript"]):
                bad.decompose()
            text = soup.get_text("\n").strip()
        except Exception:
            text = raw.decode("utf-8", errors="replace")

    return IngestedRecord(
        source=path,
        source_type="html",
        text=text or "",
        metadata={"bytes": len(raw)},
    )
