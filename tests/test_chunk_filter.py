"""
Tests for app.data_forge.chunk_filter — the noise filter that rejects
book cover pages, TOCs, indexes, and bibliography fragments before
Q/A synthesis runs.
"""
from __future__ import annotations

import pytest


# ── Individual check tests ────────────────────────────────────────
def test_digit_ratio_rejects_page_number_strips():
    from app.data_forge.chunk_filter import is_noise
    page_strip = "Page 1  Page 2  Page 3  Page 4  Page 5  Page 6  Page 7 " * 10
    noisy, reason = is_noise(page_strip)
    assert noisy
    assert reason in ("digit_dense", "toc_like", "short_sentences", "too_few_sentences")


def test_rejects_all_caps_title_page():
    from app.data_forge.chunk_filter import is_noise
    title_page = (
        "FUNDAMENTALS OF ASSET INTEGRITY MANAGEMENT\n\n"
        "FOR OFFSHORE OPERATIONS AND PIPELINE SYSTEMS\n\n"
        "AUTHORED BY JANE DOE AND JOHN SMITH\n\n"
        "PUBLISHED BY TECHNICAL PRESS INTERNATIONAL\n\n"
        "FIRST EDITION TWENTY TWENTY FOUR"
    )
    noisy, reason = is_noise(title_page)
    assert noisy
    assert reason == "all_caps"


def test_rejects_short_chunks():
    from app.data_forge.chunk_filter import is_noise
    noisy, reason = is_noise("Too short.")
    assert noisy
    assert reason == "too_short"


def test_rejects_toc_entries():
    from app.data_forge.chunk_filter import is_noise
    toc = (
        "Chapter 1 Introduction ..... 1\n"
        "Chapter 2 Background ..... 14\n"
        "Chapter 3 Methods ..... 28\n"
        "Chapter 4 Results ..... 56\n"
        "Chapter 5 Discussion ..... 82\n"
        "Chapter 6 Conclusion ..... 105\n"
        "Appendix A References ..... 120\n"
        "Appendix B Glossary ..... 135\n"
    )
    noisy, reason = is_noise(toc)
    assert noisy
    assert reason in ("toc_like", "front_matter_signature", "short_sentences")


def test_rejects_bibliography_fragment():
    from app.data_forge.chunk_filter import is_noise
    bib = (
        "Smith, J. (2020). Understanding corrosion in offshore pipelines. "
        "Journal of Asset Integrity, 12(3), 45-67.\n"
        "Doe, A. (2019). Inspection programs for FPSO operations. "
        "Offshore Engineering Review, 8(2), 112-130.\n"
        "Jones, B. (2021). API 570 compliance strategies. "
        "Pressure Vessel Quarterly, 15(4), 78-95."
    )
    noisy, reason = is_noise(bib)
    assert noisy
    # Bibliography citations are punctuation-dense (many periods inside
    # each entry -> many "sentences" that are each just a few words),
    # so the short_sentences check frequently fires first. Any of these
    # rejection reasons is a correct outcome.
    assert reason in (
        "front_matter_signature",
        "front_matter",
        "digit_dense",
        "short_sentences",
    )


def test_rejects_isbn_copyright_block():
    from app.data_forge.chunk_filter import is_noise
    copyright_block = (
        "Copyright (c) 2024 Technical Press International. "
        "All rights reserved. ISBN 978-1-234-56789-0. "
        "Published first in twenty twenty four by Technical Press. "
        "Printed on acid-free paper. No part of this publication may "
        "be reproduced without written permission from the publisher."
    )
    noisy, reason = is_noise(copyright_block)
    assert noisy
    assert reason in ("front_matter_signature", "front_matter")


def test_keeps_real_content():
    from app.data_forge.chunk_filter import is_noise
    content = (
        "The corrosion allowance in pressure vessels is an additional wall "
        "thickness specified at design time to accommodate expected metal "
        "loss over the service life. API 510 defines standard allowances "
        "for different service environments, ranging from 1/16 inch for "
        "mild service to 1/4 inch for severe service. The corrosion rate "
        "is calculated from periodic thickness measurements taken during "
        "scheduled inspections, with both long-term and short-term rates "
        "trended separately."
    )
    noisy, reason = is_noise(content)
    assert not noisy, f"Real content wrongly flagged: {reason}"


def test_keeps_technical_narrative_with_some_numbers():
    from app.data_forge.chunk_filter import is_noise
    content = (
        "In 2023, a study of 47 offshore platforms revealed that 32% "
        "experienced pipeline corrosion events exceeding the design "
        "allowance. The median time to first incident was 8.4 years, "
        "with the 90th percentile at 12.1 years. These findings suggest "
        "that the conventional 3.2 mm corrosion allowance may be "
        "insufficient for platforms operating in high-salinity tropical "
        "waters, where accelerated degradation reduces the effective "
        "service margin by roughly 40 percent compared to temperate "
        "installations."
    )
    noisy, reason = is_noise(content)
    assert not noisy, f"Technical narrative wrongly flagged: {reason}"


# ── Batch filter ────────────────────────────────────────────────
def test_filter_chunks_returns_stats():
    from app.data_forge.chunk_filter import filter_chunks
    chunks = [
        {"chunk": "Chapter 1 ..... 1\nChapter 2 ..... 14\nChapter 3 ..... 28\n", "source": "a"},  # TOC
        {"chunk": "Too short.", "source": "b"},                                                    # too short
        {"chunk": "The corrosion rate is measured from periodic thickness readings. "
                  "These readings are taken at established condition monitoring locations. "
                  "The remaining life calculation uses the more conservative of the "
                  "long-term and short-term trends as the governing value.", "source": "c"},  # good
    ]
    kept, stats = filter_chunks(chunks)
    assert len(kept) == 1
    assert kept[0]["source"] == "c"
    assert stats["total"] == 3
    assert stats["kept"] == 1
    assert stats["dropped_count"] == 2
    # At least one reason was recorded
    assert sum(stats["reasons"].values()) == 2
