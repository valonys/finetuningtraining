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


# ── v2 academic / book noise rules ────────────────────────────────
def test_rejects_numbered_reference_list():
    from app.data_forge.chunk_filter import is_noise
    refs = (
        "[1] Vaswani, A., Shazeer, N., et al. Attention is all you need. "
        "NeurIPS, 2017.\n"
        "[2] Devlin, J., Chang, M.-W., et al. BERT: Pre-training of deep "
        "bidirectional transformers. NAACL, 2019.\n"
        "[3] Brown, T., Mann, B., et al. Language models are few-shot "
        "learners. NeurIPS, 2020.\n"
        "[4] Hu, E., et al. LoRA: Low-rank adaptation of large language "
        "models. ICLR, 2022.\n"
        "[5] Rafailov, R., et al. Direct preference optimization. "
        "NeurIPS, 2023.\n"
        "[6] Shao, Z., et al. DeepSeekMath: Pushing the limits of "
        "mathematical reasoning. arXiv, 2024."
    )
    noisy, reason = is_noise(refs)
    assert noisy
    assert reason in ("bibliography", "front_matter_signature", "short_sentences")


def test_rejects_acknowledgements_section():
    from app.data_forge.chunk_filter import is_noise
    ack = (
        "Acknowledgements\n\n"
        "This work was supported by the National Science Foundation "
        "under grant IIS-2023456. The authors thank the anonymous "
        "reviewers for their constructive feedback. We also thank "
        "the compute team at XYZ Lab for providing GPU resources."
    )
    noisy, reason = is_noise(ack)
    assert noisy
    assert reason in ("acknowledgements", "front_matter_signature")


def test_rejects_figure_caption_only_chunk():
    from app.data_forge.chunk_filter import is_noise
    cap = "Figure 3: Architecture of the transformer encoder-decoder model."
    noisy, reason = is_noise(cap)
    # Short caption is caught by too_short or figure_caption
    assert noisy
    assert reason in ("too_short", "figure_caption")


def test_rejects_index_page():
    from app.data_forge.chunk_filter import is_noise
    idx = (
        "attention mechanism, 42, 105, 312\n"
        "backpropagation, 18, 23, 67\n"
        "convolutional layers, 88, 91\n"
        "dropout regularization, 55, 78\n"
        "embedding dimension, 34, 102\n"
        "feed-forward network, 43, 109\n"
        "gradient descent, 12, 15, 29\n"
        "hidden states, 36, 48, 110\n"
        "inference latency, 200, 215\n"
        "key-value cache, 188, 195, 210\n"
        "layer normalization, 40, 44\n"
        "multi-head attention, 42, 106"
    )
    noisy, reason = is_noise(idx)
    assert noisy
    assert reason in ("index_page", "short_sentences", "digit_dense")


def test_rejects_citation_heavy_paragraph():
    from app.data_forge.chunk_filter import is_noise
    cites = (
        "Several approaches have been proposed [1,2,3] for efficient "
        "fine-tuning [4,5]. Recent work [6,7,8,9] builds on earlier "
        "results [10,11] by combining LoRA [12] with quantization "
        "[13,14,15]. Further improvements [16,17,18,19,20] demonstrate "
        "that [21,22] the scaling laws [23,24,25] predict [26,27,28] "
        "the optimal [29,30] configuration [31,32,33,34,35,36]."
    )
    noisy, reason = is_noise(cites)
    assert noisy
    assert reason in ("citation_cluster", "too_few_sentences", "digit_dense")


def test_rejects_latex_artifact_chunk():
    from app.data_forge.chunk_filter import is_noise
    latex = (
        "\\begin{equation}\n"
        "\\textbf{h}_i = \\textit{Attention}(Q, K, V)\n"
        "\\end{equation}\n"
        "\\label{eq:attention}\n"
        "\\ref{fig:architecture}\n"
        "\\cite{vaswani2017attention}\n"
        "\\section{Background}\n"
        "\\subsection{Related Work}\n"
        "\\begin{theorem}\n"
        "\\end{theorem}"
    )
    noisy, reason = is_noise(latex)
    assert noisy
    assert reason in ("latex_artifacts", "short_sentences", "too_few_sentences")


def test_keeps_academic_prose_with_occasional_citations():
    """Real academic content with a few citations should NOT be rejected."""
    from app.data_forge.chunk_filter import is_noise
    content = (
        "Reinforcement Learning from Verifiable Rewards (RLVR) is a "
        "training paradigm where the reward signal comes from a "
        "deterministic verifier rather than a learned reward model. "
        "The key insight is that for domains with ground-truth answers "
        "(mathematics, code execution, formal logic), the reward can "
        "be computed exactly by checking the model's output against "
        "the known solution. This eliminates reward hacking, a "
        "persistent problem in RLHF where the policy learns to exploit "
        "artifacts in the reward model rather than genuinely improving "
        "response quality."
    )
    noisy, reason = is_noise(content)
    assert not noisy, f"Academic prose wrongly flagged: {reason}"
