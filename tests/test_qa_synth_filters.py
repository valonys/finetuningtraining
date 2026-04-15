"""
Tests for the trivial-Q + thin-A post-filters in qa_synthesis.

These run on every LLM-synthesized pair before it makes it into the
dataset. Catching the "who wrote this book" / "how many chapters"
class of garbage at this layer is how we keep the output dataset
commercially-usable.
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("q,expected", [
    # Author / title / metadata trivia — should be rejected
    ("Who is the author of this book?",                                      True),
    ("Who wrote this article?",                                              True),
    ("Who is the illustrator of this publication?",                          True),
    ("What is the title of this book?",                                      True),
    ("What is the name of the article?",                                     True),
    ("How many chapters does the book have?",                                True),
    ("How many pages are in this document?",                                 True),
    ("When was this book published?",                                        True),
    ("What is the ISBN of this book?",                                       True),
    ("What does the table of contents list as section 3?",                   True),
    ("What is the copyright year?",                                          True),
    # Real questions — should NOT be rejected
    ("Explain how corrosion allowance is calculated in pressure vessels.",   False),
    ("What causes crevice corrosion in stainless steel pipelines?",          False),
    ("How does API 570 define critical thickness?",                          False),
    ("Compare SFT with DPO for preference alignment.",                       False),
    ("Why might a LoRA rank of 8 underperform on a 65K dataset?",            False),
])
def test_is_trivial_question(q, expected):
    from app.data_forge.qa_synthesis import _is_trivial_question
    assert _is_trivial_question(q) == expected, f"Misclassified: {q!r}"


@pytest.mark.parametrize("a,expected", [
    # Thin, non-teaching answers — should be rejected
    ("Yes.",                                                                  False),
    ("No.",                                                                   False),
    ("The answer is 12.",                                                     False),
    ("Chapter 5.",                                                            False),
    ("978-1-234-56789-0.",                                                    False),
    # Substantive, multi-sentence answers — should be kept
    (
        "Corrosion allowance is extra wall thickness built into the design "
        "to absorb expected metal loss over the service life. It is tracked "
        "via periodic inspections and consumed when thickness falls below "
        "the original plus allowance value.",
        True,
    ),
    (
        "API 570 specifies inspection intervals based on corrosion rate and "
        "service class. For Class 1 piping, the interval is five years or "
        "half the remaining life, whichever is shorter, with a maximum of "
        "ten years. Class 2 extends this to ten years or half remaining life.",
        True,
    ),
])
def test_is_substantive_answer(a, expected):
    from app.data_forge.qa_synthesis import _is_substantive_answer
    assert _is_substantive_answer(a) == expected, f"Misclassified: {a!r}"
