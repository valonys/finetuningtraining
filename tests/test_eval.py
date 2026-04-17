"""
Unit tests for the eval module.

We test the scorer and judge with fakes — no model loading, no Ollama
calls. The runner integrates the two but we exercise that in notebooks
and the /v1/eval endpoint (which needs a live model).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.scorer import eval_qa_accuracy, _check_match
from app.eval.judge import _parse_verdict


# ── QA accuracy scorer ───────────────────────────────────────────
def test_qa_accuracy_perfect_score(tmp_path):
    bank = tmp_path / "qa.jsonl"
    rows = [
        {"instruction": "What is 2+2?", "response": "4"},
        {"instruction": "Capital of France?", "response": "Paris"},
    ]
    bank.write_text("\n".join(json.dumps(r) for r in rows))

    def gen(prompt):
        if "2+2" in prompt:
            return "The answer is 4."
        return "The capital is Paris."

    result = eval_qa_accuracy(gen, str(bank))
    assert result["accuracy"] == 1.0
    assert result["correct"] == 2
    assert result["total"] == 2
    assert result["wrong"] == []


def test_qa_accuracy_partial_score(tmp_path):
    bank = tmp_path / "qa.jsonl"
    rows = [
        {"instruction": "What is 2+2?", "response": "4"},
        {"instruction": "Capital of France?", "response": "Paris"},
    ]
    bank.write_text("\n".join(json.dumps(r) for r in rows))

    def gen(prompt):
        return "I don't know"  # wrong for both

    result = eval_qa_accuracy(gen, str(bank))
    assert result["accuracy"] == 0.0
    assert result["correct"] == 0
    assert len(result["wrong"]) == 2


def test_qa_accuracy_missing_file():
    result = eval_qa_accuracy(lambda p: "x", "/nope/does/not/exist.jsonl")
    assert result["total"] == 0


# ── Match modes ──────────────────────────────────────────────────
def test_substring_match():
    assert _check_match("The answer is 42, clearly.", "42", "substring")
    assert not _check_match("The answer is 43.", "42", "substring")


def test_exact_match():
    assert _check_match("42", "42", "exact")
    assert not _check_match("The answer is 42", "42", "exact")


# ── Judge verdict parser ────────────────────────────────────────
def test_parse_verdict_clean_json():
    raw = '{"winner": "A", "score_a": 5, "score_b": 2, "reason": "more detailed"}'
    v = _parse_verdict(raw)
    assert v["winner"] == "A"
    assert v["score_a"] == 5
    assert v["score_b"] == 2


def test_parse_verdict_markdown_fenced():
    raw = '```json\n{"winner": "B", "score_a": 3, "score_b": 4, "reason": "better"}\n```'
    v = _parse_verdict(raw)
    assert v["winner"] == "B"


def test_parse_verdict_garbage():
    v = _parse_verdict("I think A is better because blah blah")
    assert v["winner"] == "tie"


def test_parse_verdict_invalid_winner():
    raw = '{"winner": "C", "score_a": 3, "score_b": 3, "reason": "confused"}'
    v = _parse_verdict(raw)
    assert v["winner"] == "tie"
