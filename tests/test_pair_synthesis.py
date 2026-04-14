"""
Unit tests for app.data_forge.pair_synthesis — DPO/ORPO contrastive pair builder.
No network: the provider is mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock


class _FakeProvider:
    name = "fake"
    model = "fake-model"
    base_url = "http://fake/v1"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        return self._responses.pop(0)


def test_rule_based_fallback_when_no_provider(monkeypatch):
    # Clear all env so get_synth_provider() returns None
    for k in ("OLLAMA_API_KEY", "OLLAMA_HOST", "OPENAI_API_KEY",
              "VALONY_SYNTH_BASE_URL", "VALONY_SYNTH_MODEL",
              "VALONY_SYNTH_PROVIDER"):
        monkeypatch.delenv(k, raising=False)

    from app.data_forge.pair_synthesis import synthesize_pairs

    seeds = [
        {"instruction": "Explain corrosion allowance.",
         "response": "The corrosion allowance is extra thickness added to a component to account for expected metal loss over its service life.",
         "source": "doc1.pdf"},
    ]
    out = synthesize_pairs(seeds)
    assert len(out) == 1
    row = out[0]
    assert row["instruction"] == seeds[0]["instruction"]
    assert row["chosen"] == seeds[0]["response"]
    assert row["rejected"] != row["chosen"]
    assert row["synth"] == "rule_based_weak"


def test_llm_pair_synth_valid_json():
    from app.data_forge.pair_synthesis import synthesize_pairs

    provider = _FakeProvider([
        '{"chosen": "Good explanation.", "rejected": "Wrong explanation that sounds plausible."}',
    ])
    out = synthesize_pairs(
        [{"instruction": "What is X?", "response": "X is Y."}],
        provider=provider,
    )
    assert len(out) == 1
    assert out[0]["chosen"] == "Good explanation."
    assert out[0]["rejected"] == "Wrong explanation that sounds plausible."
    assert out[0]["synth"] == "fake"
    # The user message should carry the instruction
    assert "What is X?" in provider.calls[0]["messages"][1]["content"]
    # Reference answer should have been embedded
    assert "X is Y." in provider.calls[0]["messages"][1]["content"]


def test_llm_pair_synth_unparsable_output_falls_back():
    from app.data_forge.pair_synthesis import synthesize_pairs

    provider = _FakeProvider(["this is not json at all"])
    out = synthesize_pairs(
        [{"instruction": "Q", "response": "A"}],
        provider=provider,
    )
    assert len(out) == 1
    assert out[0]["synth"] == "rule_based_weak"


def test_llm_pair_synth_json_in_code_fence():
    from app.data_forge.pair_synthesis import synthesize_pairs

    provider = _FakeProvider([
        'Here you go:\n```json\n{"chosen": "Right.", "rejected": "Wrong."}\n```',
    ])
    out = synthesize_pairs(
        [{"instruction": "Q", "response": "A"}],
        provider=provider,
    )
    assert len(out) == 1
    assert out[0]["chosen"] == "Right."
    assert out[0]["rejected"] == "Wrong."


def test_llm_pair_synth_drops_identical_pair():
    from app.data_forge.pair_synthesis import synthesize_pairs

    provider = _FakeProvider([
        '{"chosen": "same thing", "rejected": "same thing"}',
    ])
    out = synthesize_pairs(
        [{"instruction": "Q", "response": "A"}],
        provider=provider,
    )
    assert out == []


def test_llm_pair_synth_provider_error_falls_back():
    from app.data_forge.pair_synthesis import synthesize_pairs

    class _RaisingProvider:
        name = "raising"
        model = "m"
        base_url = "u"
        def chat(self, *a, **kw):
            raise RuntimeError("boom")

    out = synthesize_pairs(
        [{"instruction": "Q", "response": "A", "source": "s"}],
        provider=_RaisingProvider(),
    )
    assert len(out) == 1
    assert out[0]["synth"] == "rule_based_weak"
    assert out[0]["source"] == "s"
