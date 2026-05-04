"""
Unit tests for app.registry.ModelRegistry.

The registry is JSONL-backed, so every test gets its own ``tmp_path``
root. Tests cover:

  * register / list / get / current_production
  * promote up the lifecycle (candidate → staging → production)
  * promoting a 2nd version to PRODUCTION auto-demotes the prior one
  * rollback with explicit target_version
  * rollback auto-pick of replacement
  * rollback raises when no production exists
  * invalid transitions raise InvalidTransition
  * unknown version raises UnknownVersion
  * append-only durability — fresh ModelRegistry instance reads the
    same state from disk
  * promotion events log every transition
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from app.registry import (
    InvalidTransition,
    ModelRegistry,
    ModelStatus,
    UnknownVersion,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
class _Clock:
    """Monotonic fake clock so generated version ids are unique + stable."""

    def __init__(self, start: datetime | None = None):
        self.t = start or datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.t = self.t + timedelta(seconds=1)
        return self.t


def _make_registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(root=tmp_path / "registry", clock=_Clock())


def _register(reg: ModelRegistry, *, domain: str = "ai_llm", **overrides):
    defaults = dict(
        domain=domain,
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path=f"outputs/{domain}",
        artifact_path=f"outputs/{domain}/artifacts/model-q4_k_m.gguf",
        artifact_sha256="a" * 64,
        dataset_manifest_path=f"data/processed/{domain}_sft.jsonl",
        eval_report_path=f"outputs/{domain}/eval_2026.json",
    )
    defaults.update(overrides)
    return reg.register_candidate(**defaults)


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────
def test_register_candidate_writes_initial_state(tmp_path):
    reg = _make_registry(tmp_path)
    v = _register(reg)

    assert v.status == ModelStatus.CANDIDATE
    assert v.domain == "ai_llm"
    assert v.model_version.startswith("ai_llm-")
    assert v.created_at == v.updated_at
    assert v.promoted_from is None

    listed = reg.list()
    assert len(listed) == 1
    assert listed[0].model_version == v.model_version

    # get() returns same record
    assert reg.get(v.model_version).model_version == v.model_version

    # An initial-registration event was logged
    events = reg.list_events(model_version=v.model_version)
    assert len(events) == 1
    assert events[0].from_status is None
    assert events[0].to_status == ModelStatus.CANDIDATE


def test_list_filters_by_domain_and_status(tmp_path):
    reg = _make_registry(tmp_path)
    a = _register(reg, domain="ai_llm")
    b = _register(reg, domain="legal")
    reg.promote(b.model_version, to_status=ModelStatus.STAGING)

    assert {v.model_version for v in reg.list(domain="ai_llm")} == {a.model_version}
    assert {v.model_version for v in reg.list(domain="legal")} == {b.model_version}
    assert {v.model_version for v in reg.list(status=ModelStatus.CANDIDATE)} == {a.model_version}
    assert {v.model_version for v in reg.list(status=ModelStatus.STAGING)} == {b.model_version}


def test_full_lifecycle_candidate_staging_production(tmp_path):
    reg = _make_registry(tmp_path)
    v = _register(reg)
    staged = reg.promote(v.model_version, to_status=ModelStatus.STAGING, actor="alice")
    assert staged.status == ModelStatus.STAGING
    promoted = reg.promote(v.model_version, to_status=ModelStatus.PRODUCTION, actor="alice")
    assert promoted.status == ModelStatus.PRODUCTION
    assert promoted.promoted_from == v.model_version

    cur = reg.current_production("ai_llm")
    assert cur is not None and cur.model_version == v.model_version

    # Three transition events recorded (initial + 2 promotions)
    events = reg.list_events(model_version=v.model_version)
    assert [e.to_status for e in events] == [
        ModelStatus.CANDIDATE,
        ModelStatus.STAGING,
        ModelStatus.PRODUCTION,
    ]
    assert events[1].actor == "alice"


def test_promoting_second_to_production_auto_demotes_first(tmp_path):
    reg = _make_registry(tmp_path)
    v1 = _register(reg)
    reg.promote(v1.model_version, to_status=ModelStatus.STAGING)
    reg.promote(v1.model_version, to_status=ModelStatus.PRODUCTION)
    assert reg.current_production("ai_llm").model_version == v1.model_version

    v2 = _register(reg, artifact_sha256="b" * 64)
    reg.promote(v2.model_version, to_status=ModelStatus.STAGING)
    reg.promote(v2.model_version, to_status=ModelStatus.PRODUCTION)

    # v1 was auto-demoted, v2 is now production
    assert reg.get(v1.model_version).status == ModelStatus.ROLLED_BACK
    assert reg.get(v2.model_version).status == ModelStatus.PRODUCTION
    assert reg.current_production("ai_llm").model_version == v2.model_version

    # Event log for v1 includes the auto-demote with a system reason
    v1_events = reg.list_events(model_version=v1.model_version)
    auto_demote = [e for e in v1_events if e.to_status == ModelStatus.ROLLED_BACK]
    assert len(auto_demote) == 1
    assert "auto-demoted" in (auto_demote[0].reason or "")


def test_rollback_with_explicit_target_promotes_replacement(tmp_path):
    reg = _make_registry(tmp_path)
    v_old = _register(reg, artifact_sha256="o" * 64)
    reg.promote(v_old.model_version, to_status=ModelStatus.STAGING)
    reg.promote(v_old.model_version, to_status=ModelStatus.PRODUCTION)

    v_replacement = _register(reg, artifact_sha256="r" * 64)
    reg.promote(v_replacement.model_version, to_status=ModelStatus.STAGING)

    result = reg.rollback(
        domain="ai_llm",
        target_version=v_replacement.model_version,
        actor="oncall",
        reason="prod regression",
    )

    assert result.rolled_back.model_version == v_old.model_version
    assert result.rolled_back.status == ModelStatus.ROLLED_BACK
    assert result.new_production is not None
    assert result.new_production.model_version == v_replacement.model_version
    assert result.new_production.status == ModelStatus.PRODUCTION
    assert reg.current_production("ai_llm").model_version == v_replacement.model_version


def test_rollback_auto_picks_most_recent_replacement(tmp_path):
    reg = _make_registry(tmp_path)
    # Old PRODUCTION
    v_old = _register(reg, artifact_sha256="o" * 64)
    reg.promote(v_old.model_version, to_status=ModelStatus.STAGING)
    reg.promote(v_old.model_version, to_status=ModelStatus.PRODUCTION)
    # Two staged candidates; the second registered wins (most recent)
    v_first = _register(reg, artifact_sha256="1" * 64)
    reg.promote(v_first.model_version, to_status=ModelStatus.STAGING)
    v_second = _register(reg, artifact_sha256="2" * 64)
    reg.promote(v_second.model_version, to_status=ModelStatus.STAGING)

    result = reg.rollback(domain="ai_llm")
    assert result.new_production is not None
    assert result.new_production.model_version == v_second.model_version


def test_rollback_raises_when_no_production_exists(tmp_path):
    reg = _make_registry(tmp_path)
    _register(reg)  # candidate only
    with pytest.raises(InvalidTransition, match="no production model"):
        reg.rollback(domain="ai_llm")


def test_invalid_transition_candidate_to_production(tmp_path):
    reg = _make_registry(tmp_path)
    v = _register(reg)
    # Skipping STAGING isn't allowed
    with pytest.raises(InvalidTransition, match="not allowed"):
        reg.promote(v.model_version, to_status=ModelStatus.PRODUCTION)


def test_invalid_transition_production_back_to_staging(tmp_path):
    reg = _make_registry(tmp_path)
    v = _register(reg)
    reg.promote(v.model_version, to_status=ModelStatus.STAGING)
    reg.promote(v.model_version, to_status=ModelStatus.PRODUCTION)
    with pytest.raises(InvalidTransition):
        reg.promote(v.model_version, to_status=ModelStatus.STAGING)


def test_get_unknown_version_raises(tmp_path):
    reg = _make_registry(tmp_path)
    with pytest.raises(UnknownVersion):
        reg.get("ai_llm-nonexistent")


def test_state_survives_fresh_registry_instance(tmp_path):
    reg1 = _make_registry(tmp_path)
    v = _register(reg1)
    reg1.promote(v.model_version, to_status=ModelStatus.STAGING)
    reg1.promote(v.model_version, to_status=ModelStatus.PRODUCTION)

    # New instance pointed at the same root replays the JSONL
    reg2 = ModelRegistry(root=tmp_path / "registry")
    rehydrated = reg2.get(v.model_version)
    assert rehydrated.status == ModelStatus.PRODUCTION
    assert reg2.current_production("ai_llm").model_version == v.model_version

    # Event log persisted across instances
    assert len(reg2.list_events(model_version=v.model_version)) == 3


def test_rerun_from_rollback_back_through_staging(tmp_path):
    """A rolled-back version should be promotable back to STAGING (re-attempt path)."""
    reg = _make_registry(tmp_path)
    v = _register(reg)
    reg.promote(v.model_version, to_status=ModelStatus.STAGING)
    reg.promote(v.model_version, to_status=ModelStatus.PRODUCTION)
    # Bring a replacement to production so v gets demoted.
    other = _register(reg, artifact_sha256="o" * 64)
    reg.promote(other.model_version, to_status=ModelStatus.STAGING)
    reg.promote(other.model_version, to_status=ModelStatus.PRODUCTION)
    assert reg.get(v.model_version).status == ModelStatus.ROLLED_BACK

    # Now revive v
    revived = reg.promote(v.model_version, to_status=ModelStatus.STAGING)
    assert revived.status == ModelStatus.STAGING


def test_rollback_target_must_be_staging_or_rolled_back(tmp_path):
    reg = _make_registry(tmp_path)
    v_prod = _register(reg, artifact_sha256="p" * 64)
    reg.promote(v_prod.model_version, to_status=ModelStatus.STAGING)
    reg.promote(v_prod.model_version, to_status=ModelStatus.PRODUCTION)
    # This second version is still CANDIDATE — invalid as a rollback target
    v_cand = _register(reg, artifact_sha256="c" * 64)
    with pytest.raises(InvalidTransition, match="only STAGING / ROLLED_BACK"):
        reg.rollback(domain="ai_llm", target_version=v_cand.model_version)


def test_register_candidate_without_artifact_sha_still_generates_id(tmp_path):
    reg = _make_registry(tmp_path)
    v = reg.register_candidate(
        domain="ai_llm",
        base_model_id="Qwen/x",
        adapter_path="outputs/x",
        artifact_sha256=None,
    )
    assert v.model_version.startswith("ai_llm-")
    assert v.artifact_sha256 is None
