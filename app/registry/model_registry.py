"""
app/registry/model_registry.py
──────────────────────────────
JSONL-backed model registry with promote / rollback control. Closes A3
of the Lane A blueprint (see ``docs/SPRINTS.md``).

Storage:
    <root>/model_versions.jsonl    append-only state log; current state
                                   is the LAST line per ``model_version``
    <root>/promotion_events.jsonl  append-only audit log of transitions

Append-only avoids file-rewrite races and gives a free history for
compliance. Reads scan the whole file (O(N) where N is total versions
across all time) — fine for the MVP scale (< 10k versions per domain
ever); a DB-backed registry is the A6 follow-up.

State machine:
    None        -> CANDIDATE
    CANDIDATE   -> STAGING | ROLLED_BACK
    STAGING     -> PRODUCTION | ROLLED_BACK
    PRODUCTION  -> ROLLED_BACK     (rollback path)
    ROLLED_BACK -> STAGING         (re-attempt allowed)

Invariant: at most one PRODUCTION model per domain. Promoting a second
to PRODUCTION auto-rolls back the prior one in the same call.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .schemas import (
    ModelStatus,
    ModelVersion,
    PromotionEvent,
    RollbackResult,
)

logger = logging.getLogger(__name__)


_VERSIONS_FILE = "model_versions.jsonl"
_EVENTS_FILE = "promotion_events.jsonl"


_ALLOWED: dict[ModelStatus | None, set[ModelStatus]] = {
    None: {ModelStatus.CANDIDATE},
    ModelStatus.CANDIDATE: {ModelStatus.STAGING, ModelStatus.ROLLED_BACK},
    ModelStatus.STAGING: {ModelStatus.PRODUCTION, ModelStatus.ROLLED_BACK},
    ModelStatus.PRODUCTION: {ModelStatus.ROLLED_BACK},
    ModelStatus.ROLLED_BACK: {ModelStatus.STAGING},
}


# ──────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────
class UnknownVersion(KeyError):
    """Raised when a model_version id is not in the registry."""


class InvalidTransition(ValueError):
    """Raised when a status transition is not allowed by the state machine."""


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────
class ModelRegistry:
    """JSONL-backed model registry."""

    def __init__(
        self,
        root: Path | str = "outputs/registry",
        *,
        clock: Callable[[], datetime] | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._versions_path = self.root / _VERSIONS_FILE
        self._events_path = self.root / _EVENTS_FILE
        self._now = clock or (lambda: datetime.now(timezone.utc))

    # ── Reads ─────────────────────────────────────────────────
    def list(
        self,
        *,
        domain: str | None = None,
        status: ModelStatus | None = None,
    ) -> list[ModelVersion]:
        """Materialize current state by replaying the JSONL file.
        Optional ``domain`` and ``status`` filters are applied last."""
        state = self._materialize()
        out = list(state.values())
        if domain is not None:
            out = [v for v in out if v.domain == domain]
        if status is not None:
            out = [v for v in out if v.status == status]
        # Stable sort: most recently updated first.
        out.sort(key=lambda v: v.updated_at, reverse=True)
        return out

    def get(self, model_version: str) -> ModelVersion:
        state = self._materialize()
        if model_version not in state:
            raise UnknownVersion(model_version)
        return state[model_version]

    def current_production(self, domain: str) -> ModelVersion | None:
        for v in self.list(domain=domain, status=ModelStatus.PRODUCTION):
            return v
        return None

    def list_events(
        self, *, model_version: str | None = None
    ) -> list[PromotionEvent]:
        events = self._read_jsonl(self._events_path, PromotionEvent)
        if model_version is not None:
            events = [e for e in events if e.model_version == model_version]
        return events

    # ── Writes ────────────────────────────────────────────────
    def register_candidate(
        self,
        *,
        domain: str,
        base_model_id: str,
        adapter_path: str,
        artifact_path: str | None = None,
        artifact_sha256: str | None = None,
        dataset_manifest_path: str | None = None,
        eval_report_path: str | None = None,
        model_version: str | None = None,
        notes: str | None = None,
        extra: dict[str, Any] | None = None,
        actor: str | None = None,
    ) -> ModelVersion:
        """Register a freshly-exported artifact as a CANDIDATE."""
        ts = self._now().isoformat(timespec="seconds")
        version_id = model_version or self._generate_version_id(
            domain=domain, artifact_sha256=artifact_sha256
        )
        record = ModelVersion(
            model_version=version_id,
            domain=domain,
            base_model_id=base_model_id,
            adapter_path=adapter_path,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            dataset_manifest_path=dataset_manifest_path,
            eval_report_path=eval_report_path,
            status=ModelStatus.CANDIDATE,
            promoted_from=None,
            created_at=ts,
            updated_at=ts,
            notes=notes,
            extra=extra or {},
        )
        self._append_version(record)
        self._append_event(PromotionEvent(
            model_version=version_id,
            domain=domain,
            from_status=None,
            to_status=ModelStatus.CANDIDATE,
            actor=actor,
            reason="initial registration",
            timestamp=ts,
        ))
        logger.info(f"📝 Registered candidate {version_id} ({domain})")
        return record

    def promote(
        self,
        model_version: str,
        *,
        to_status: ModelStatus,
        actor: str | None = None,
        reason: str | None = None,
    ) -> ModelVersion:
        """Move a version to ``to_status``. Validates the state machine
        transition and, when promoting to PRODUCTION, atomically rolls
        back the existing production for the same domain."""
        current = self.get(model_version)
        if to_status not in _ALLOWED.get(current.status, set()):
            raise InvalidTransition(
                f"{model_version}: {current.status.value} -> {to_status.value} not allowed"
            )

        ts = self._now().isoformat(timespec="seconds")

        # Auto-rollback the existing production for this domain, if any.
        if to_status == ModelStatus.PRODUCTION:
            prior_prod = self.current_production(current.domain)
            if prior_prod and prior_prod.model_version != model_version:
                self._write_transition(
                    prior_prod,
                    new_status=ModelStatus.ROLLED_BACK,
                    actor=actor or "system",
                    reason=f"auto-demoted: {model_version} promoted to production",
                    timestamp=ts,
                )

        promoted = self._write_transition(
            current,
            new_status=to_status,
            actor=actor,
            reason=reason,
            timestamp=ts,
            promoted_from=current.model_version if to_status == ModelStatus.PRODUCTION else current.promoted_from,
        )
        logger.info(
            f"⬆️  Promoted {model_version}: {current.status.value} -> {to_status.value}"
        )
        return promoted

    def rollback(
        self,
        *,
        domain: str,
        target_version: str | None = None,
        actor: str | None = None,
        reason: str | None = None,
    ) -> RollbackResult:
        """Demote the current production for ``domain`` to ROLLED_BACK.

        If ``target_version`` is provided, that version is promoted to
        PRODUCTION as the replacement. Otherwise, the most recently
        updated ROLLED_BACK or STAGING version (excluding the one we
        just demoted) is promoted. If no replacement exists, the domain
        is left without an active production model.
        """
        prod = self.current_production(domain)
        if prod is None:
            raise InvalidTransition(f"no production model for domain={domain}")

        ts = self._now().isoformat(timespec="seconds")
        rolled = self._write_transition(
            prod,
            new_status=ModelStatus.ROLLED_BACK,
            actor=actor,
            reason=reason or "manual rollback",
            timestamp=ts,
        )

        # Pick a replacement.
        replacement: ModelVersion | None = None
        if target_version is not None:
            candidate = self.get(target_version)
            if candidate.status not in {ModelStatus.STAGING, ModelStatus.ROLLED_BACK}:
                raise InvalidTransition(
                    f"target {target_version} is {candidate.status.value}; "
                    f"only STAGING / ROLLED_BACK can be promoted to PRODUCTION"
                )
            # Bring ROLLED_BACK back through STAGING first to honor the state machine.
            if candidate.status == ModelStatus.ROLLED_BACK:
                candidate = self._write_transition(
                    candidate,
                    new_status=ModelStatus.STAGING,
                    actor=actor or "system",
                    reason="rollback replacement: ROLLED_BACK -> STAGING",
                    timestamp=ts,
                )
            replacement = self.promote(
                candidate.model_version,
                to_status=ModelStatus.PRODUCTION,
                actor=actor,
                reason=reason or f"rollback replacement for {prod.model_version}",
            )
        else:
            replacement = self._auto_pick_replacement(domain, exclude=prod.model_version)
            if replacement is not None:
                if replacement.status == ModelStatus.ROLLED_BACK:
                    replacement = self._write_transition(
                        replacement,
                        new_status=ModelStatus.STAGING,
                        actor=actor or "system",
                        reason="rollback auto-replacement: ROLLED_BACK -> STAGING",
                        timestamp=ts,
                    )
                replacement = self.promote(
                    replacement.model_version,
                    to_status=ModelStatus.PRODUCTION,
                    actor=actor,
                    reason=reason or f"auto-replacement after rollback of {prod.model_version}",
                )

        return RollbackResult(rolled_back=rolled, new_production=replacement)

    # ── Internals ─────────────────────────────────────────────
    def _materialize(self) -> dict[str, ModelVersion]:
        state: dict[str, ModelVersion] = {}
        for record in self._read_jsonl(self._versions_path, ModelVersion):
            state[record.model_version] = record
        return state

    def _read_jsonl(self, path: Path, model_cls):
        if not path.is_file():
            return []
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(model_cls.model_validate_json(line))
                except Exception as exc:
                    logger.warning(f"⚠️  skipping malformed registry line: {exc}")
        return out

    def _append_version(self, record: ModelVersion) -> None:
        self._append_jsonl(self._versions_path, record.model_dump_json())

    def _append_event(self, event: PromotionEvent) -> None:
        self._append_jsonl(self._events_path, event.model_dump_json())

    def _append_jsonl(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 'a' is atomic on POSIX for line-sized writes (< PIPE_BUF). The
        # registry's records are well under that, so concurrent appends
        # don't interleave on the same line.
        with path.open("a", encoding="utf-8") as f:
            f.write(payload + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # Some filesystems / sandboxes refuse fsync — non-fatal.
                pass

    def _write_transition(
        self,
        current: ModelVersion,
        *,
        new_status: ModelStatus,
        actor: str | None,
        reason: str | None,
        timestamp: str,
        promoted_from: str | None = None,
    ) -> ModelVersion:
        if new_status not in _ALLOWED.get(current.status, set()):
            raise InvalidTransition(
                f"{current.model_version}: {current.status.value} -> {new_status.value} not allowed"
            )
        updated = current.model_copy(update={
            "status": new_status,
            "updated_at": timestamp,
            "promoted_from": promoted_from if promoted_from is not None else current.promoted_from,
        })
        self._append_version(updated)
        self._append_event(PromotionEvent(
            model_version=current.model_version,
            domain=current.domain,
            from_status=current.status,
            to_status=new_status,
            actor=actor,
            reason=reason,
            timestamp=timestamp,
        ))
        return updated

    def _auto_pick_replacement(
        self, domain: str, *, exclude: str
    ) -> ModelVersion | None:
        candidates = [
            v for v in self.list(domain=domain)
            if v.model_version != exclude
            and v.status in {ModelStatus.STAGING, ModelStatus.ROLLED_BACK}
        ]
        # list() sorts by updated_at desc, so first match is most recent.
        return candidates[0] if candidates else None

    def _generate_version_id(
        self, *, domain: str, artifact_sha256: str | None
    ) -> str:
        ts = self._now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", domain.lower()).strip("-") or "domain"
        if artifact_sha256:
            short = artifact_sha256[:8]
        else:
            # Use a hash of (slug + ts + monotonic counter via existing files)
            short = hashlib.sha256(f"{slug}-{ts}-{self._versions_path}".encode()).hexdigest()[:8]
        return f"{slug}-{ts}-{short}"


def default_registry() -> ModelRegistry:
    """Return a registry rooted at ``outputs/registry`` (the canonical
    location used by the deploy stage and the API endpoints)."""
    return ModelRegistry(root="outputs/registry")
