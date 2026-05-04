"""
app/inference/canary.py
───────────────────────
Canary routing for the inference manager (A5).

A canary config splits requests for one ``domain`` between two
registered model versions: a ``stable`` (gets the majority) and a
``canary`` (gets ``canary_pct`` percent). Versions are model_version
ids that resolve through the A3 registry's PRODUCTION + STAGING rows.

Per-version metrics (latency, error count, total count) are
accumulated in process so SLOEvaluator can verdict the canary's
performance after a configurable window. The auto-abort *closed loop*
(actually demoting a misbehaving canary) is intentionally deferred to
A5b once we have real traffic to validate threshold choices.

Routing decision is a simple uniform random draw — no sticky-session
guarantees yet (a future sprint can hash-by-tenant for stickiness).
"""
from __future__ import annotations

import random
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class CanaryConfig:
    """One canary split for one domain. ``stable_version`` and
    ``canary_version`` are model_version ids known to the registry
    (or string sentinels when the manager is wired without a registry
    in dev / tests)."""

    domain: str
    stable_version: str
    canary_version: str
    canary_pct: float = 10.0     # 0–100; hard-clamped on assignment

    def __post_init__(self):
        if not (0.0 <= self.canary_pct <= 100.0):
            raise ValueError(
                f"canary_pct must be in [0, 100], got {self.canary_pct}"
            )


@dataclass
class _VersionMetrics:
    requests: int = 0
    errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    quality_probes: list[bool] = field(default_factory=list)


class CanaryRouter:
    """Manages a set of per-domain CanaryConfigs plus per-version
    metric buckets. Thread-safe via a single lock."""

    def __init__(self, *, rng: random.Random | None = None):
        self._configs: dict[str, CanaryConfig] = {}
        self._metrics: dict[tuple[str, str], _VersionMetrics] = defaultdict(
            _VersionMetrics
        )
        self._lock = threading.Lock()
        # Allow injecting a seeded RNG for deterministic tests.
        self._rng = rng or random.Random()

    # ── Configuration ────────────────────────────────────────
    def configure(self, cfg: CanaryConfig) -> None:
        with self._lock:
            self._configs[cfg.domain] = cfg

    def remove(self, domain: str) -> None:
        with self._lock:
            self._configs.pop(domain, None)

    def get_config(self, domain: str) -> CanaryConfig | None:
        return self._configs.get(domain)

    # ── Routing ──────────────────────────────────────────────
    def pick_version(self, domain: str, *, fallback: str = "base") -> str:
        """Return the model_version this request should hit. ``fallback``
        is used when the domain has no canary configured (i.e. all
        traffic goes there). Routing is uniform random per request."""
        cfg = self._configs.get(domain)
        if cfg is None:
            return fallback
        # Uniform draw against canary_pct; e.g. 10.0 → 10% of requests
        # go to canary, 90% to stable.
        roll = self._rng.random() * 100.0
        return cfg.canary_version if roll < cfg.canary_pct else cfg.stable_version

    # ── Metrics ──────────────────────────────────────────────
    def record(
        self,
        *,
        domain: str,
        version: str,
        latency_ms: float,
        error: bool = False,
        quality_probe: bool | None = None,
    ) -> None:
        with self._lock:
            m = self._metrics[(domain, version)]
            m.requests += 1
            m.latencies_ms.append(latency_ms)
            if error:
                m.errors += 1
            if quality_probe is not None:
                m.quality_probes.append(quality_probe)

    def metrics_for(self, domain: str, version: str) -> dict:
        with self._lock:
            m = self._metrics[(domain, version)]
            return {
                "domain": domain,
                "version": version,
                "requests": m.requests,
                "errors": m.errors,
                "latencies_ms": list(m.latencies_ms),
                "quality_probes": list(m.quality_probes),
            }

    def reset_metrics(self, *, domain: str | None = None) -> None:
        """Clear metrics — typically called at the start of a fresh
        canary window. Pass ``domain`` to only reset one."""
        with self._lock:
            if domain is None:
                self._metrics.clear()
            else:
                for key in list(self._metrics.keys()):
                    if key[0] == domain:
                        del self._metrics[key]

    def domains(self) -> Iterable[str]:
        return list(self._configs.keys())
