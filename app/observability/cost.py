"""
app/observability/cost.py
─────────────────────────
Per-request token + USD accounting (A5).

The inference manager (or whoever wraps a generation call) calls
``tracker.record(...)`` after each request with the token counts the
backend reported. We map (backend, model) → ($/1M tokens) via a
pluggable cost table and accumulate totals in memory. ``snapshot()``
writes the running totals out as a JSON artifact under
``outputs/metrics/cost_<timestamp>.json`` for offline analysis or
inclusion in a release report.

Cost table notes:
  * Local backends (HF Transformers on this box, llama.cpp, MLX) are
    treated as $0 — we burn local compute / electricity, not API
    credit. If you want to attribute electricity costs add a row.
  * Hosted backends use rough public rates valid as of 2026-Q2; the
    table is overridable via ``CostTracker(cost_table=...)``.
  * Unknown (backend, model) pairs are recorded with $0 and emit a
    one-shot warning so it's visible but doesn't crash the request.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default rates: USD per 1,000,000 tokens. Source: public pricing
# pages as of 2026-Q2. Override via constructor arg for accuracy.
_DEFAULT_RATES: dict[str, dict[str, float]] = {
    # Ollama Cloud (Nemotron family, public pricing)
    "ollama-cloud:nemotron-3-super":   {"in": 0.20, "out": 0.40},
    "ollama-cloud:nemotron-3-nano":    {"in": 0.10, "out": 0.20},
    "ollama-cloud:default":            {"in": 0.20, "out": 0.40},
    # OpenRouter typical mid-tier
    "openrouter:default":              {"in": 0.50, "out": 1.00},
    # HF Inference Endpoints, on-demand small CPU
    "hf-inference:default":            {"in": 0.10, "out": 0.20},
    # Local: vLLM / HF / MLX / llama.cpp on user hardware
    "local:default":                   {"in": 0.0,  "out": 0.0},
}


@dataclass
class _Bucket:
    requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0


@dataclass
class CostTracker:
    """Process-local cost accumulator. Thread-safe via a single lock
    around all mutations."""

    cost_table: dict[str, dict[str, float]] = field(
        default_factory=lambda: dict(_DEFAULT_RATES)
    )
    metrics_dir: Path = field(default_factory=lambda: Path("outputs/metrics"))

    def __post_init__(self):
        self._lock = threading.Lock()
        # Per (backend, model, version) totals so canary cost vs stable
        # is trivially comparable.
        self._totals: dict[tuple[str, str, str], _Bucket] = defaultdict(_Bucket)
        self._unknown_pairs_warned: set[tuple[str, str]] = set()
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        backend: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        version: str = "stable",
    ) -> dict[str, Any]:
        """Record one request. Returns a dict with the request's USD cost
        plus the running per-version totals (handy for log lines)."""
        rate = self._rate_for(backend, model)
        usd = (tokens_in * rate["in"] + tokens_out * rate["out"]) / 1_000_000.0
        with self._lock:
            bucket = self._totals[(backend, model, version)]
            bucket.requests += 1
            bucket.tokens_in += tokens_in
            bucket.tokens_out += tokens_out
            bucket.usd += usd
        return {
            "backend": backend,
            "model": model,
            "version": version,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "usd": round(usd, 6),
        }

    def totals(self) -> dict[str, Any]:
        """Snapshot of running totals broken out by (backend, model, version)."""
        with self._lock:
            return {
                "by_pair": [
                    {
                        "backend": b,
                        "model": m,
                        "version": v,
                        "requests": bucket.requests,
                        "tokens_in": bucket.tokens_in,
                        "tokens_out": bucket.tokens_out,
                        "usd": round(bucket.usd, 6),
                    }
                    for (b, m, v), bucket in self._totals.items()
                ],
                "grand_total_usd": round(
                    sum(b.usd for b in self._totals.values()), 6
                ),
                "snapshot_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

    def snapshot(self) -> Path:
        """Write the current totals to ``outputs/metrics/cost_<ts>.json``
        and return the artifact path."""
        payload = self.totals()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = self.metrics_dir / f"cost_{ts}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        return out

    def reset(self) -> None:
        """Wipe accumulated totals — used between canary windows."""
        with self._lock:
            self._totals.clear()

    # ── Internals ─────────────────────────────────────────────
    def _rate_for(self, backend: str, model: str) -> dict[str, float]:
        key = f"{backend}:{model}"
        if key in self.cost_table:
            return self.cost_table[key]
        fallback = f"{backend}:default"
        if fallback in self.cost_table:
            return self.cost_table[fallback]
        # Unknown pair: warn once, charge $0.
        sig = (backend, model)
        if sig not in self._unknown_pairs_warned:
            logger.warning(
                f"💸 No cost rate for {backend}:{model} — charging $0. "
                f"Add a row to cost_table to fix."
            )
            self._unknown_pairs_warned.add(sig)
        return {"in": 0.0, "out": 0.0}
