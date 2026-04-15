"""
app/trainers/callbacks.py
─────────────────────────
TrainerCallback that captures per-step metrics (loss, learning rate,
grad norm, epoch, step) into a caller-supplied list. Wired into every
agnostic trainer so the /v1/jobs/{id} endpoint can surface live-ish
training graphs in the UI, similar to what Unsloth's studio shows.

Design notes:

  * We tap into HuggingFace `transformers.TrainerCallback.on_log`,
    which fires whenever the Trainer emits a log dict. That dict
    contains `loss`, `learning_rate`, `grad_norm`, `epoch`, and
    (on_train_end) `train_loss` / `train_runtime` etc. We keep the
    handful of fields a loss-curve chart actually cares about.

  * The callback writes to a list passed in by the caller (the job
    registry in main.py). That list is a plain Python list -- thread
    safe enough for the background-task pattern here, which runs one
    training job at a time on the FastAPI event loop thread pool.

  * We cap the history length (default 10k entries) so a pathologically
    long training run can't balloon memory. When the cap is reached,
    oldest entries are dropped (FIFO).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# HuggingFace transformers provides the callback base class. We import
# it lazily so the module stays importable without transformers
# installed (relevant for CI smoke tests).
try:
    from transformers import TrainerCallback as _BaseCallback
    _TRANSFORMERS_AVAILABLE = True
except Exception:          # pragma: no cover - import shim
    _BaseCallback = object
    _TRANSFORMERS_AVAILABLE = False


class LossHistoryCallback(_BaseCallback):  # type: ignore[misc]
    """Append per-step metrics to a shared list.

    Each appended entry is a dict::

        {
            "step": int,
            "loss": float | None,
            "learning_rate": float | None,
            "grad_norm": float | None,
            "epoch": float | None,
            "ts": float,         # wall-clock seconds since epoch
        }

    The caller (main.py's background-task) passes in the list it wants
    populated. That list is the same one returned by /v1/jobs/{id},
    so polling the job status gives the UI a live stream of metrics.
    """

    # Fields we copy from the Trainer's log dict into our history.
    _KEEP_KEYS = ("loss", "learning_rate", "grad_norm", "epoch")

    def __init__(self, sink: List[Dict[str, Any]], *, max_entries: int = 10_000):
        self.sink = sink
        self.max_entries = max_entries

    # ── TrainerCallback hook ────────────────────────────────────
    def on_log(self, args, state, control, logs: Optional[Dict[str, Any]] = None, **kwargs):
        if not logs:
            return control

        # Ignore the final summary log (has `train_loss`/`train_runtime`
        # but no per-step `loss`). Per-step logs have `loss`.
        has_step_loss = any(k in logs for k in ("loss", "grad_norm"))
        if not has_step_loss:
            return control

        import time
        entry: Dict[str, Any] = {"step": getattr(state, "global_step", 0), "ts": time.time()}
        for k in self._KEEP_KEYS:
            v = logs.get(k)
            if v is not None:
                entry[k] = float(v) if not isinstance(v, float) else v
            else:
                entry[k] = None

        self.sink.append(entry)
        if len(self.sink) > self.max_entries:
            # Drop oldest entries to stay under cap
            del self.sink[: len(self.sink) - self.max_entries]

        return control


def make_loss_callback(sink: List[Dict[str, Any]]) -> Optional["LossHistoryCallback"]:
    """Return a ready-to-attach callback, or None if transformers isn't installed."""
    if not _TRANSFORMERS_AVAILABLE:
        logger.warning(
            "transformers not installed -- training-metrics callback disabled. "
            "The /v1/jobs/{id} endpoint will return an empty loss_history."
        )
        return None
    return LossHistoryCallback(sink)
