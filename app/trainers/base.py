"""
BaseAgnosticTrainer — shared plumbing for every training method.

Responsibilities:
  * Load the domain config
  * Detect hardware and resolve the right backend
  * Load the model + tokenizer (Unsloth / MLX-LM / plain TRL)
  * Apply LoRA (where supported)
  * Load and format the dataset using the template registry
  * Save the final adapter + a `training_method.txt` tag
  * Call `progress_callback` at well-defined phases
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.hardware import detect_hardware, resolve_profile
from app.templates import get_template_for

logger = logging.getLogger(__name__)


@dataclass
class TrainRequest:
    """Generic training request — fields every trainer accepts."""
    config: Dict[str, Any]
    base_model_id: str
    dataset_path: Optional[str] = None
    hf_dataset_config: Optional[dict] = None
    progress_callback: Optional[Callable[[float], None]] = None
    # Caller-supplied list receiving per-step metric dicts from
    # LossHistoryCallback. Reference is shared with the /v1/jobs/{id}
    # endpoint so the UI can render live loss curves.
    loss_history_sink: Optional[List[Dict[str, Any]]] = None


class BaseAgnosticTrainer(ABC):
    method: str = "base"

    def __init__(self, req: TrainRequest):
        self.req = req
        self.config = req.config
        self.base_model_id = req.base_model_id
        self.dataset_path = req.dataset_path
        self.hf_dataset_cfg = req.hf_dataset_config
        self.progress_cb = req.progress_callback or (lambda p: None)
        self.loss_history_sink = req.loss_history_sink

        self.hw = detect_hardware()
        self.profile = resolve_profile(self.hw)
        self.template = get_template_for(self.base_model_id)

        self.domain_name = self.config.get("domain_name", "default")
        self.output_dir = f"outputs/{self.domain_name}"
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(
            f"🎯 {self.method.upper()} | model={self.base_model_id} | "
            f"backend={self.profile.training_backend} | "
            f"hw={self.hw.tier} ({self.hw.effective_memory_gb} GB) | "
            f"template={self.template.name}"
        )

    # ── Public entry point ────────────────────────────────────
    def train(self) -> Dict[str, Any]:
        self.progress_cb(0.05)
        dataset = self._load_dataset()
        self.progress_cb(0.15)

        model, tokenizer = self._load_model()
        self.progress_cb(0.35)

        model = self._apply_lora_if_supported(model)
        self.progress_cb(0.45)

        formatted = self._format_dataset(dataset, tokenizer)
        self.progress_cb(0.55)

        final_loss = self._run(model, tokenizer, formatted)
        self.progress_cb(0.95)

        self._save(model, tokenizer)
        self.progress_cb(1.0)

        return {
            "method": self.method,
            "domain": self.domain_name,
            "backend": self.profile.training_backend,
            "hardware": self.hw.tier,
            "template": self.template.name,
            "samples": len(dataset),
            "final_loss": final_loss,
            "adapter_path": self.output_dir,
        }

    # ── Pluggable phases ──────────────────────────────────────
    def _load_dataset(self):
        from app.data_forge.dataset_builder import DatasetBuilder  # noqa: F401
        from datasets import load_from_disk, load_dataset

        if self.hf_dataset_cfg:
            repo = self.hf_dataset_cfg["repo_id"]
            split = self.hf_dataset_cfg.get("split", "train")
            subset = self.hf_dataset_cfg.get("subset")
            token = self.hf_dataset_cfg.get("token") or os.environ.get("HF_TOKEN")
            ds = load_dataset(repo, name=subset, split=split, token=token)
            logger.info(f"📦 HF dataset: {repo} ({len(ds)} rows)")
            return ds

        if self.dataset_path is None:
            raise ValueError("Trainer: provide dataset_path or hf_dataset_config")

        if os.path.isdir(self.dataset_path):
            ds = load_from_disk(self.dataset_path)
        elif self.dataset_path.endswith(".json") or self.dataset_path.endswith(".jsonl"):
            ds = load_dataset("json", data_files=self.dataset_path, split="train")
        elif self.dataset_path.endswith(".csv"):
            ds = load_dataset("csv", data_files=self.dataset_path, split="train")
        else:
            raise ValueError(f"Unsupported dataset path: {self.dataset_path}")
        logger.info(f"📦 Local dataset: {self.dataset_path} ({len(ds)} rows)")
        return ds

    def _load_model(self):
        from .backends import load_model_and_tokenizer
        return load_model_and_tokenizer(
            model_id=self.base_model_id,
            profile=self.profile,
            hardware=self.hw,
        )

    def _apply_lora_if_supported(self, model):
        from .backends import apply_lora
        return apply_lora(model, profile=self.profile, backend=self.profile.training_backend)

    def _build_callbacks(self) -> list:
        """Assemble the TRL/Transformers callback list for this run.

        Currently just the loss-history callback when a sink was supplied.
        Returns an empty list when there's no sink or transformers is
        missing, so callers can always splat it into `callbacks=...`.
        """
        if self.loss_history_sink is None:
            return []
        from .callbacks import make_loss_callback
        cb = make_loss_callback(self.loss_history_sink)
        return [cb] if cb is not None else []

    @abstractmethod
    def _format_dataset(self, dataset, tokenizer): ...

    @abstractmethod
    def _run(self, model, tokenizer, dataset) -> float: ...

    def _save(self, model, tokenizer):
        model.save_pretrained(self.output_dir)
        try:
            tokenizer.save_pretrained(self.output_dir)
        except Exception:
            pass
        with open(os.path.join(self.output_dir, "training_method.txt"), "w") as f:
            f.write(self.method)
        logger.info(f"💾 Adapter saved → {self.output_dir}")
