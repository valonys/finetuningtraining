"""
KTO (Kahneman-Tversky Optimization) — alignment from single-sided ± labels.

Dataset schema:
    {"prompt": str, "completion": str, "label": bool}

Useful when you have thumbs-up/-down feedback instead of paired preferences.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .base import BaseAgnosticTrainer, TrainRequest


class AgnosticKTOTrainer(BaseAgnosticTrainer):
    method = "kto"

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        base_model_id: str,
        dataset_path: Optional[str] = None,
        hf_dataset_config: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        loss_history_sink: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(TrainRequest(
            config=config,
            base_model_id=base_model_id,
            dataset_path=dataset_path,
            hf_dataset_config=hf_dataset_config,
            progress_callback=progress_callback,
            loss_history_sink=loss_history_sink,
        ))

    def _format_dataset(self, dataset, tokenizer):
        needed = {"prompt", "completion", "label"}
        if not needed.issubset(set(dataset.column_names)):
            raise ValueError(f"KTO needs columns {needed}, got {dataset.column_names}")
        return dataset

    def _run(self, model, tokenizer, dataset) -> float:
        from trl import KTOTrainer, KTOConfig
        import torch

        args = self.config.get("kto_args", {})
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        cfg = KTOConfig(
            output_dir=self.output_dir,
            num_train_epochs=args.get("num_train_epochs", 1),
            per_device_train_batch_size=args.get("batch_size", self.profile.per_device_batch_size),
            gradient_accumulation_steps=args.get(
                "gradient_accumulation_steps", self.profile.gradient_accumulation_steps
            ),
            learning_rate=args.get("learning_rate", 5e-6),
            beta=args.get("beta", 0.1),
            desirable_weight=args.get("desirable_weight", 1.0),
            undesirable_weight=args.get("undesirable_weight", 1.0),
            fp16=not bf16 and torch.cuda.is_available(),
            bf16=bf16,
            logging_steps=5,
            save_strategy="no",
            gradient_checkpointing=self.profile.gradient_checkpointing,
            report_to="none",
        )

        trainer = KTOTrainer(
            model=model,
            args=cfg,
            train_dataset=dataset,
            tokenizer=tokenizer,
            callbacks=self._build_callbacks(),
        )
        result = trainer.train()
        return float(result.training_loss)
