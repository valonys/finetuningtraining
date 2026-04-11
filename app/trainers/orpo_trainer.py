"""
ORPO (Odds Ratio Preference Optimization) — single-stage SFT + DPO combined.

TRL's `ORPOTrainer` expects the same `{prompt, chosen, rejected}` schema as DPO
but no reference model is needed. Useful when you want preference alignment
without paying for SFT first.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import BaseAgnosticTrainer, TrainRequest
from .dpo_trainer import AgnosticDPOTrainer


class AgnosticORPOTrainer(BaseAgnosticTrainer):
    method = "orpo"

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        base_model_id: str,
        dataset_path: Optional[str] = None,
        hf_dataset_config: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ):
        super().__init__(TrainRequest(
            config=config,
            base_model_id=base_model_id,
            dataset_path=dataset_path,
            hf_dataset_config=hf_dataset_config,
            progress_callback=progress_callback,
        ))

    _format_dataset = AgnosticDPOTrainer._format_dataset   # reuse the DPO row-shape logic

    def _run(self, model, tokenizer, dataset) -> float:
        from trl import ORPOTrainer, ORPOConfig
        import torch

        args = self.config.get("orpo_args", {})
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        cfg = ORPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=args.get("num_train_epochs", 1),
            per_device_train_batch_size=args.get("batch_size", self.profile.per_device_batch_size),
            gradient_accumulation_steps=args.get(
                "gradient_accumulation_steps", self.profile.gradient_accumulation_steps
            ),
            learning_rate=args.get("learning_rate", 8e-6),
            beta=args.get("beta", 0.1),
            fp16=not bf16 and torch.cuda.is_available(),
            bf16=bf16,
            logging_steps=5,
            save_strategy="no",
            gradient_checkpointing=self.profile.gradient_checkpointing,
            report_to="none",
            max_length=args.get("max_length", self.profile.max_seq_length),
        )

        trainer = ORPOTrainer(
            model=model,
            args=cfg,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        result = trainer.train()
        return float(result.training_loss)
