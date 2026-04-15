"""
DPO (Direct Preference Optimization) trainer.

Dataset expectations:
    {"prompt": str, "chosen": str, "rejected": str}

TRL's `DPOTrainer` / `DPOConfig` handles the loss; we just wire up model
loading, chat-template application (via registry), and hardware-aware defaults.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .base import BaseAgnosticTrainer, TrainRequest

logger = logging.getLogger(__name__)


class AgnosticDPOTrainer(BaseAgnosticTrainer):
    method = "dpo"

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
        cols = set(dataset.column_names)
        required = {"prompt", "chosen", "rejected"}
        if required.issubset(cols):
            return dataset

        # If rows are in {instruction, chosen, rejected} form, wrap the prompt
        in_col = "instruction" if "instruction" in cols else ("input" if "input" in cols else None)
        if in_col and "chosen" in cols and "rejected" in cols:
            sys_prompt = self.config.get("system_prompt", "You are a helpful assistant.")
            template = self.template

            def _fmt(ex):
                return {
                    "prompt": template.format_prompt_only(
                        system=sys_prompt, instruction=ex[in_col]
                    ),
                    "chosen": ex["chosen"],
                    "rejected": ex["rejected"],
                }
            return dataset.map(_fmt, remove_columns=[c for c in cols if c not in {"chosen", "rejected"}])

        raise ValueError(
            f"DPO dataset must contain 'prompt', 'chosen', 'rejected' columns. Found: {cols}"
        )

    def _run(self, model, tokenizer, dataset) -> float:
        from trl import DPOTrainer, DPOConfig
        import torch

        dpo_args = self.config.get("dpo_args", {})
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        cfg = DPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=dpo_args.get("num_train_epochs", 1),
            per_device_train_batch_size=dpo_args.get("batch_size", self.profile.per_device_batch_size),
            gradient_accumulation_steps=dpo_args.get(
                "gradient_accumulation_steps", self.profile.gradient_accumulation_steps
            ),
            learning_rate=dpo_args.get("learning_rate", 5e-5),
            beta=dpo_args.get("beta", 0.1),
            fp16=not bf16 and torch.cuda.is_available(),
            bf16=bf16,
            logging_steps=5,
            save_strategy="no",
            gradient_checkpointing=self.profile.gradient_checkpointing,
            report_to="none",
            max_length=dpo_args.get("max_length", self.profile.max_seq_length),
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,             # auto-created frozen copy
            args=cfg,
            train_dataset=dataset,
            tokenizer=tokenizer,
            callbacks=self._build_callbacks(),
        )
        result = trainer.train()
        return float(result.training_loss)
