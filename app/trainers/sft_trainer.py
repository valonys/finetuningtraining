"""
SFT trainer.

Uses:
  * `trl.SFTTrainer` on CUDA / MPS / CPU (works for Unsloth-wrapped models too)
  * `mlx_lm.lora.train` on Apple Silicon when `training_backend == "mlx"`

Dataset expectations:
  * Rows with a `text` field (already template-formatted), OR
  * Rows with a `messages` field, OR
  * Rows with `instruction`/`response` (we'll template them)

The trainer always passes `DataCollatorForCompletionOnlyLM` when the template
exposes a `response_template`, so the loss is computed **only** on response
tokens (key for quality).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .base import BaseAgnosticTrainer, TrainRequest

logger = logging.getLogger(__name__)


class AgnosticSFTTrainer(BaseAgnosticTrainer):
    method = "sft"

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

    # ── Format dataset into a "text" column ───────────────────
    def _format_dataset(self, dataset, tokenizer):
        cols = set(dataset.column_names)
        sys_prompt = self.config.get("system_prompt", "You are a helpful assistant.")

        # Path 1: already formatted with "text" field
        if "text" in cols:
            return dataset

        # Path 2: {"messages": [...]} — use the tokenizer's chat template
        if "messages" in cols:
            def _apply(example):
                return {
                    "text": tokenizer.apply_chat_template(
                        example["messages"], tokenize=False, add_generation_prompt=False
                    )
                }
            return dataset.map(_apply, remove_columns=list(cols))

        # Path 3: raw instruction / response
        in_col = _pick(cols, "instruction", "input", "question", "prompt")
        out_col = _pick(cols, "response", "output", "answer", "completion")
        if not in_col or not out_col:
            raise ValueError(
                f"SFT dataset missing instruction/response columns. Found: {cols}. "
                "Expected one of: (instruction/input/question/prompt) + (response/output/answer/completion)."
            )

        template = self.template

        def _format(example):
            return {
                "text": template.format_sft(
                    system=sys_prompt,
                    instruction=example[in_col],
                    response=example[out_col],
                )
            }

        return dataset.map(_format, remove_columns=list(cols))

    # ── Run training (TRL or MLX) ─────────────────────────────
    def _run(self, model, tokenizer, dataset) -> float:
        if self.profile.training_backend == "mlx":
            return self._run_mlx(model, tokenizer, dataset)
        return self._run_trl(model, tokenizer, dataset)

    # ── TRL path ──────────────────────────────────────────────
    def _run_trl(self, model, tokenizer, dataset) -> float:
        from trl import SFTTrainer, SFTConfig
        from transformers import DataCollatorForCompletionOnlyLM

        train_args = self.config.get("training_args", {})
        max_seq = train_args.get("max_seq_length", self.profile.max_seq_length)
        max_steps = train_args.get("max_steps", -1)
        num_epochs = train_args.get("num_train_epochs", 1)
        lr = train_args.get("learning_rate", 2e-4)
        bs = train_args.get("batch_size", self.profile.per_device_batch_size)
        ga = train_args.get("gradient_accumulation_steps", self.profile.gradient_accumulation_steps)

        import torch
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        args = SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            learning_rate=lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            fp16=not bf16 and torch.cuda.is_available(),
            bf16=bf16,
            logging_steps=5,
            save_strategy="no",
            optim="adamw_8bit" if self.profile.load_in_4bit else "adamw_torch",
            gradient_checkpointing=self.profile.gradient_checkpointing,
            max_seq_length=max_seq,
            dataset_text_field="text",
            report_to="none",
        )

        data_collator = None
        if self.template.response_template:
            try:
                data_collator = DataCollatorForCompletionOnlyLM(
                    response_template=self.template.response_template,
                    tokenizer=tokenizer,
                )
            except Exception as e:
                logger.warning(f"⚠️  Completion-only collator unavailable: {e}")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=args,
            data_collator=data_collator,
            callbacks=self._build_callbacks(),
        )
        result = trainer.train()
        return float(result.training_loss)

    # ── MLX path ──────────────────────────────────────────────
    def _run_mlx(self, model, tokenizer, dataset) -> float:
        """Write a JSONL + call mlx_lm.lora.train via Python API."""
        from mlx_lm.lora import train as mlx_train
        from mlx_lm.tuner.trainer import TrainingArgs as MLXTrainArgs

        train_jsonl = os.path.join(self.output_dir, "_train.jsonl")
        import json
        with open(train_jsonl, "w") as f:
            for row in dataset:
                f.write(json.dumps({"text": row["text"]}) + "\n")

        args = MLXTrainArgs(
            batch_size=self.profile.per_device_batch_size,
            iters=self.config.get("training_args", {}).get("max_steps", 200),
            val_batches=0,
            steps_per_report=10,
            steps_per_eval=0,
            learning_rate=self.config.get("training_args", {}).get("learning_rate", 1e-4),
            grad_checkpoint=True,
            adapter_file=os.path.join(self.output_dir, "adapters.safetensors"),
        )

        logger.info("🍎 Starting MLX-LM LoRA training")
        final_loss = mlx_train(model=model, tokenizer=tokenizer, args=args, train=dataset)
        return float(final_loss) if final_loss is not None else 0.0


def _pick(cols: set[str], *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None
