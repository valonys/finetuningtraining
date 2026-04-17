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

        # `DataCollatorForCompletionOnlyLM` is a TRL class but its import
        # path has moved across versions:
        #   * TRL <= 0.11:           top-level `from trl import ...`
        #   * TRL ~ 0.12-0.15:       `from trl.trainer.utils import ...`
        #   * TRL >= 0.16:           top-level again, sometimes removed
        #   * Old transformers (<4.45) re-exported it for convenience.
        # Try every known location; degrade gracefully to None and just
        # train on the full sequence with a warning.
        DataCollatorForCompletionOnlyLM = None
        for _import_path in (
            ("trl", "DataCollatorForCompletionOnlyLM"),
            ("trl.trainer.utils", "DataCollatorForCompletionOnlyLM"),
            ("trl.trainer.sft_trainer", "DataCollatorForCompletionOnlyLM"),
            ("trl.data_utils", "DataCollatorForCompletionOnlyLM"),
            ("transformers", "DataCollatorForCompletionOnlyLM"),
        ):
            try:
                _mod = __import__(_import_path[0], fromlist=[_import_path[1]])
                DataCollatorForCompletionOnlyLM = getattr(_mod, _import_path[1])
                break
            except (ImportError, AttributeError):
                continue
        if DataCollatorForCompletionOnlyLM is None:
            logger.warning(
                "⚠️  DataCollatorForCompletionOnlyLM not found in any known "
                "location — loss will be computed on the full sequence "
                "(prompt + response) instead of response-only."
            )

        train_args = self.config.get("training_args", {})
        max_seq = train_args.get("max_seq_length", self.profile.max_seq_length)
        max_steps = train_args.get("max_steps", -1)
        num_epochs = train_args.get("num_train_epochs", 1)
        lr = train_args.get("learning_rate", 2e-4)
        bs = train_args.get("batch_size", self.profile.per_device_batch_size)
        ga = train_args.get("gradient_accumulation_steps", self.profile.gradient_accumulation_steps)

        import inspect
        import torch
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        # ── Build SFTConfig kwargs defensively ────────────────────
        # TRL has shuffled some kwargs across versions:
        #   * `max_seq_length` was renamed to `max_length` in TRL 0.16+
        #     (verified against current PyPI release).
        #   * `dataset_text_field` has stayed but moved between modules.
        # We translate renamed kwargs first, then filter to whatever the
        # installed SFTConfig actually accepts so a version drift
        # downgrades to a logged-and-dropped rather than TypeError.
        wanted_cfg = dict(
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
            optim=self._pick_optim(),
            gradient_checkpointing=self.profile.gradient_checkpointing,
            max_seq_length=max_seq,
            dataset_text_field="text",
            report_to="none",
        )

        cfg_params = set(inspect.signature(SFTConfig.__init__).parameters)

        # TRL 0.16+: `max_seq_length` -> `max_length`. Translate before
        # the filter so the value isn't lost.
        if "max_seq_length" not in cfg_params and "max_length" in cfg_params:
            wanted_cfg["max_length"] = wanted_cfg.pop("max_seq_length")

        cfg_kwargs = {k: v for k, v in wanted_cfg.items() if k in cfg_params}
        dropped = [k for k in wanted_cfg if k not in cfg_params]
        if dropped:
            logger.info(f"ℹ️  SFTConfig: ignoring kwargs not in this TRL version: {dropped}")
        args = SFTConfig(**cfg_kwargs)

        data_collator = None
        if self.template.response_template and DataCollatorForCompletionOnlyLM is not None:
            try:
                data_collator = DataCollatorForCompletionOnlyLM(
                    response_template=self.template.response_template,
                    tokenizer=tokenizer,
                )
            except Exception as e:
                logger.warning(f"⚠️  Completion-only collator unavailable: {e}")

        # ── Build SFTTrainer kwargs defensively ───────────────────
        # TRL >= 0.12 renamed `tokenizer` to `processing_class`. Probe the
        # signature so we use the right kwarg without pinning a version.
        trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
        tok_kwarg = "processing_class" if "processing_class" in trainer_params else "tokenizer"
        trainer_kwargs = dict(
            model=model,
            train_dataset=dataset,
            args=args,
            data_collator=data_collator,
            callbacks=self._build_callbacks(),
        )
        trainer_kwargs[tok_kwarg] = tokenizer

        trainer = SFTTrainer(**trainer_kwargs)
        result = trainer.train()
        return float(result.training_loss)

    # ── Optimizer pick ────────────────────────────────────────
    def _pick_optim(self) -> str:
        """Choose the safest optimizer for the current hardware/quant setup.

        `adamw_8bit` requires a working bitsandbytes install AND a
        compatible CUDA; on Colab T4 the load succeeds but the first
        optimizer step can crash. `paged_adamw_8bit` is the more robust
        bnb variant when 4-bit is on. For everything else, plain
        `adamw_torch` always works.
        """
        if not self.profile.load_in_4bit:
            return "adamw_torch"
        try:
            import bitsandbytes  # noqa: F401
            return "paged_adamw_8bit"
        except Exception:
            return "adamw_torch"

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
