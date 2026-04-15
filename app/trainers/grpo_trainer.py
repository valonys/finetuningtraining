"""
GRPO (Group Relative Policy Optimization) trainer.

Preserves the battle-tested `GSM8KRewardSignal` from v2.0 (graduated, tolerant
of parse failures) and adds a pluggable reward-function slot so users can
supply their own verifier (code unit tests, function-calling sandboxes, etc.).

Dataset schema:
    {"prompt": str, "ground_truth": str}
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .base import BaseAgnosticTrainer, TrainRequest
from .reward_signals import GSM8KRewardSignal, RewardSignal

logger = logging.getLogger(__name__)


class AgnosticGRPOTrainer(BaseAgnosticTrainer):
    method = "grpo"

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        base_model_id: str,
        dataset_path: Optional[str] = None,
        hf_dataset_config: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        reward_signal: Optional[RewardSignal] = None,
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
        self.reward_signal: RewardSignal = reward_signal or GSM8KRewardSignal()

    def _format_dataset(self, dataset, tokenizer):
        cols = set(dataset.column_names)
        if {"prompt", "ground_truth"}.issubset(cols):
            return dataset

        # Accept input/output and wrap the instruction into a proper prompt
        in_col = "input" if "input" in cols else ("instruction" if "instruction" in cols else None)
        out_col = "output" if "output" in cols else ("response" if "response" in cols else None)
        if not in_col or not out_col:
            raise ValueError(f"GRPO dataset needs (input/output) or (prompt/ground_truth); got {cols}")

        sys_prompt = self.config.get("system_prompt", "You are a helpful assistant.")
        template = self.template

        def _fmt(ex):
            return {
                "prompt": template.format_prompt_only(system=sys_prompt, instruction=ex[in_col]),
                "ground_truth": ex[out_col],
            }
        return dataset.map(_fmt, remove_columns=list(cols))

    def _run(self, model, tokenizer, dataset) -> float:
        from trl import GRPOTrainer, GRPOConfig

        args = self.config.get("grpo_args", {})
        cfg = GRPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=args.get("num_train_epochs", 2),
            per_device_train_batch_size=args.get(
                "per_device_train_batch_size", self.profile.per_device_batch_size
            ),
            gradient_accumulation_steps=args.get(
                "gradient_accumulation_steps", max(16, self.profile.gradient_accumulation_steps)
            ),
            learning_rate=args.get("learning_rate", 5e-6),
            num_generations=args.get("num_generations", 8),
            temperature=args.get("temperature", 0.8),
            max_new_tokens=args.get("max_new_tokens", 400),
            logging_steps=10,
            save_steps=100,
            report_to="none",
        )

        reward_signal = self.reward_signal

        def reward_fn(completions: List[str], **kwargs) -> List[float]:
            ground_truths = kwargs.get("ground_truth", [""] * len(completions))
            return [
                reward_signal.compute_reward(c, g)
                for c, g in zip(completions, ground_truths)
            ]

        trainer = GRPOTrainer(
            model=model,
            args=cfg,
            train_dataset=dataset,
            reward_funcs=reward_fn,
            callbacks=self._build_callbacks(),
        )
        trainer.train()
        # GRPOTrainer doesn't always return a loss — read from state
        try:
            final = trainer.state.log_history[-1].get("loss")
        except Exception:
            final = 0.0
        return float(final or 0.0)

    def _save(self, model, tokenizer):
        # GRPO uses trainer.save_model; the default save_pretrained also works.
        super()._save(model, tokenizer)
