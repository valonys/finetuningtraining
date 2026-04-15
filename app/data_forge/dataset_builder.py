"""
DatasetBuilder — the final step of the Data Forge.

Given a list of IngestedRecord (from parsers) and a target **task**
(sft / dpo / grpo), produce a HF `Dataset` whose rows are in the native
HF-TRL schema for that task, with the **correct chat template** applied
for the chosen base model.

SFT schema (post-tokenizer):
    {"text": "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n..."}

DPO schema (raw — TRL tokenises internally):
    {"prompt": str, "chosen": str, "rejected": str}

GRPO schema:
    {"prompt": str, "ground_truth": str}

Conversational schema (messages list, works across SFT/DPO if TRL version is new enough):
    {"messages": [{"role": "system|user|assistant", "content": str}, ...]}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

logger = logging.getLogger(__name__)

Task = Literal["sft", "dpo", "grpo", "kto", "orpo"]


@dataclass
class DatasetBuilder:
    task: Task = "sft"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: str = "You are a helpful assistant."
    synth_qa: bool = True
    synth_mode: str = "auto"          # "auto" | "rule_based" | "llm"
    target_size: Optional[int] = None
    chunk_target_chars: int = 1200
    chunk_max_chars: int = 1800
    template_override: Optional[str] = None    # explicit template name, or None = auto
    filter_noise: bool = True                   # skip metadata/TOC/cover chunks

    def build(self, records: Iterable):
        from datasets import Dataset
        from .chunker import chunk_records
        from .qa_synthesis import synthesize_qa
        from .pair_synthesis import synthesize_pairs
        from .chunk_filter import filter_chunks
        from app.templates import get_template_for

        # Template resolution:
        #  - If `template_override` is set, use that template literally.
        #  - Otherwise, auto-resolve from the base model id.
        # This gives the UI a dropdown for power users who want to
        # train, e.g., Qwen data with the Alpaca template, without
        # having to rename the base model.
        if self.template_override:
            template = get_template_for(self.base_model, template=self.template_override)
            logger.info(
                f"🧩 Dataset template: {template.name} "
                f"(explicit override; base model={self.base_model})"
            )
        else:
            template = get_template_for(self.base_model)
            logger.info(
                f"🧩 Dataset template: {template.name} (auto from model={self.base_model})"
            )

        # Chunk the docs
        chunks = chunk_records(
            records,
            target_chars=self.chunk_target_chars,
            max_chars=self.chunk_max_chars,
        )
        if not chunks:
            raise ValueError("Data Forge: no chunks produced -- is the input empty?")
        logger.info(f"🧱 Chunked into {len(chunks)} pieces")

        # Filter noise (cover pages, TOCs, indexes, bibliography fragments,
        # all-caps headers, digit-dense page-number strips) BEFORE Q/A
        # synthesis. Without this, rule-based and LLM synthesis alike
        # produce trivia like "who wrote this book?" and "how many
        # chapters are there?" from front-matter scraps.
        if self.filter_noise:
            kept_chunks, stats = filter_chunks(chunks)
            logger.info(
                f"🧹 Noise filter: kept {len(kept_chunks)}/{len(chunks)} chunks "
                f"(dropped: {stats['dropped_count']}, reasons: {stats['reasons']})"
            )
            chunks = kept_chunks
        if not chunks:
            raise ValueError(
                "Data Forge: every chunk was rejected by the noise filter. "
                "Input may be entirely metadata/TOC/index with no body text. "
                "Disable filtering with `filter_noise=False` to bypass."
            )

        # Phase 1: derive base (instruction, response) seeds from chunks
        if self.synth_qa:
            seeds = synthesize_qa(chunks, mode=self.synth_mode)
        else:
            seeds = [{
                "instruction": "Summarise the following passage.",
                "response": c["chunk"],
                "source": c.get("source", ""),
            } for c in chunks]

        if self.target_size:
            seeds = seeds[: self.target_size]

        # Phase 2: task-specific shaping
        if self.task == "sft":
            rows = [self._to_sft_row(p, template) for p in seeds]
        elif self.task == "grpo":
            rows = [self._to_grpo_row(p, template) for p in seeds]
        elif self.task == "kto":
            rows = [self._to_kto_row(p, template) for p in seeds]
        elif self.task in ("dpo", "orpo"):
            # Upgrade seeds → real (chosen, rejected) pairs via the synth
            # provider. Set OLLAMA_API_KEY for Nemotron on Ollama Cloud.
            contrastive = synthesize_pairs(seeds)
            rows = [self._to_preference_row(p, template) for p in contrastive]
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        ds = Dataset.from_list(rows)
        logger.info(f"✅ Built {self.task} dataset with {len(ds)} examples")
        return ds

    # ── Row builders ──────────────────────────────────────────
    def _to_sft_row(self, pair: dict, template) -> dict:
        formatted = template.format_sft(
            system=self.system_prompt,
            instruction=pair["instruction"],
            response=pair["response"],
        )
        return {
            "text": formatted,
            "messages": template.as_messages(
                system=self.system_prompt,
                instruction=pair["instruction"],
                response=pair["response"],
            ),
            "source": pair.get("source", ""),
        }

    def _to_preference_row(self, pair: dict, template) -> dict:
        """
        Turn a contrastive pair from `pair_synthesis.synthesize_pairs` into
        a TRL-ready preference row. The heavy lifting (generating a plausible
        rejected response) has already happened upstream via the LLM provider
        — this just applies the chat template for the base model.

        Shared between DPO and ORPO which both consume `{prompt, chosen, rejected}`.
        """
        prompt = template.format_prompt_only(
            system=self.system_prompt,
            instruction=pair["instruction"],
        )
        return {
            "prompt": prompt,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "source": pair.get("source", ""),
            "synth": pair.get("synth", "unknown"),
        }

    def _to_grpo_row(self, pair: dict, template) -> dict:
        prompt = template.format_prompt_only(
            system=self.system_prompt,
            instruction=pair["instruction"],
        )
        return {
            "prompt": prompt,
            "ground_truth": pair["response"],
            "source": pair.get("source", ""),
        }

    def _to_kto_row(self, pair: dict, template) -> dict:
        prompt = template.format_prompt_only(
            system=self.system_prompt,
            instruction=pair["instruction"],
        )
        return {
            "prompt": prompt,
            "completion": pair["response"],
            "label": True,          # all examples positive; generate negatives separately
            "source": pair.get("source", ""),
        }
