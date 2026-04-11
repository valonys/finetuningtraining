"""Training engines — SFT / DPO / ORPO / KTO / GRPO with backend auto-selection."""
from .sft_trainer import AgnosticSFTTrainer
from .dpo_trainer import AgnosticDPOTrainer
from .orpo_trainer import AgnosticORPOTrainer
from .kto_trainer import AgnosticKTOTrainer
from .grpo_trainer import AgnosticGRPOTrainer

__all__ = [
    "AgnosticSFTTrainer",
    "AgnosticDPOTrainer",
    "AgnosticORPOTrainer",
    "AgnosticKTOTrainer",
    "AgnosticGRPOTrainer",
]
