"""
Reward signals for GRPO.

Ships with:
  * `GSM8KRewardSignal`    — math verification, graduated rewards (from v2.0)
  * `ExactMatchReward`     — case-insensitive string comparison
  * `RegexExtractReward`   — extract via regex then compare
  * `CodeUnitTestReward`   — run Python unit tests in a subprocess (sandbox)

Users can implement their own by subclassing `RewardSignal` and passing an
instance to `AgnosticGRPOTrainer(reward_signal=...)`.
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RewardSignal(ABC):
    name: str = "base"

    @abstractmethod
    def compute_reward(self, response: str, ground_truth: str) -> float: ...


# ──────────────────────────────────────────────────────────────
class GSM8KRewardSignal(RewardSignal):
    """Graduated reward system from v2.0 — robust to malformed outputs."""
    name = "gsm8k"

    def extract_numerical_answer(self, text: str) -> Optional[float]:
        if "####" in text:
            try:
                return float(text.split("####")[-1].strip().replace(",", "").replace("$", ""))
            except ValueError:
                pass
        patterns = [
            r"(?:The answer is|answer:|Answer:)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"(?:equals?|=)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"(?:total|sum|result|Final answer)\s*(?:is|:|=)?\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"Therefore,?\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
            r"\\boxed\{([+-]?\d+(?:\.\d+)?)\}",
        ]
        for p in patterns:
            matches = re.findall(p, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                try:
                    return float(matches[-1].replace(",", "").replace("$", ""))
                except ValueError:
                    continue
        numbers = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass
        return None

    def analyze_quality(self, response: str) -> Dict[str, bool | int]:
        calc_indicators = ["=", "+", "-", "*", "/", "multiply", "divide", "add", "subtract"]
        reasoning_words = ["therefore", "because", "since", "so", "thus", "hence",
                           "first", "then", "next", "finally", "step"]
        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
        return {
            "has_calculation": any(i in response.lower() for i in calc_indicators),
            "has_steps":       len(sentences) >= 2 or "step" in response.lower(),
            "has_reasoning":   any(w in response.lower() for w in reasoning_words),
            "has_numbers":     bool(re.search(r"\d", response)),
            "length":          len(response),
            "sentences":       len(sentences),
        }

    def compute_reward(self, response: str, ground_truth: str) -> float:
        correct = self.extract_numerical_answer(ground_truth)
        predicted = self.extract_numerical_answer(response)
        quality = self.analyze_quality(response)

        if predicted is None:
            if quality["length"] < 20:
                return 0.0
            reward = 0.05
            if quality["has_calculation"]: reward += 0.05
            if quality["has_steps"]:       reward += 0.05
            if quality["has_numbers"]:     reward += 0.05
            return min(0.3, reward)

        if correct is None:
            return 0.1

        if abs(predicted - correct) < 0.001:
            reward = 1.0
            if quality["has_steps"]:       reward += 0.10
            if quality["length"] > 100:    reward += 0.10
            if quality["has_reasoning"]:   reward += 0.05
            if quality["has_calculation"]: reward += 0.05
            return min(1.3, reward)

        rel_error = abs(predicted - correct) / abs(correct) if correct != 0 else abs(predicted)
        if   rel_error < 0.01:  reward = 0.90
        elif rel_error < 0.05:  reward = 0.70
        elif rel_error < 0.10:  reward = 0.50
        elif rel_error < 0.30:  reward = 0.30
        else:                   reward = 0.10

        if quality["has_calculation"] and quality["has_steps"]: reward += 0.10
        elif quality["has_calculation"]:                        reward += 0.05
        if quality["length"] > 100:                             reward += 0.05
        return min(0.9, max(0.1, reward))


# ──────────────────────────────────────────────────────────────
class ExactMatchReward(RewardSignal):
    name = "exact_match"

    def compute_reward(self, response: str, ground_truth: str) -> float:
        return 1.0 if response.strip().lower() == ground_truth.strip().lower() else 0.0


class RegexExtractReward(RewardSignal):
    name = "regex_extract"

    def __init__(self, pattern: str, flags: int = 0):
        self.pattern = re.compile(pattern, flags)

    def compute_reward(self, response: str, ground_truth: str) -> float:
        m = self.pattern.search(response)
        if not m:
            return 0.0
        extracted = (m.group(1) if m.groups() else m.group(0)).strip()
        return 1.0 if extracted.lower() == ground_truth.strip().lower() else 0.0


class CodeUnitTestReward(RewardSignal):
    """Reward Python code by running it against a unit-test `ground_truth`."""
    name = "code_unit_test"

    def __init__(self, timeout_s: int = 10):
        self.timeout_s = timeout_s

    def compute_reward(self, response: str, ground_truth: str) -> float:
        # response: candidate Python function(s)
        # ground_truth: `assert ...` statements referring to symbols in response
        script = f"{response}\n\n{ground_truth}\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            path = f.name
        try:
            out = subprocess.run(
                ["python", path],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            return 1.0 if out.returncode == 0 else 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception as e:
            logger.warning(f"⚠️  code reward error: {e}")
            return 0.0
