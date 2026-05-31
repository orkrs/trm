"""GSM8K benchmark: grade-school math word problems."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from benchmarks.base import Benchmark, BenchmarkResult


class GSM8KBenchmark(Benchmark):
    """GSM8K (Grade School Math 8K) benchmark.

    Evaluates mathematical reasoning by parsing the model's
    generated answer and comparing against the ground-truth
    numeric answer.

    Args:
        data_path: Path to the GSM8K dataset file (JSONL).
        split: Dataset split ("train" or "test").
    """

    def __init__(self, data_path: str, split: str = "test") -> None:
        super().__init__(name="gsm8k", data_path=data_path)
        self.split: str = split
        self.examples: List[Dict[str, str]] = []

    def load(self) -> None:
        """Load GSM8K examples from a JSONL file."""
        path = os.path.join(self.data_path, f"{self.split}.jsonl")
        if not os.path.isfile(path):
            logger.warning(f"GSM8K data not found at {path}. Using dummy data.")
            self.examples = self._dummy_examples()
            return

        self.examples = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        logger.info(f"Loaded {len(self.examples)} GSM8K examples from {path}")

    def _dummy_examples(self, num: int = 4) -> List[Dict[str, str]]:
        return [
            {
                "question": "What is 2 + 2?",
                "answer": "4",
            },
            {
                "question": "If John has 5 apples and gives 2 away, how many does he have?",
                "answer": "3",
            },
        ] * (num // 2)

    @staticmethod
    def _extract_answer(text: str) -> Optional[float]:
        """Extract the final numeric answer from generated text."""
        patterns = [
            r"answer is\s*\$?(-?\d+(?:\.\d+)?)",
            r"answer:\s*\$?(-?\d+(?:\.\d+)?)",
            r"####\s*(-?\d+(?:\.\d+)?)",
            r"result is\s*(-?\d+(?:\.\d+)?)",
            r"= (-?\d+(?:\.\d+)?)",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _extract_ground_truth(answer: str) -> Optional[float]:
        """Extract numeric answer from the ground truth string."""
        match = re.search(r"(-?\d+(?:\.\d+)?)", answer)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def evaluate(
        self,
        model: Any,
        max_examples: Optional[int] = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Evaluate on GSM8K math problems.

        Args:
            model: A callable that accepts a prompt and returns text.
            max_examples: Limit on number of examples.

        Returns:
            BenchmarkResult with accuracy metrics.
        """
        if not self.examples:
            self.load()

        examples = self.examples[:max_examples] if max_examples else self.examples
        correct: int = 0
        details: List[Dict[str, Any]] = []

        for idx, ex in enumerate(examples):
            question = ex["question"]
            ground_truth = self._extract_ground_truth(ex["answer"])
            prompt = f"Solve the following math problem step by step.\n\nQuestion: {question}\nAnswer:"

            try:
                if hasattr(model, "generate"):
                    output = model.generate(prompt, max_new_tokens=512)
                elif callable(model):
                    output = model(prompt)
                else:
                    output = str(model)
            except Exception as exc:
                output = f"ERROR: {exc}"

            if isinstance(output, dict):
                output = output.get("generated_text", str(output))

            output = str(output)
            pred_answer = self._extract_answer(output)

            is_correct = False
            if pred_answer is not None and ground_truth is not None:
                is_correct = abs(pred_answer - ground_truth) < 0.01

            if is_correct:
                correct += 1

            details.append({
                "idx": idx,
                "question": question,
                "predicted": pred_answer,
                "ground_truth": ground_truth,
                "correct": is_correct,
            })

            if (idx + 1) % 20 == 0:
                logger.info(
                    f"GSM8K [{idx + 1}/{len(examples)}] "
                    f"accuracy={correct / max(1, len(details)):.3f}"
                )

        total = len(details)
        accuracy = correct / max(1, total)
        logger.info(
            f"GSM8K final: {correct}/{total} = {accuracy:.4f}"
        )
        return BenchmarkResult(
            name=self.name,
            accuracy=accuracy,
            num_correct=correct,
            num_total=total,
            details=details,
        )
