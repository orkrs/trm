"""ProntoQA benchmark: formal logical deduction."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from benchmarks.base import Benchmark, BenchmarkResult


class ProntoQABenchmark(Benchmark):
    """ProntoQA benchmark for formal logical deduction.

    Evaluates the model's ability to perform multi-step logical
    reasoning over a given set of facts and rules.

    Args:
        data_path: Path to the ProntoQA dataset file (JSONL).
        split: Dataset split ("train" or "test").
    """

    def __init__(self, data_path: str, split: str = "test") -> None:
        super().__init__(name="prontoqa", data_path=data_path)
        self.split: str = split
        self.examples: List[Dict[str, Any]] = []

    def load(self) -> None:
        """Load ProntoQA examples."""
        path = os.path.join(self.data_path, f"{self.split}.jsonl")
        if os.path.isfile(path):
            self.examples = []
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
            logger.info(f"Loaded {len(self.examples)} ProntoQA examples from {path}")
        else:
            logger.warning(f"ProntoQA data not found at {path}. Using dummy data.")
            self.examples = self._dummy_examples()

    def _dummy_examples(self, num: int = 4) -> List[Dict[str, Any]]:
        return [
            {
                "context": "All humans are mortal. Socrates is a human.",
                "question": "Is Socrates mortal?",
                "answer": "Yes",
            },
            {
                "context": "All birds can fly. Penguins are birds. Penguins cannot fly.",
                "question": "Is the statement consistent?",
                "answer": "No",
            },
        ] * (num // 2)

    @staticmethod
    def _normalise_answer(text: str) -> str:
        """Normalise a yes/no answer."""
        text = text.strip().lower()
        if re.search(r"\byes\b", text):
            return "yes"
        if re.search(r"\bno\b", text):
            return "no"
        if re.search(r"\btrue\b", text):
            return "yes"
        if re.search(r"\bfalse\b", text):
            return "no"
        return text

    def evaluate(
        self,
        model: Any,
        max_examples: Optional[int] = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Evaluate on ProntoQA logical deduction.

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
            context = ex.get("context", "")
            question = ex.get("question", "")
            expected_answer = ex.get("answer", "").strip()

            prompt = (
                f"Given the following facts:\n{context}\n\n"
                f"Question: {question}\n"
                f"Answer (Yes or No):"
            )

            try:
                if hasattr(model, "generate"):
                    output = model.generate(prompt, max_new_tokens=128)
                elif callable(model):
                    output = model(prompt)
                else:
                    output = str(model)
            except Exception as exc:
                output = f"ERROR: {exc}"

            if isinstance(output, dict):
                output = output.get("generated_text", str(output))

            output = str(output)
            pred_answer = self._normalise_answer(output)
            expected_norm = self._normalise_answer(expected_answer)
            is_correct = pred_answer == expected_norm

            if is_correct:
                correct += 1

            details.append({
                "idx": idx,
                "question": question,
                "predicted": pred_answer,
                "expected": expected_norm,
                "correct": is_correct,
            })

            if (idx + 1) % 10 == 0:
                logger.info(
                    f"ProntoQA [{idx + 1}/{len(examples)}] "
                    f"accuracy={correct / max(1, len(details)):.3f}"
                )

        total = len(details)
        accuracy = correct / max(1, total)
        logger.info(
            f"ProntoQA final: {correct}/{total} = {accuracy:.4f}"
        )
        return BenchmarkResult(
            name=self.name,
            accuracy=accuracy,
            num_correct=correct,
            num_total=total,
            details=details,
        )
