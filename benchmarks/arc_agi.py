"""ARC-AGI benchmark: abstract 2D grid reasoning."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from benchmarks.base import Benchmark, BenchmarkResult


class ARCBenchmark(Benchmark):
    """ARC-AGI (Abstraction and Reasoning Corpus) benchmark.

    Evaluates the model's ability to solve 2D grid-based abstract
    reasoning tasks.  Each task provides a few input-output grid
    examples; the model must infer the transformation and apply it
    to a test input grid.

    Args:
        data_path: Path to the ARC-AGI dataset directory.
        split: Dataset split ("training", "evaluation", or "test").
    """

    def __init__(self, data_path: str, split: str = "evaluation") -> None:
        super().__init__(name="arc_agi", data_path=data_path)
        self.split: str = split
        self.tasks: List[Dict[str, Any]] = []

    def load(self) -> None:
        """Load ARC-AGI tasks from JSON files."""
        split_path = os.path.join(self.data_path, self.split)
        if not os.path.isdir(split_path):
            logger.warning(f"ARC-AGI data not found at {split_path}. Using dummy data.")
            self.tasks = self._dummy_tasks()
            return

        self.tasks = []
        for fname in sorted(os.listdir(split_path)):
            if fname.endswith(".json"):
                with open(os.path.join(split_path, fname), "r") as f:
                    task = json.load(f)
                    self.tasks.append(task)
        logger.info(f"Loaded {len(self.tasks)} ARC-AGI tasks from {split_path}")

    def _dummy_tasks(self, num: int = 4) -> List[Dict[str, Any]]:
        """Create dummy tasks for testing the harness."""
        tasks = []
        for i in range(num):
            grid = [[0, 1], [1, 0]]
            tasks.append({
                "train": [{"input": grid, "output": grid}],
                "test": [{"input": grid, "output": grid}],
            })
        return tasks

    def _grid_to_string(self, grid: List[List[int]]) -> str:
        """Convert a 2D grid to a string representation for prompting."""
        return "\n".join(" ".join(str(c) for c in row) for row in grid)

    def _check_grid_equal(
        self, pred: List[List[int]], target: List[List[int]]
    ) -> bool:
        """Check if two grids are identical."""
        try:
            pred_arr = np.array(pred, dtype=int)
            target_arr = np.array(target, dtype=int)
            return pred_arr.shape == target_arr.shape and bool(
                (pred_arr == target_arr).all()
            )
        except (ValueError, TypeError):
            return False

    def evaluate(
        self,
        model: Any,
        max_examples: Optional[int] = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Evaluate on ARC-AGI tasks.

        Args:
            model: A callable that accepts a prompt string and returns
                   a generated grid string (or a dict with 'grid' key).
            max_examples: Limit on number of tasks to evaluate.

        Returns:
            BenchmarkResult with accuracy metrics.
        """
        if not self.tasks:
            self.load()

        tasks = self.tasks[:max_examples] if max_examples else self.tasks
        correct: int = 0
        details: List[Dict[str, Any]] = []

        for idx, task in enumerate(tasks):
            train_examples = task.get("train", [])
            test_examples = task.get("test", [])

            for test_idx, test_pair in enumerate(test_examples):
                input_grid = test_pair["input"]
                target_grid = test_pair["output"]

                prompt = self._build_prompt(train_examples, input_grid)
                prediction = self._predict_grid(model, prompt)

                is_correct = self._check_grid_equal(prediction, target_grid)
                if is_correct:
                    correct += 1

                details.append({
                    "task_idx": idx,
                    "test_idx": test_idx,
                    "correct": is_correct,
                    "input_shape": np.array(input_grid).shape,
                    "target_shape": np.array(target_grid).shape,
                })

                if (idx + 1) % 10 == 0:
                    logger.info(
                        f"ARC-AGI [{idx + 1}/{len(tasks)}] "
                        f"accuracy={correct / max(1, len(details)):.3f}"
                    )

        total = len(details)
        accuracy = correct / max(1, total)
        logger.info(
            f"ARC-AGI final: {correct}/{total} = {accuracy:.4f}"
        )
        return BenchmarkResult(
            name=self.name,
            accuracy=accuracy,
            num_correct=correct,
            num_total=total,
            details=details,
        )

    def _build_prompt(
        self, train_examples: List[Dict[str, Any]], test_input: List[List[int]]
    ) -> str:
        """Build a prompt from few-shot examples and test input."""
        lines = ["You are given input-output grid pairs. Infer the rule."]
        for i, ex in enumerate(train_examples):
            lines.append(f"\nExample {i + 1}:")
            lines.append("Input:\n" + self._grid_to_string(ex["input"]))
            lines.append("Output:\n" + self._grid_to_string(ex["output"]))
        lines.append("\nTest input:\n" + self._grid_to_string(test_input))
        lines.append("\nProvide the output grid as rows of space-separated integers:")
        return "\n".join(lines)

    def _predict_grid(
        self, model: Any, prompt: str
    ) -> List[List[int]]:
        """Generate a grid prediction from the model.

        Attempts to parse the output as a 2D integer grid.
        Returns an empty grid on failure.
        """
        try:
            if hasattr(model, "generate"):
                output = model.generate(prompt, max_new_tokens=256)
            elif callable(model):
                output = model(prompt)
            else:
                output = str(model)

            if isinstance(output, dict):
                output = output.get("generated_text", str(output))

            output = str(output)
            return self._parse_grid(output)
        except Exception:
            return []

    @staticmethod
    def _parse_grid(text: str) -> List[List[int]]:
        """Parse a generated text into a 2D integer grid."""
        grid: List[List[int]] = []
        for line in text.strip().split("\n"):
            line = line.strip().strip("[](),.")
            if not line:
                continue
            try:
                row = [int(x) for x in line.replace(",", " ").split() if x.strip()]
                if row:
                    grid.append(row)
            except ValueError:
                continue
        return grid
