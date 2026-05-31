"""HumanEval benchmark: code synthesis from docstrings."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from loguru import logger

from benchmarks.base import Benchmark, BenchmarkResult


class HumanEvalBenchmark(Benchmark):
    """HumanEval benchmark for code generation.

    Evaluates the model's ability to synthesise Python functions
    from docstrings.  Correctness is measured by running the
    provided unit tests.

    Args:
        data_path: Path to the HumanEval dataset file (JSONL).
    """

    def __init__(self, data_path: str) -> None:
        super().__init__(name="humaneval", data_path=data_path)
        self.problems: List[Dict[str, Any]] = []

    def load(self) -> None:
        """Load HumanEval problems."""
        path = os.path.join(self.data_path, "HumanEval.jsonl")
        if not os.path.isfile(path):
            logger.warning(f"HumanEval data not found at {path}. Using dummy data.")
            self.problems = self._dummy_problems()
            return

        self.problems = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    self.problems.append(json.loads(line))
        logger.info(f"Loaded {len(self.problems)} HumanEval problems from {path}")

    def _dummy_problems(self, num: int = 2) -> List[Dict[str, Any]]:
        return [
            {
                "task_id": "test/0",
                "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
                "entry_point": "add",
                "test": "def test_add():\n    assert add(1, 2) == 3\n",
            },
            {
                "task_id": "test/1",
                "prompt": "def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n",
                "entry_point": "is_even",
                "test": "def test_is_even():\n    assert is_even(2)\n    assert not is_even(3)\n",
            },
        ] * (num // 2)

    @staticmethod
    def _run_test(code: str, test: str, entry_point: str) -> bool:
        """Execute the generated function against its test.

        Args:
            code: Generated function implementation.
            test: Test code string.
            entry_point: Name of the function under test.

        Returns:
            True if all tests pass, False otherwise.
        """
        try:
            namespace: Dict[str, Any] = {}
            exec(code + "\n" + test, namespace)
            namespace[f"test_{entry_point}"]()
            return True
        except Exception:
            return False

    def evaluate(
        self,
        model: Any,
        max_examples: Optional[int] = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Evaluate on HumanEval code synthesis.

        Args:
            model: A callable that accepts a prompt and returns code.
            max_examples: Limit on number of problems.

        Returns:
            BenchmarkResult with pass@1 accuracy.
        """
        if not self.problems:
            self.load()

        problems = self.problems[:max_examples] if max_examples else self.problems
        correct: int = 0
        details: List[Dict[str, Any]] = []

        for idx, prob in enumerate(problems):
            prompt = prob["prompt"]
            test = prob["test"]
            entry_point = prob["entry_point"]

            try:
                if hasattr(model, "generate"):
                    generated = model.generate(prompt, max_new_tokens=512)
                elif callable(model):
                    generated = model(prompt)
                else:
                    generated = str(model)
            except Exception as exc:
                generated = f"ERROR: {exc}"

            if isinstance(generated, dict):
                generated = generated.get("generated_text", str(generated))

            generated = str(generated)
            passed = self._run_test(generated, test, entry_point)
            if passed:
                correct += 1

            details.append({
                "task_id": prob.get("task_id", idx),
                "entry_point": entry_point,
                "passed": passed,
            })

            if (idx + 1) % 10 == 0:
                logger.info(
                    f"HumanEval [{idx + 1}/{len(problems)}] "
                    f"pass@1={correct / max(1, len(details)):.3f}"
                )

        total = len(details)
        accuracy = correct / max(1, total)
        logger.info(
            f"HumanEval final: {correct}/{total} = {accuracy:.4f}"
        )
        return BenchmarkResult(
            name=self.name,
            accuracy=accuracy,
            num_correct=correct,
            num_total=total,
            details=details,
        )
