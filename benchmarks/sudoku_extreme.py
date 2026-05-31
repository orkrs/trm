"""Sudoku-Extreme benchmark: constraint satisfaction."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

from loguru import logger

from benchmarks.base import Benchmark, BenchmarkResult


class SudokuExtremeBenchmark(Benchmark):
    """Sudoku-Extreme benchmark.

    Tests the model's ability to solve Sudoku puzzles of varying
    difficulty using constraint satisfaction.  The model must
    output a complete 9x9 grid that satisfies all Sudoku rules.

    Args:
        data_path: Path to the Sudoku dataset directory.
        num_puzzles: Number of puzzles to generate (if no dataset).
    """

    def __init__(self, data_path: str, num_puzzles: int = 10) -> None:
        super().__init__(name="sudoku_extreme", data_path=data_path)
        self.num_puzzles: int = num_puzzles
        self.puzzles: List[Dict[str, Any]] = []

    def load(self) -> None:
        """Load or generate Sudoku puzzles."""
        if os.path.isdir(self.data_path):
            logger.info(f"Loading Sudoku puzzles from {self.data_path}")
        else:
            logger.warning(f"Sudoku data not found. Generating {self.num_puzzles} puzzles.")
            self.puzzles = self._generate_puzzles(self.num_puzzles)

    @staticmethod
    def _generate_puzzles(num: int) -> List[Dict[str, Any]]:
        """Generate synthetic Sudoku puzzles with solutions.

        For testing purposes; real evaluation should use curated puzzles.

        Args:
            num: Number of puzzles to generate.

        Returns:
            List of puzzle dicts with 'puzzle' and 'solution' keys.
        """
        puzzles = []
        for _ in range(num):
            puzzle, solution = SudokuExtremeBenchmark._generate_sudoku()
            puzzles.append({"puzzle": puzzle, "solution": solution})
        return puzzles

    @staticmethod
    def _generate_sudoku() -> tuple:
        """Generate a random solved Sudoku board and a puzzle.

        Returns:
            Tuple of (puzzle_grid, solution_grid) as 9x9 lists.
        """
        base = list(range(9))
        random.shuffle(base)
        solution = [[base[(i * 3 + i // 3 + j) % 9] for j in range(9)] for i in range(9)]

        puzzle = [row[:] for row in solution]
        clues = 30
        cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(cells)
        for r, c in cells[clues:]:
            puzzle[r][c] = 0
        return puzzle, solution

    @staticmethod
    def _is_valid_sudoku(grid: List[List[int]]) -> bool:
        """Check if a 9x9 grid satisfies all Sudoku constraints."""
        if len(grid) != 9 or any(len(row) != 9 for row in grid):
            return False

        for i in range(9):
            row_vals = [v for v in grid[i] if v != 0]
            if len(row_vals) != len(set(row_vals)):
                return False
            col_vals = [grid[j][i] for j in range(9) if grid[j][i] != 0]
            if len(col_vals) != len(set(col_vals)):
                return False

        for br in range(3):
            for bc in range(3):
                block_vals = []
                for r in range(br * 3, br * 3 + 3):
                    for c in range(bc * 3, bc * 3 + 3):
                        if grid[r][c] != 0:
                            block_vals.append(grid[r][c])
                if len(block_vals) != len(set(block_vals)):
                    return False
        return True

    @staticmethod
    def _grid_to_string(grid: List[List[int]]) -> str:
        return "\n".join(" ".join(str(v) if v != 0 else "." for v in row) for row in grid)

    @staticmethod
    def _parse_grid(text: str) -> Optional[List[List[int]]]:
        """Parse model output into a 9x9 grid."""
        grid = []
        for line in text.strip().split("\n"):
            line = line.strip().strip("[](),.")
            if not line:
                continue
            try:
                row = [int(x) for x in line.replace(",", " ").split() if x.strip() and x.strip() != "."]
                if len(row) == 9:
                    grid.append(row)
            except ValueError:
                continue
        return grid if len(grid) == 9 else None

    def evaluate(
        self,
        model: Any,
        max_examples: Optional[int] = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Evaluate on Sudoku puzzles.

        Args:
            model: A callable that accepts a prompt and returns grid text.
            max_examples: Limit on number of puzzles.

        Returns:
            BenchmarkResult with accuracy metrics.
        """
        if not self.puzzles:
            self.load()

        puzzles = self.puzzles[:max_examples] if max_examples else self.puzzles
        correct: int = 0
        valid: int = 0
        details: List[Dict[str, Any]] = []

        for idx, puzzle_data in enumerate(puzzles):
            puzzle = puzzle_data["puzzle"]
            solution = puzzle_data["solution"]

            prompt = (
                "Solve this Sudoku puzzle. Fill all empty cells (marked as 0).\n"
                "Return the complete 9x9 grid as space-separated rows.\n\n"
                f"Puzzle:\n{self._grid_to_string(puzzle)}\n\nSolution:"
            )

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
            parsed = self._parse_grid(output)

            is_valid = parsed is not None and self._is_valid_sudoku(parsed)
            is_correct = is_valid and parsed == solution
            if is_valid:
                valid += 1
            if is_correct:
                correct += 1

            details.append({
                "idx": idx,
                "valid": is_valid,
                "correct": is_correct,
            })

            if (idx + 1) % 5 == 0:
                logger.info(
                    f"Sudoku [{idx + 1}/{len(puzzles)}] "
                    f"accuracy={correct / max(1, len(details)):.3f}"
                )

        total = len(details)
        accuracy = correct / max(1, total)
        logger.info(
            f"Sudoku final: {correct}/{total} = {accuracy:.4f}"
        )
        return BenchmarkResult(
            name=self.name,
            accuracy=accuracy,
            num_correct=correct,
            num_total=total,
            details=details,
            extra={"num_valid": valid},
        )
