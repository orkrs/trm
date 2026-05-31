"""Unit tests for benchmark evaluation scaffolding."""

from __future__ import annotations

import pytest

from benchmarks.base import Benchmark, BenchmarkResult
from benchmarks.arc_agi import ARCBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.humaneval import HumanEvalBenchmark
from benchmarks.sudoku_extreme import SudokuExtremeBenchmark
from benchmarks.prontoqa import ProntoQABenchmark


class MockModel:
    """Mock model that returns a fixed response for any prompt."""

    def __init__(self, response: str = "42") -> None:
        self.response = response

    def generate(self, prompt: str, **kwargs: str) -> str:
        return self.response

    def __call__(self, prompt: str) -> str:
        return self.response


class TestBenchmarkBase:
    """Verification suite for Benchmark base class."""

    def test_benchmark_result_dataclass(self) -> None:
        """BenchmarkResult stores attributes correctly."""
        result = BenchmarkResult(
            name="test", accuracy=0.75, num_correct=3, num_total=4,
        )
        assert result.name == "test"
        assert result.accuracy == 0.75
        assert result.num_correct == 3
        assert result.num_total == 4

    def test_benchmark_abstract(self) -> None:
        """Cannot instantiate Benchmark directly."""
        with pytest.raises(TypeError):
            Benchmark(name="x", data_path="/tmp")  # type: ignore


class TestARCBenchmark:
    """Verification suite for ARC-AGI benchmark."""

    def test_load_generates_dummy(self) -> None:
        """ARCBenchmark loads dummy data when path is missing."""
        bm = ARCBenchmark(data_path="/nonexistent/path")
        bm.load()
        assert len(bm.tasks) > 0

    def test_evaluate_returns_benchmark_result(self) -> None:
        """ARCBenchmark.evaluate returns a BenchmarkResult."""
        bm = ARCBenchmark(data_path="/nonexistent/path")
        bm.load()
        model = MockModel(response="0 1\n1 0")
        result = bm.evaluate(model, max_examples=2)
        assert isinstance(result, BenchmarkResult)
        assert result.name == "arc_agi"

    def test_grid_equal(self) -> None:
        """_check_grid_equal correctly compares grids."""
        bm = ARCBenchmark(data_path="/nonexistent/path")
        assert bm._check_grid_equal([[0, 1], [1, 0]], [[0, 1], [1, 0]])
        assert not bm._check_grid_equal([[0, 1], [1, 0]], [[0, 0], [1, 0]])

    def test_grid_to_string(self) -> None:
        """_grid_to_string produces correct string."""
        bm = ARCBenchmark(data_path="/nonexistent/path")
        s = bm._grid_to_string([[0, 1], [1, 0]])
        assert "0 1" in s
        assert "1 0" in s


class TestGSM8KBenchmark:
    """Verification suite for GSM8K benchmark."""

    def test_load_generates_dummy(self) -> None:
        """GSM8K loads dummy data when path is missing."""
        bm = GSM8KBenchmark(data_path="/nonexistent/path")
        bm.load()
        assert len(bm.examples) > 0

    def test_evaluate_returns_benchmark_result(self) -> None:
        """GSM8K evaluation returns a BenchmarkResult."""
        bm = GSM8KBenchmark(data_path="/nonexistent/path")
        bm.load()
        model = MockModel(response="The answer is 4")
        result = bm.evaluate(model, max_examples=2)
        assert isinstance(result, BenchmarkResult)
        assert result.name == "gsm8k"

    def test_extract_answer(self) -> None:
        """_extract_answer finds numeric answers in text."""
        assert GSM8KBenchmark._extract_answer("The answer is 42") == 42.0
        assert GSM8KBenchmark._extract_answer("#### 3.14") == 3.14
        assert GSM8KBenchmark._extract_answer("No number here") is None


class TestHumanEvalBenchmark:
    """Verification suite for HumanEval benchmark."""

    def test_load_generates_dummy(self) -> None:
        """HumanEval loads dummy problems when path is missing."""
        bm = HumanEvalBenchmark(data_path="/nonexistent/path")
        bm.load()
        assert len(bm.problems) > 0

    def test_evaluate_returns_benchmark_result(self) -> None:
        """HumanEval evaluation returns a BenchmarkResult."""
        bm = HumanEvalBenchmark(data_path="/nonexistent/path")
        bm.load()
        model = MockModel(response="def add(a, b): return a + b")
        result = bm.evaluate(model, max_examples=1)
        assert isinstance(result, BenchmarkResult)
        assert result.name == "humaneval"

    def test_run_test_pass(self) -> None:
        """_run_test returns True for correct code."""
        code = "def add(a, b): return a + b"
        test = "def test_add():\n    assert add(1, 2) == 3"
        assert HumanEvalBenchmark._run_test(code, test, "add")

    def test_run_test_fail(self) -> None:
        """_run_test returns False for incorrect code."""
        code = "def add(a, b): return a - b"
        test = "def test_add():\n    assert add(1, 2) == 3"
        assert not HumanEvalBenchmark._run_test(code, test, "add")


class TestSudokuBenchmark:
    """Verification suite for Sudoku benchmark."""

    def test_load_generates_puzzles(self) -> None:
        """Sudoku generates puzzles when path is missing."""
        bm = SudokuExtremeBenchmark(data_path="/nonexistent/path")
        bm.load()
        assert len(bm.puzzles) > 0

    def test_is_valid_sudoku(self) -> None:
        """_is_valid_sudoku correctly validates boards."""
        valid = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]
        assert SudokuExtremeBenchmark._is_valid_sudoku(valid)

        invalid = [[1] * 9 for _ in range(9)]
        assert not SudokuExtremeBenchmark._is_valid_sudoku(invalid)

    def test_generate_puzzles_valid(self) -> None:
        """Generated puzzles are valid."""
        puzzles = SudokuExtremeBenchmark._generate_puzzles(3)
        assert len(puzzles) == 3
        for p in puzzles:
            assert len(p["puzzle"]) == 9
            assert len(p["solution"]) == 9
            assert p["solution"] is not None


class TestProntoQABenchmark:
    """Verification suite for ProntoQA benchmark."""

    def test_load_generates_dummy(self) -> None:
        """ProntoQA loads dummy data when path is missing."""
        bm = ProntoQABenchmark(data_path="/nonexistent/path")
        bm.load()
        assert len(bm.examples) > 0

    def test_evaluate_returns_benchmark_result(self) -> None:
        """ProntoQA evaluation returns a BenchmarkResult."""
        bm = ProntoQABenchmark(data_path="/nonexistent/path")
        bm.load()
        model = MockModel(response="Yes")
        result = bm.evaluate(model, max_examples=2)
        assert isinstance(result, BenchmarkResult)
        assert result.name == "prontoqa"

    def test_normalise_answer(self) -> None:
        """_normalise_answer extracts yes/no."""
        assert ProntoQABenchmark._normalise_answer("Yes, it is") == "yes"
        assert ProntoQABenchmark._normalise_answer("No way") == "no"
        assert ProntoQABenchmark._normalise_answer("True") == "yes"
        assert ProntoQABenchmark._normalise_answer("False") == "no"
