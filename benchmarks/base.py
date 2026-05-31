"""Base class and protocol for all TRM-Bank benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkResult:
    """Standardised result container for a single benchmark.

    Attributes:
        name: Benchmark name (e.g., "arc_agi", "gsm8k").
        accuracy: Primary accuracy / pass rate in [0, 1].
        num_correct: Number of correct answers.
        num_total: Total number of test examples.
        details: Per-example results for analysis.
        extra: Additional benchmark-specific metrics.
    """
    name: str
    accuracy: float
    num_correct: int
    num_total: int
    details: Optional[List[Dict[str, Any]]] = None
    extra: Optional[Dict[str, Any]] = None


class Benchmark(ABC):
    """Abstract base class for TRM-Bank benchmarks.

    Each benchmark subclass implements load() and evaluate().
    The evaluate() method returns a BenchmarkResult with standardised
    metrics that the orchestrator can log and compare.
    """

    def __init__(self, name: str, data_path: str) -> None:
        self.name: str = name
        self.data_path: str = data_path

    @abstractmethod
    def load(self) -> None:
        """Load the benchmark dataset from disk."""
        ...

    @abstractmethod
    def evaluate(self, model: Any, **kwargs: Any) -> BenchmarkResult:
        """Run evaluation and return standardised results."""
        ...
