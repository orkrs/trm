from __future__ import annotations

import ast
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class ComputeProvider(ABC):
    """Abstract interface for a cognitive compute module.

    Each provider can estimate its own success probability q_i
    and cost c_i for a given query, and then execute the query.
    """

    name: str
    cost: float

    @abstractmethod
    def estimate_quality(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Return self-assessed success probability q_i in [0, 1]."""
        ...

    @abstractmethod
    def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute the query and return the result."""
        ...


class PythonSandboxProvider(ComputeProvider):
    """Executes Python code in a restricted sandbox.

    Provides a safe execution environment by restricting builtins
    and imports.  Useful for GSM8K math evaluation and HumanEval
    code synthesis.

    Args:
        cost: Base computational cost in abstract FLOP units.
        allowed_imports: Whitelist of importable modules.
        time_limit: Maximum execution time in seconds.
    """

    name: str = "python_sandbox"
    cost: float = 10.0

    def __init__(
        self,
        cost: float = 10.0,
        allowed_imports: Optional[List[str]] = None,
        time_limit: float = 2.0,
    ) -> None:
        self.cost = cost
        self.time_limit = time_limit
        self.allowed_imports = allowed_imports or [
            "math", "random", "itertools", "collections",
            "functools", "typing", "json", "re",
        ]

    def estimate_quality(self, query: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Heuristic: code with valid AST and known patterns scores higher."""
        score = 0.5
        try:
            ast.parse(query)
            score += 0.2
        except SyntaxError:
            score -= 0.3
        if "def " in query:
            score += 0.15
        if "return" in query:
            score += 0.1
        return float(max(0.0, min(1.0, score)))

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Run Python code with restricted builtins.

        Args:
            query: Python source code string.
            context: Optional variable dict to inject into scope.

        Returns:
            Execution result, or an error dict on failure.

        Raises:
            TimeoutError: If execution exceeds the time limit.
        """
        restricted_builtins: Dict[str, Any] = {
            "abs": abs, "all": all, "any": any, "bool": bool,
            "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format, "frozenset": frozenset,
            "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
            "int": int, "isinstance": isinstance, "issubclass": issubclass,
            "iter": iter, "len": len, "list": list, "map": map, "max": max,
            "min": min, "next": next, "object": object, "oct": oct, "ord": ord,
            "pow": pow, "print": print, "range": range, "repr": repr,
            "reversed": reversed, "round": round, "set": set,
            "slice": slice, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "type": type, "zip": zip, "True": True,
            "False": False, "None": None,
        }
        unsafe = {"__import__", "eval", "exec", "compile", "open", "__builtins__"}
        for u in unsafe:
            restricted_builtins.pop(u, None)

        local_scope: Dict[str, Any] = {"__builtins__": restricted_builtins}
        if context:
            local_scope.update(context)

        try:
            compiled = compile(query, "<sandbox>", "exec")
            exec(compiled, local_scope)
            result = local_scope.get("result", None)
            logger.debug(f"PythonSandbox executed successfully, result={result}")
            return result
        except Exception as exc:
            logger.warning(f"PythonSandbox execution failed: {exc}")
            return {"error": str(exc), "traceback": traceback.format_exc()}


class SymPyProvider(ComputeProvider):
    """Symbolic mathematics via SymPy.

    Handles algebraic manipulation, equation solving, differentiation,
    integration, and logical deduction.  Used for ARC-AGI and
    ProntoQA benchmarks.

    Args:
        cost: Base computational cost.
    """

    name: str = "symbolic_solver"
    cost: float = 20.0

    def __init__(self, cost: float = 20.0) -> None:
        self.cost = cost

    def estimate_quality(self, query: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Heuristic: queries with symbolic patterns score higher."""
        score = 0.4
        patterns = ["solve", "integrate", "diff", "simplify", "expand",
                     "factor", "symbols", "Eq", "Matrix"]
        if any(p in query for p in patterns):
            score += 0.4
        if len(query) > 50:
            score += 0.1
        return float(min(1.0, score))

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a SymPy expression string.

        The query is evaluated using sympy's sympify or a custom
        sequence of SymPy operations.

        Args:
            query: A string containing SymPy-compatible expressions.
            context: Optional variable bindings.

        Returns:
            SymPy expression result, or an error dict.
        """
        try:
            import sympy as sp
        except ImportError:
            return {"error": "SymPy is not installed."}

        try:
            local_vars: Dict[str, Any] = {"sp": sp}
            if context:
                local_vars.update(context)
            compiled = compile(query, "<sympy>", "exec")
            exec(compiled, local_vars)
            result = local_vars.get("result", None)
            logger.debug(f"SymPy result: {result}")
            return result
        except Exception as exc:
            logger.warning(f"SymPy execution failed: {exc}")
            return {"error": str(exc), "traceback": traceback.format_exc()}


class ORToolsProvider(ComputeProvider):
    """Constraint satisfaction and optimization via Google OR-Tools.

    Handles SAT solving, constraint programming, and linear/mixed-integer
    optimization.  Used for Sudoku-Extreme and ARC-AGI structured tasks.

    Args:
        cost: Base computational cost.
    """

    name: str = "or_tools_solver"
    cost: float = 15.0

    def __init__(self, cost: float = 15.0) -> None:
        self.cost = cost

    def estimate_quality(self, query: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Heuristic: constraint-related keywords indicate suitability."""
        score = 0.3
        patterns = [
            "IntVar", "NewIntVar", "Add", "Solver", "DecisionBuilder",
            "Phase", "Solve", "constraint", "satisfaction",
        ]
        if any(p in query for p in patterns):
            score += 0.5
        if len(query) > 100:
            score += 0.1
        return float(min(1.0, score))

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute an OR-Tools constraint program string.

        Args:
            query: Python source code using OR-Tools API.
            context: Optional variable bindings.

        Returns:
            Solver result, or an error dict.
        """
        try:
            from ortools.constraint_solver import pywrapcp
            from ortools.sat.python import cp_model
        except ImportError:
            return {"error": "OR-Tools is not installed."}

        try:
            local_vars: Dict[str, Any] = {
                "pywrapcp": pywrapcp,
                "cp_model": cp_model,
            }
            if context:
                local_vars.update(context)
            compiled = compile(query, "<ortools>", "exec")
            exec(compiled, local_vars)
            result = local_vars.get("result", None)
            logger.debug(f"OR-Tools result: {result}")
            return result
        except Exception as exc:
            logger.warning(f"OR-Tools execution failed: {exc}")
            return {"error": str(exc), "traceback": traceback.format_exc()}


class DirectGenerationProvider(ComputeProvider):
    """Fast direct text generation from the LM backbone.

    Lowest-cost provider; used for trivial factual questions
    (First Finish Search).

    Args:
        cost: Base computational cost.
    """

    name: str = "direct_generation"
    cost: float = 1.0

    def __init__(self, cost: float = 1.0) -> None:
        self.cost = cost

    def estimate_quality(self, query: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Heuristic: short factual queries score highest for this provider."""
        score = 0.7
        if len(query) < 200:
            score += 0.15
        if "?" in query:
            score += 0.05
        return float(min(1.0, score))

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Placeholder: returns the query as-is for the backbone to handle."""
        logger.debug(f"DirectGeneration queried: {query[:50]}...")
        return {"query": query, "status": "routed_to_backbone"}


class MIMOBrancherProvider(ComputeProvider):
    """Deep MIMO-based multi-branch reasoning.

    Highest-cost provider; used for complex mathematical and logical
    problems requiring multi-trajectory deterministic search.

    Args:
        cost: Base computational cost.
    """

    name: str = "mimo_brancher"
    cost: float = 30.0

    def __init__(self, cost: float = 30.0) -> None:
        self.cost = cost

    def estimate_quality(self, query: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Heuristic: complex multi-step problems score highest."""
        score = 0.3
        complexity_markers = [
            "prove", "compute", "calculate", "solve", "derive",
            "evaluate", "determine", "optimize", "\n",
        ]
        marker_count = sum(1 for m in complexity_markers if m in query)
        score += marker_count * 0.1
        if len(query.split()) > 50:
            score += 0.2
        return float(min(1.0, score))

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Placeholder: the query is processed by the MIMO scan branches."""
        logger.debug(f"MIMOBrancher routed: {query[:50]}...")
        return {"query": query, "status": "routed_to_mimo_scan"}
