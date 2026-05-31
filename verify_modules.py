"""verify_modules.py — Validate ESSM provider dependencies and sandbox execution.

Usage:
    python verify_modules.py

Tests:
    - SymPyProvider: symbolic integration
    - ORToolsProvider: CP-SAT model creation
    - PythonSandboxProvider: restricted factorial computation
"""

from __future__ import annotations

import sys
from typing import Any


def test_sympy() -> str:
    """Test SymPy symbolic integration: integral of sin(x)*exp(x)."""
    import sympy as sp
    x = sp.Symbol("x")
    result = sp.integrate(sp.sin(x) * sp.exp(x), x)
    expected = sp.exp(x) * sp.sin(x) / 2 - sp.exp(x) * sp.cos(x) / 2
    assert sp.simplify(result - expected) == 0, f"Unexpected result: {result}"
    return f"[OK] SymPy: integral of sin(x)*exp(x) = {result}"


def test_ortools() -> str:
    """Test OR-Tools CP-SAT model creation and solve."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    x = model.NewIntVar(0, 10, "x")
    y = model.NewIntVar(0, 10, "y")
    model.Add(x + y == 7)
    # Maximise the sum (linear objective, safe across all OR-Tools versions).
    model.Maximize(x + y)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status == cp_model.OPTIMAL, f"Solver did not find optimal: status={status}"
    assert solver.Value(x) + solver.Value(y) == 7, (
        f"Constraint x+y=7 violated: x={solver.Value(x)}, y={solver.Value(y)}"
    )
    return f"[OK] OR-Tools: x+y=7 solved -> x={solver.Value(x)}, y={solver.Value(y)}"


def test_python_sandbox() -> str:
    """Test PythonSandboxProvider with factorial computation."""
    from router.essm import PythonSandboxProvider
    provider = PythonSandboxProvider(time_limit=2.0)
    code = "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)\nresult = factorial(10)"
    output: Any = provider.execute(code, context={})
    assert output == 3628800, f"Factorial(10) expected 3628800, got {output}"
    return f"[OK] PythonSandbox: factorial(10) = {output}"


def main() -> None:
    print("=" * 60)
    print("TRM-Bank v3.0 — ESSM Provider Validation")
    print("=" * 60)

    tests = [
        ("SymPy Integration", test_sympy),
        ("OR-Tools CP-SAT", test_ortools),
        ("Python Sandbox", test_python_sandbox),
    ]

    exit_code = 0
    for name, fn in tests:
        try:
            msg = fn()
            print(f"  {msg}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            exit_code = 1

    print("-" * 60)
    if exit_code == 0:
        print("  All modules validated successfully.")
    else:
        print("  Some modules failed. Check errors above.")
    print("=" * 60)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
