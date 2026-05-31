"""colab.py — Single entry point for TRM-Bank v3.0 on Google Colab (T4 GPU).

Run this script in a Colab cell:

    !python colab.py

It will:
  1. Install Python dependencies.
  2. Verify ESSM providers (SymPy, OR-Tools, PythonSandbox).
  3. Stream and filter datasets to ./data/processed/.
  4. Run two-stage training (QRandLoRA + Router/LPRM).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import List


def _run(cmd: List[str], desc: str) -> None:
    """Run a subprocess, stream output, abort on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\n[ERROR] {desc} failed (exit code {result.returncode}).")
        sys.exit(result.returncode)


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable

    t0 = time.time()

    # ── 1. Install dependencies ────────────────────────────────────
    _run(
        [py, "-m", "pip", "install", "-q",
         "torch", "transformers", "bitsandbytes", "accelerate",
         "datasets", "einops", "loguru", "tqdm", "numpy",
         "sympy", "ortools", "pydantic"],
        "Step 1/4 — Installing dependencies",
    )

    # ── 2. Verify modules ──────────────────────────────────────────
    _run([py, os.path.join(root, "verify_modules.py")],
         "Step 2/4 — Verifying ESSM providers")

    # ── 3. Prepare datasets ────────────────────────────────────────
    _run([py, os.path.join(root, "prepare_datasets.py")],
         "Step 3/4 — Preparing datasets (streaming)")

    # ── 4. Train ───────────────────────────────────────────────────
    _run([py, os.path.join(root, "colab_train.py")],
         "Step 4/4 — Training TRM-Bank v3.0")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  All done in {elapsed / 60:.1f} min.")
    print(f"  Checkpoints: {os.path.join(root, 'outputs_colab')}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
