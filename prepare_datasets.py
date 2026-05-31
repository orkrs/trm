"""prepare_datasets.py — Stream, filter, and save datasets for TRM-Bank v3.0.

Usage:
    python prepare_datasets.py

Requirements:
    pip install datasets transformers tqdm

All data is streamed (streaming=True) — nothing is fully downloaded
to disk.  Only examples with <= 640 tokens (Mamba-2.8B tokenizer)
are kept.

Output:
    ./data/processed/stage1_qrandlora.jsonl   (6 500 examples)
    ./data/processed/stage2_router.jsonl      (3 000 examples)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Dict, List

# ── constants ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)

MAX_TOKENS = 640
STAGE1_TOTAL = 6_500
STAGE1_MATH  = 3_900
STAGE1_CHAT  = 2_600
STAGE2_TOTAL = 3_000
STAGE2_PER_SRC = 750


def _flush() -> None:
    """Force flush stdout so Colab sees output immediately."""
    sys.stdout.flush()


# ── tokenizer ──────────────────────────────────────────────────────

def _load_tokenizer() -> Any:
    """Load the Mamba-2.8B tokenizer, fallback to GPT-2."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b")
        print(f"  Tokenizer loaded: state-spaces/mamba-2.8b  (vocab {tok.vocab_size})")
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2")
        print("  [WARN] Fallback to GPT-2 tokenizer")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _flush()
    return tok


def _tok_len(text: str, tok: Any) -> int:
    return len(tok.encode(text, truncation=False))


# ── generic streaming helpers ──────────────────────────────────────

def _stream_dataset(name: str, split: str = "train",
                    streaming: bool = True) -> Any:
    """Stream a HuggingFace dataset."""
    from datasets import load_dataset
    print(f"    [LOAD] Connecting to HuggingFace Hub ...", end=" ")
    _flush()
    ds = load_dataset(name, split=split, streaming=streaming)
    print("done.")
    _flush()
    return ds


def _take_n(stream: Any, tok: Any, n: int,
            formatter: Any, label: str = "") -> List[Dict[str, str]]:
    """Take exactly *n* examples from a stream that pass the token limit."""
    out: List[Dict[str, str]] = []
    scanned = 0
    kept = 0
    t0 = time.time()
    print(f"    [SCAN] Starting iteration, looking for {n} examples (max {MAX_TOKENS} tokens) ...")
    _flush()

    for row in stream:
        scanned += 1
        text = formatter(row)
        if text and _tok_len(text, tok) <= MAX_TOKENS:
            out.append({"text": text})
            kept += 1

        # Progress every 100 collected or every 500 scanned
        if kept > 0 and kept % 100 == 0:
            elapsed = time.time() - t0
            speed = scanned / elapsed if elapsed > 0 else 0
            print(f"    [SCAN] Collected {kept}/{n}  |  scanned {scanned} rows  |  "
                  f"keep ratio {kept/scanned:.1%}  |  {speed:.0f} rows/s  |  "
                  f"{elapsed:.0f}s elapsed")
            _flush()

        if len(out) >= n:
            break

    elapsed = time.time() - t0
    print(f"    [DONE] {label}: {len(out)} kept / {scanned} scanned  "
          f"({elapsed:.1f}s)")
    _flush()
    return out


# ── Stage-1 formatters ────────────────────────────────────────────

def _fmt_numina(row: Dict[str, Any]) -> str:
    problem = row.get("problem", "")
    solution = row.get("solution", "")
    if not problem or not solution:
        return ""
    return f"### Problem\n{problem}\n### Solution\n{solution}"


def _fmt_deepseek_r1_math(row: Dict[str, Any]) -> str:
    problem = row.get("problem", row.get("question", ""))
    solution = row.get("solution", row.get("response",
                row.get("output", "")))
    if not problem or not solution:
        return ""
    return f"### Problem\n{problem}\n### Solution\n{solution}"


def _fmt_magpie(row: Dict[str, Any]) -> str:
    q = row.get("question", row.get("prompt", row.get("input", "")))
    a = row.get("response", row.get("output", row.get("answer", "")))
    if not q or not a:
        convs = row.get("conversations", row.get("messages", []))
        if convs:
            parts = []
            for turn in convs:
                role = turn.get("from", turn.get("role", "user"))
                val  = turn.get("value", turn.get("content", ""))
                parts.append(f"### {role}\n{val}")
            return "\n\n".join(parts)
        return ""
    return f"### Human\n{q}\n### Assistant\n{a}"


def _fmt_sharegpt(row: Dict[str, Any]) -> str:
    convs = row.get("conversations", row.get("messages", []))
    if not convs:
        return ""
    parts = []
    for turn in convs:
        role = turn.get("from", turn.get("role", "user"))
        val  = turn.get("value", turn.get("content", ""))
        parts.append(f"### {role}\n{val}")
    return "\n\n".join(parts)


# ── Stage-2 formatters ────────────────────────────────────────────

def _fmt_livecodebench(row: Dict[str, Any]) -> str:
    q = row.get("question", row.get("prompt", row.get("problem", "")))
    c = row.get("code", row.get("solution", row.get("output", "")))
    if not q or not c:
        return ""
    return f"### Problem\n{q}\n### Code\n{c}"


def _fmt_musr(row: Dict[str, Any]) -> str:
    q = row.get("question", row.get("input", row.get("prompt", "")))
    a = row.get("answer", row.get("output", row.get("response", "")))
    if not q or not a:
        return ""
    return f"### Question\n{q}\n### Answer\n{a}"


def _fmt_bbh(row: Dict[str, Any]) -> str:
    q = row.get("input", row.get("question", row.get("prompt", "")))
    a = row.get("target", row.get("answer", row.get("output", "")))
    if not q or not a:
        return ""
    return f"### Question\n{q}\n### Answer\n{a}"


def _fmt_mmlupro(row: Dict[str, Any]) -> str:
    q   = row.get("question", row.get("input", ""))
    opts = row.get("options", row.get("choices", []))
    ans = row.get("answer", row.get("answer_idx", ""))
    if not q:
        return ""
    opt_str = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
    return f"### Question\n{q}\n{opt_str}\n### Answer\n{ans}"


# ── main pipeline ──────────────────────────────────────────────────

def _shuffle_and_save(data: List[Dict[str, str]], path: str,
                      label: str) -> None:
    """Shuffle *data* and write to *path* as JSONL."""
    random.shuffle(data)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {label}: {len(data)} examples -> {path}")
    _flush()


def main() -> None:
    t_total = time.time()
    print("=" * 60)
    print("TRM-Bank v3.0 — Dataset Preparation (streaming)")
    print("=" * 60)
    _flush()

    tok = _load_tokenizer()

    # ── Stage 1 ────────────────────────────────────────────────────
    t_stage = time.time()
    print("\n--- Stage 1: QRandLoRA Pretraining (6 500 examples) ---")
    _flush()

    # (a) NuminaMath-CoT — 3 900 math CoT examples
    print("  [1a] AI-MO/NuminaMath-CoT  (target: 3 900)")
    ds_numina = _stream_dataset("AI-MO/NuminaMath-CoT")
    s1_math = _take_n(ds_numina, tok, STAGE1_MATH, _fmt_numina,
                      label="NuminaMath-CoT")

    # (b) Magpie-Ultra — 1 300 dialogue examples
    print("  [1b] argilla/magpie-ultra-v1.0  (target: 1 300)")
    ds_magpie = _stream_dataset("argilla/magpie-ultra-v1.0")
    s1_chat = _take_n(ds_magpie, tok, STAGE1_CHAT // 2, _fmt_magpie,
                      label="Magpie-Ultra")

    # (c) ShareGPT-Cleaned — 1 300 dialogue examples
    print("  [1c] Vtuber-plan/sharegpt-cleaned  (target: 1 300)")
    ds_sg = _stream_dataset("Vtuber-plan/sharegpt-cleaned")
    s1_chat += _take_n(ds_sg, tok, STAGE1_CHAT - len(s1_chat),
                       _fmt_sharegpt, label="ShareGPT-Cleaned")

    s1_all = s1_math + s1_chat
    random.shuffle(s1_all)
    _shuffle_and_save(s1_all, os.path.join(DATA_DIR,
                       "stage1_qrandlora.jsonl"), "Stage 1")
    print(f"  Stage 1 done in {time.time() - t_stage:.0f}s")

    # ── Stage 2 ────────────────────────────────────────────────────
    t_stage = time.time()
    print("\n--- Stage 2: Router + LPRM Tuning (3 000 examples) ---")
    _flush()

    s2_all: List[Dict[str, str]] = []

    # (a) LiveCodeBench — 750 code generation
    print("  [2a] livecodebench/code_generation_lite  (target: 750)")
    ds_lcb = _stream_dataset("livecodebench/code_generation_lite")
    s2_all += _take_n(ds_lcb, tok, STAGE2_PER_SRC, _fmt_livecodebench,
                      label="LiveCodeBench")

    # (b) MuSR — 750 multi-step reasoning
    print("  [2b] TAUR-Lab/MuSR  (target: 750)")
    ds_musr = _stream_dataset("TAUR-Lab/MuSR")
    s2_all += _take_n(ds_musr, tok, STAGE2_PER_SRC, _fmt_musr,
                      label="MuSR")

    # (c) BBH — 750 hard reasoning
    print("  [2c] lukaemon/bbh  (target: 750)")
    ds_bbh = _stream_dataset("lukaemon/bbh")
    s2_all += _take_n(ds_bbh, tok, STAGE2_PER_SRC, _fmt_bbh,
                      label="BBH")

    # (d) MMLU-Pro — 750 factual
    print("  [2d] TIGER-Lab/MMLU-Pro  (target: 750)")
    ds_mmlu = _stream_dataset("TIGER-Lab/MMLU-Pro", split="test")
    s2_all += _take_n(ds_mmlu, tok, STAGE2_PER_SRC, _fmt_mmlupro,
                      label="MMLU-Pro")

    random.shuffle(s2_all)
    _shuffle_and_save(s2_all, os.path.join(DATA_DIR,
                       "stage2_router.jsonl"), "Stage 2")
    print(f"  Stage 2 done in {time.time() - t_stage:.0f}s")

    # ── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Stage 1: {len(s1_all)} examples -> "
          f"{os.path.join(DATA_DIR, 'stage1_qrandlora.jsonl')}")
    print(f"  Stage 2: {len(s2_all)} examples -> "
          f"{os.path.join(DATA_DIR, 'stage2_router.jsonl')}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)
    _flush()


if __name__ == "__main__":
    main()
