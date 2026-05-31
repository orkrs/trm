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
    return tok


def _tok_len(text: str, tok: Any) -> int:
    return len(tok.encode(text, truncation=False))


# ── generic streaming helpers ──────────────────────────────────────

def _stream_dataset(name: str, split: str = "train",
                    streaming: bool = True) -> Any:
    """Stream a HuggingFace dataset."""
    from datasets import load_dataset
    return load_dataset(name, split=split, streaming=streaming)


def _take_n(stream: Any, tok: Any, n: int,
            formatter: Any, label: str = "") -> List[Dict[str, str]]:
    """Take exactly *n* examples from a stream that pass the token limit."""
    out: List[Dict[str, str]] = []
    scanned = 0
    for row in stream:
        scanned += 1
        text = formatter(row)
        if text and _tok_len(text, tok) <= MAX_TOKENS:
            out.append({"text": text})
            if len(out) % 100 == 0:
                print(f"       Собрано {len(out)} / {n}  (просканировано {scanned} строк) ...")
        if len(out) >= n:
            break
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


def main() -> None:
    print("=" * 60)
    print("TRM-Bank v3.0 — Dataset Preparation (streaming)")
    print("=" * 60)

    tok = _load_tokenizer()

    # ── Stage 1 ────────────────────────────────────────────────────
    print("\n--- Stage 1: QRandLoRA Pretraining (6 500 examples) ---")

    s1_math: List[Dict[str, str]] = []

    # (a) NuminaMath-CoT — 3 900 math CoT examples
    print("  [1a] Streaming AI-MO/NuminaMath-CoT ...")
    ds_numina = _stream_dataset("AI-MO/NuminaMath-CoT")
    s1_math += _take_n(ds_numina, tok, STAGE1_MATH, _fmt_numina)
    print(f"       Got {len(s1_math)} / {STAGE1_MATH} math examples")

    # (b) Magpie-Ultra — 1 300 dialogue examples
    print("  [1b] Streaming argilla/magpie-ultra-v1.0 ...")
    ds_magpie = _stream_dataset("argilla/magpie-ultra-v1.0")
    s1_chat: List[Dict[str, str]] = []
    s1_chat += _take_n(ds_magpie, tok, STAGE1_CHAT // 2, _fmt_magpie)
    print(f"       Got {len(s1_chat)} from Magpie")

    # (c) ShareGPT-Cleaned — 1 300 dialogue examples
    print("  [1c] Streaming Vtuber-plan/sharegpt-cleaned ...")
    ds_sg = _stream_dataset("Vtuber-plan/sharegpt-cleaned")
    s1_chat += _take_n(ds_sg, tok, STAGE1_CHAT - len(s1_chat), _fmt_sharegpt)
    print(f"       Got {len(s1_chat)} / {STAGE1_CHAT} chat examples")

    s1_all = s1_math + s1_chat
    random.shuffle(s1_all)
    _shuffle_and_save(s1_all, os.path.join(DATA_DIR,
                       "stage1_qrandlora.jsonl"), "Stage 1")

    # ── Stage 2 ────────────────────────────────────────────────────
    print("\n--- Stage 2: Router + LPRM Tuning (3 000 examples) ---")

    s2_all: List[Dict[str, str]] = []

    # (a) LiveCodeBench — 750 code generation
    print("  [2a] Streaming livecodebench/code_generation_lite ...")
    ds_lcb = _stream_dataset("livecodebench/code_generation_lite")
    s2_all += _take_n(ds_lcb, tok, STAGE2_PER_SRC, _fmt_livecodebench)
    print(f"       Got {len(s2_all)} / {STAGE2_PER_SRC}")

    # (b) MuSR — 750 multi-step reasoning
    print("  [2b] Streaming TAUR-Lab/MuSR ...")
    ds_musr = _stream_dataset("TAUR-Lab/MuSR")
    before = len(s2_all)
    s2_all += _take_n(ds_musr, tok, STAGE2_PER_SRC,
                      _fmt_musr)
    print(f"       Got {len(s2_all) - before} / {STAGE2_PER_SRC}")

    # (c) BBH — 750 hard reasoning
    print("  [2c] Streaming lukaemon/bbh ...")
    ds_bbh = _stream_dataset("lukaemon/bbh")
    before = len(s2_all)
    s2_all += _take_n(ds_bbh, tok, STAGE2_PER_SRC, _fmt_bbh)
    print(f"       Got {len(s2_all) - before} / {STAGE2_PER_SRC}")

    # (d) MMLU-Pro — 750 factual
    print("  [2d] Streaming TIGER-Lab/MMLU-Pro ...")
    ds_mmlu = _stream_dataset("TIGER-Lab/MMLU-Pro", split="test")
    before = len(s2_all)
    s2_all += _take_n(ds_mmlu, tok, STAGE2_PER_SRC, _fmt_mmlupro)
    print(f"       Got {len(s2_all) - before} / {STAGE2_PER_SRC}")

    random.shuffle(s2_all)
    _shuffle_and_save(s2_all, os.path.join(DATA_DIR,
                       "stage2_router.jsonl"), "Stage 2")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Stage 1: {STAGE1_TOTAL} examples -> "
          f"{os.path.join(DATA_DIR, 'stage1_qrandlora.jsonl')}")
    print(f"  Stage 2: {STAGE2_TOTAL} examples -> "
          f"{os.path.join(DATA_DIR, 'stage2_router.jsonl')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
