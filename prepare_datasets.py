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
        try:
            text = formatter(row)
        except (KeyError, TypeError, AttributeError):
            text = ""
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


def _fmt_messages(row: Dict[str, Any]) -> str:
    """Parse a messages-format row: [{"role": ..., "content": ...}, ...].

    Also handles prompt+chosen fallback for HH-RLHF style datasets.
    """
    messages = row.get("messages", [])
    if messages:
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    # Fallback: prompt + chosen format (HH-RLHF style)
    prompt = row.get("prompt", "")
    chosen = row.get("chosen", "")
    if prompt and chosen:
        return f"user: {prompt}\nassistant: {chosen}"

    return ""


# ── Stage-2 formatters ────────────────────────────────────────────

def _fmt_stable_code(row: Dict[str, Any]) -> str:
    """Formatter for bunyaminergen/Stable-Code-Python-SFT."""
    instr = row.get("instruction", "")
    if isinstance(instr, list):
        instr = " ".join(
            m.get("content", str(m)) if isinstance(m, dict) else str(m)
            for m in instr
        )
    code = row.get("output", "")
    if not instr or not code:
        return ""
    return f"Instruction: {instr}\nCode: {code}"


def _fmt_flytech(row: Dict[str, Any]) -> str:
    """Formatter for flytech/python-codes-25k."""
    instr = row.get("instruction", "")
    code = row.get("output", "")
    if not instr or not code:
        return ""
    return f"Instruction: {instr}\nCode: {code}"


def _fmt_glaive(row: Dict[str, Any]) -> str:
    """Formatter for glaiveai/glaive-code-assistant-v3."""
    q = row.get("question", "")
    a = row.get("answer", "")
    if not q or not a:
        return ""
    return f"Question: {q}\nAnswer: {a}"


def _fmt_valley(row: Dict[str, Any]) -> str:
    """Formatter for collinear-ai/valley-of-reasoning-data."""
    prompt = row.get("instruction", row.get("input", row.get("prompt", "")))
    answer = row.get("output", row.get("response", row.get("target", "")))
    if isinstance(prompt, list):
        prompt = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt
        )
    if isinstance(answer, list):
        answer = "\n".join(
            f"{m.get('role', 'assistant')}: {m.get('content', '')}" for m in answer
        )
    if not prompt or not answer:
        return ""
    return f"Instruction: {prompt}\nReasoning/Answer: {answer}"



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

    # (b) no_robots — 1 300 instruction-following examples
    print("  [1b] HuggingFaceH4/no_robots  (target: 1 300)")
    ds_norobots = _stream_dataset("HuggingFaceH4/no_robots", split="train")
    s1_chat = _take_n(ds_norobots, tok, 1300, _fmt_messages,
                      label="no_robots")

    # (c) ultrachat_200k — 800 multi-turn dialogue examples
    print("  [1c] HuggingFaceH4/ultrachat_200k  (target: 800)")
    ds_ultra = _stream_dataset("HuggingFaceH4/ultrachat_200k",
                               split="train_sft")
    s1_chat += _take_n(ds_ultra, tok, 800, _fmt_messages,
                       label="ultrachat_200k")

    # (d) hh-rlhf — 500 helpful/harmless dialogue examples
    print("  [1d] trl-internal-testing/hh-rlhf-trl-style  (target: 500)")
    ds_hh = _stream_dataset("trl-internal-testing/hh-rlhf-trl-style",
                            split="train")
    s1_chat += _take_n(ds_hh, tok, 500, _fmt_messages,
                       label="hh-rlhf")

    s1_all = s1_math + s1_chat
    random.shuffle(s1_all)
    print(f"  Stage 1 breakdown: {len(s1_math)} math + {len(s1_chat)} chat = {len(s1_all)} total")
    _shuffle_and_save(s1_all, os.path.join(DATA_DIR,
                       "stage1_qrandlora.jsonl"), "Stage 1")
    print(f"  Stage 1 done in {time.time() - t_stage:.0f}s")

    # ── Stage 2 ────────────────────────────────────────────────────
    t_stage = time.time()
    print("\n--- Stage 2: Router + LPRM Tuning (3 000 examples) ---")
    _flush()

    s2_all: List[Dict[str, str]] = []

    # (a) Stable-Code-Python-SFT — 400 code generation examples
    print("  [2a] bunyaminergen/Stable-Code-Python-SFT  (target: 400)")
    ds_stable = _stream_dataset("bunyaminergen/Stable-Code-Python-SFT")
    s2_all += _take_n(ds_stable, tok, 400, _fmt_stable_code,
                      label="Stable-Code-Python-SFT")

    # (a2) flytech/python-codes-25k — 200 code generation examples
    print("  [2a2] flytech/python-codes-25k  (target: 200)")
    ds_flytech = _stream_dataset("flytech/python-codes-25k")
    s2_all += _take_n(ds_flytech, tok, 200, _fmt_flytech,
                      label="python-codes-25k")

    # (a3) glaive-code-assistant-v3 — 150 code Q&A examples
    print("  [2a3] glaiveai/glaive-code-assistant-v3  (target: 150)")
    ds_glaive = _stream_dataset("glaiveai/glaive-code-assistant-v3")
    s2_all += _take_n(ds_glaive, tok, 150, _fmt_glaive,
                      label="glaive-code-assistant-v3")

    # (b) Valley-of-Reasoning — 750 logical reasoning examples
    print("  [2b] collinear-ai/valley-of-reasoning-data  (target: 750)")
    ds_valley = _stream_dataset("collinear-ai/valley-of-reasoning-data")
    s2_all += _take_n(ds_valley, tok, STAGE2_PER_SRC, _fmt_valley,
                      label="Valley-of-Reasoning")

    # (c) BBH — 750 hard reasoning
    print("  [2c] maveriq/bigbenchhard  (target: 750)")
    ds_bbh = _stream_dataset("maveriq/bigbenchhard")
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
