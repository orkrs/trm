"""colab_train.py — TRM-Bank v3.0 training entry point for Google Colab (T4, 16 GB).

Usage:
    python prepare_datasets.py   # 1. prepare data
    python verify_modules.py     # 2. verify providers
    python colab_train.py        # 3. train

The script downloads Mamba-1.4B in 4-bit NF4, patches all SSM mixers
with ComplexMIMOMamba3, applies QRandLoRA (r=64), attaches the LPRM
head and game-theoretic router, and runs truncated-BPTT training.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger


# ------------------------------------------------------------------
# Colab T4 optimised config
# ------------------------------------------------------------------

STAGE1_OVERRIDES: Dict[str, Any] = {
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "truncation_length": 128,
    "learning_rate": 3e-4,
    "warmup_steps": 30,
    "max_steps": 500,
    "num_epochs": 999,
    "log_every_n_steps": 10,
    "eval_every_n_steps": 250,
    "save_every_n_steps": 250,
    "output_dir": "./outputs_colab",
    "max_seq_length": 512,
    "lprm_weight": 0.1,
    "gradient_checkpointing": True,
}

STAGE2_OVERRIDES: Dict[str, Any] = {
    **STAGE1_OVERRIDES,
    "learning_rate": 1e-4,
    "max_steps": 300,
    "warmup_steps": 20,
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")


def check_gpu() -> torch.device:
    """Exit with a clear message if no GPU is available."""
    if not torch.cuda.is_available():
        print("[ERROR] Colab requires GPU (CUDA). No GPU detected.")
        print("  Go to Runtime -> Change runtime type -> T4 GPU.")
        sys.exit(1)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    free_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[OK] GPU: {gpu_name} ({free_mem:.1f} GB)")
    return device


def _load_tokenizer() -> Any:
    """Load tokenizer. Prefer GPT-2 (matches Mamba-1.4B vocab)."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
        print(f"  Tokenizer loaded: gpt2  (vocab {tok.vocab_size})")
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2")
        print("  [WARN] Using default GPT-2 tokenizer")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _reconstruct_text(item: Dict[str, Any]) -> str:
    """Reconstruct a text field from a JSONL dict that may lack 'text'.

    Tries common column name patterns used by HuggingFace SFT datasets.
    Returns empty string if nothing usable is found.
    """
    # 1) Already has 'text'
    t = item.get("text", "")
    if t:
        return t

    # 2) messages-style: list of {role, content}
    messages = item.get("messages", item.get("conversations", []))
    if isinstance(messages, list) and messages:
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", m.get("from", ""))
                content = m.get("content", m.get("value", ""))
                if role and content:
                    parts.append(f"{role}: {content}")
        if parts:
            return "\n".join(parts)

    # 3) prompt/instruction + output/response/answer
    prompt = (item.get("prompt", "")
              or item.get("instruction", "")
              or item.get("question", "")
              or item.get("input", "")
              or item.get("problem", ""))
    answer = (item.get("output", "")
              or item.get("response", "")
              or item.get("answer", "")
              or item.get("solution", "")
              or item.get("target", "")
              or item.get("code", ""))
    if prompt and answer:
        return f"{prompt}\n{answer}"
    if prompt:
        return prompt

    return ""


def load_tokenized_sequences(jsonl_path: str,
                             max_seq_length: int = 1024) -> List[torch.Tensor]:
    """Load a .jsonl file, tokenize, split into chunks.

    Automatically reconstructs 'text' from available fields if missing.

    Args:
        jsonl_path: Path to the .jsonl file.
        max_seq_length: Maximum tokens per sequence chunk.

    Returns:
        List of 1-D token-id tensors.
    """
    tokenizer = _load_tokenizer()

    sequences: List[torch.Tensor] = []
    fixed = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = _reconstruct_text(item)
            if not text:
                continue
            ids = tokenizer.encode(text, truncation=False)
            if len(ids) < 2:
                continue
            for start in range(0, len(ids), max_seq_length):
                chunk = ids[start : start + max_seq_length]
                if len(chunk) >= 2:
                    sequences.append(torch.tensor(chunk, dtype=torch.long))

    print(f"  Loaded {len(sequences)} sequences from {jsonl_path}")
    return sequences


def build_trainer_for_stage(
    stage_name: str,
    jsonl_path: str,
    device: torch.device,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build model, LPRM, memory, router, and trainer for one stage."""
    from config import CONFIG
    from core.complex_mimo_mamba import TRMBankModel
    from training.trainer import TruncatedBPTTDataset, TRMBankTrainer

    cfg = CONFIG.training
    if config_overrides:
        from dataclasses import replace
        cfg = replace(cfg, **config_overrides)

    print(f"\n{'=' * 60}")
    print(f"Stage: {stage_name}")
    print(f"{'=' * 60}")

    # ---- 1. Load tokenized sequences ----
    print("\n[1/5] Loading dataset...")
    sequences = load_tokenized_sequences(jsonl_path,
                                         max_seq_length=cfg.truncation_length)
    if not sequences:
        raise RuntimeError(f"No sequences loaded from {jsonl_path}")
    train_dataset = TruncatedBPTTDataset(
        data=sequences,
        truncation_length=cfg.truncation_length,
        seq_length=cfg.max_seq_length,
    )
    print(f"  Dataset chunks: {len(train_dataset)}")

    # ---- 2. Build TRMBankModel ----
    print("\n[2/5] Building TRMBankModel (Mamba-1.4B 4-bit)...")
    model = TRMBankModel(
        pretrained_name="state-spaces/mamba-1.4b-hf",
        mimo_rank=CONFIG.mimo.mimo_rank,
        d_state=CONFIG.model.d_state,
        use_4bit=True,
        device=device,
        qrandlora_r=64,
        qrandlora_alpha=CONFIG.qrandlora.scaling_init,
        qrandlora_sparsity=CONFIG.qrandlora.sparsity,
        qrandlora_num_components=CONFIG.qrandlora.num_components,
        qrandlora_target_modules=CONFIG.qrandlora.target_modules,
    )
    model.build()
    # 4-bit model is already on GPU via device_map="auto".
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Hidden dim: {model.hidden_dim}, Layers: {model.num_layers}")
    print(f"  QRandLoRA patched: {model._qr_patched_count} layers")

    # ---- 3. MultiHeadLPRM ----
    print("\n[3/5] Initialising MultiHeadLPRM...")
    from router.lprm import MultiHeadLPRM

    lprm = MultiHeadLPRM(
        module_names=list(CONFIG.router.module_names),
        hidden_dim=CONFIG.router.lprm.hidden_dim,
        num_heads=model.hidden_dim // 8,
        dropout=CONFIG.router.lprm.dropout,
    )
    lprm = lprm.to(device)
    lprm.train()
    print(f"  LPRM heads: {len(lprm.module_names)}, hidden: {lprm.hidden_dim}")

    # ---- 4. Pipeline (memory + router) ----
    print("\n[4/5] Setting up memory and router...")
    from memory.hierarchical_memory import HierarchicalMemory
    from router.game_theoretic_router import GameTheoreticRouter

    memory = HierarchicalMemory(
        tier_names=CONFIG.memory.tier_names,
        tier_slots=CONFIG.memory.memory_slots_per_tier,
        embedding_dim=CONFIG.memory.embedding_dim,
        faiss_index_type=CONFIG.memory.faiss_index_type,
    )
    router = GameTheoreticRouter(
        lprm=lprm,
        flops_budget=CONFIG.router.flops_budget,
    )
    model.set_pipeline(
        memory=memory,
        router=router,
        lprm=lprm,
        memory_top_k=CONFIG.memory.top_k_retrieval,
        router_interval=0,
    )
    print("  Pipeline attached: memory + router + LPRM")

    # ---- 5. Trainer ----
    print(f"\n[5/5] Initialising TRMBankTrainer...")
    trainer = TRMBankTrainer(
        model=model,
        lprm=lprm,
        config=cfg,
        train_dataset=train_dataset,
        memory=memory,
    )
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Gradient accumulation: {cfg.gradient_accumulation_steps}")
    print(f"  Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps}")
    print(f"  Truncation length: {cfg.truncation_length}")
    print(f"  Max steps: {cfg.max_steps}")
    print(f"  Output dir: {cfg.output_dir}")

    return trainer


def main() -> None:
    print("=" * 60)
    print("TRM-Bank v3.0 — Colab Training (T4 optimised)")
    print("=" * 60)

    device = check_gpu()

    stage1_path = os.path.join(DATA_DIR, "stage1_qrandlora.jsonl")
    stage2_path = os.path.join(DATA_DIR, "stage2_router.jsonl")
    for p in (stage1_path, stage2_path):
        if not os.path.isfile(p):
            print(f"\n[ERROR] Dataset not found: {p}")
            print("  Run `python prepare_datasets.py` first.")
            sys.exit(1)

    # ---- Stage 1: QRandLoRA pretraining (500 steps) ----
    trainer1 = build_trainer_for_stage(
        "Stage 1 — QRandLoRA Pretraining (500 steps)",
        stage1_path,
        device,
        config_overrides=STAGE1_OVERRIDES,
    )
    print("\n[Training] Stage 1: QRandLoRA pretraining...")
    history1 = trainer1.train(max_steps=STAGE1_OVERRIDES["max_steps"])
    final_loss1 = history1["train_loss"][-1] if history1["train_loss"] else float("nan")
    print(f"  Stage 1 complete. Final train loss: {final_loss1:.4f}")

    # ---- Stage 2: Router + LPRM tuning (300 steps) ----
    trainer2 = build_trainer_for_stage(
        "Stage 2 — Router + LPRM Tuning (300 steps)",
        stage2_path,
        device,
        config_overrides=STAGE2_OVERRIDES,
    )
    print("\n[Training] Stage 2: Router + LPRM tuning...")
    history2 = trainer2.train(max_steps=STAGE2_OVERRIDES["max_steps"])
    final_loss2 = history2["train_loss"][-1] if history2["train_loss"] else float("nan")
    print(f"  Stage 2 complete. Final train loss: {final_loss2:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("TRM-Bank v3.0 training complete.")
    print(f"  Stage 1 (QRandLoRA, 500 steps) loss: {final_loss1:.4f}")
    print(f"  Stage 2 (Router+LPRM, 300 steps) loss: {final_loss2:.4f}")
    print(f"  Checkpoints: {STAGE1_OVERRIDES['output_dir']}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
