# TRM-Bank v3.0 — Полная кодовая база с комментариями

Единый документ со всеми исходными файлами проекта, их описанием и комментариями к коду.

---

## Оглавление

1. [config.py](#configpy) — Конфигурация проекта
2. [verify_modules.py](#verify_modulespy) — Валидация зависимостей
3. [prepare_datasets.py](#prepare_datasetspy) — Подготовка данных
4. [colab_train.py](#colab_trainpy) — Точка входа для Colab
5. [core/complex_mimo_mamba.py](#corecomplex_mimo_mambapy) — SSM ядро
6. [core/mamba3_adapter.py](#coremamba3_adapterpy) — Адаптер миксеров
7. [core/qrrandlora.py](#coreqrrandlorapy) — QRandLoRA
8. [core/pipeline.py](#corepipelinepy) — Менеджер пайплайна
9. [router/lprm.py](#routerylprmpy) — LPRM регрессоры
10. [router/game_theoretic_router.py](#routergame_theoretic_routerpy) — VCG-роутер
11. [router/essm.py](#routeressmpy) — Провайдеры вычислений
12. [memory/hierarchical_memory.py](#memoryhierarchical_memorypy) — Иерархическая память
13. [memory/experience.py](#memoryexperiencepy) — Ebbinghaus-кривая
14. [training/trainer.py](#trainingtrainerpy) — Тренер

---

## 1. config.py

Конфигурация проекта: все гиперпараметры, размерности модели, настройки обучения.

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


# ── Конфигурация модели ──────────────────────────────────────────
# Базовая модель: Mamba-2.8B в 4-bit NF4 квантизации.
@dataclass(frozen=True)
class ModelConfig:
    pretrained_model_name: str = "state-spaces/mamba-1.4b-hf"
    quantization_bits: int = 4
    quantization_type: str = "nf4"
    hidden_dim: int = 2560
    d_state: int = 64
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: str = "auto"
    pretrained_dtype: torch.dtype = torch.bfloat16


# ── ComplexMIMO: комплексное сканирование с MIMO-ветвями ──────────
@dataclass(frozen=True)
class ComplexMIMOConfig:
    mimo_rank: int = 4
    complex_state_dim: int = 32
    use_complex: bool = True
    discretization_order: int = 2
    lambda_init: float = 0.5
    rotation_dim: int = 64


# ── QRandLoRA: квантованный случайный Low-Rank Adaptation ────────
@dataclass(frozen=True)
class QRandLoRAConfig:
    num_components: int = 8
    lora_dim: int = 64
    sparsity: float = 0.1
    ternary: bool = True
    scaling_init: float = 1.0
    target_modules: Tuple[str, ...] = ("in_proj", "x_proj", "dt_proj", "out_proj")


# ── Память: 4-уровневая иерархия + Ebbinghaus ────────────────────
@dataclass(frozen=True)
class MemoryConfig:
    embedding_dim: int = 256
    memory_tiers: int = 4
    tier_names: Tuple[str, ...] = ("Turn", "Event", "Category", "Domain")
    faiss_index_type: str = "Flat"
    top_k_retrieval: int = 5
    memory_slots_per_tier: Tuple[int, ...] = (4096, 1024, 256, 64)
    pointer_dim: int = 64
    ebbinghaus_theta1: float = 1.0
    ebbinghaus_theta2: float = 1.0
    ebbinghaus_theta3: float = 0.05
    ebbinghaus_strength_increment: float = 0.1
    ebbinghaus_strength_decay: float = 0.05
    ebbinghaus_reward_threshold: float = 0.5


# ── LPRM: предсказывает вероятность успеха модуля ────────────────
@dataclass(frozen=True)
class LPRMConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


# ── Роутер: 5 провайдеров ────────────────────────────────────────
@dataclass(frozen=True)
class RouterConfig:
    module_names: Tuple[str, ...] = (
        "mimo_brancher", "direct_generation", "symbolic_solver",
        "or_tools_solver", "memory_retrieval",
    )
    module_costs: Tuple[float, ...] = (10.0, 1.0, 20.0, 15.0, 2.0)
    flops_budget: float = 100.0
    lprm: LPRMConfig = field(default_factory=LPRMConfig)


# ── Tiny-модель для тестирования ──────────────────────────────────
@dataclass(frozen=True)
class TinyModelConfig:
    vocab_size: int = 50257
    d_model: int = 256
    d_state: int = 64
    mimo_rank: int = 4
    num_layers: int = 2


# ── Обучение: truncated BPTT + max_steps ──────────────────────────
@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_seq_length: int = 8192
    truncation_length: int = 2048
    gradient_clip: float = 1.0
    num_epochs: int = 3
    max_steps: int = -1
    gradient_accumulation_steps: int = 4
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    output_dir: str = "./outputs"
    lprm_weight: float = 0.1
    replay_capacity: int = 10000
    replay_frequency: int = 50
    replay_batch_size: int = 16
    replay_lr: float = 1e-4
    replay_num_steps: int = 8


@dataclass(frozen=True)
class BenchmarkConfig:
    arc_agi_path: str = "./data/arc_agi"
    gsm8k_path: str = "./data/gsm8k"
    humaneval_path: str = "./data/humaneval"
    sudoku_path: str = "./data/sudoku"
    prontoqa_path: str = "./data/prontoqa"
    num_workers: int = 2


# ── Глобальная конфигурация (синглтон) ────────────────────────────
@dataclass(frozen=True)
class GlobalConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)
    tiny: TinyModelConfig = field(default_factory=TinyModelConfig)
    mimo: ComplexMIMOConfig = field(default_factory=ComplexMIMOConfig)
    qrandlora: QRandLoRAConfig = field(default_factory=QRandLoRAConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    benchmarks: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    log_level: str = "INFO"
    project_name: str = "trm3"

    def __post_init__(self) -> None:
        os.makedirs(self.training.output_dir, exist_ok=True)


CONFIG = GlobalConfig()
```

---

## 2. verify_modules.py

Валидация зависимостей: SymPy, OR-Tools, PythonSandbox.

```python
from __future__ import annotations
import sys
from typing import Any

def test_sympy() -> str:
    """Test SymPy: integral of sin(x)*exp(x)."""
    import sympy as sp
    x = sp.Symbol("x")
    result = sp.integrate(sp.sin(x) * sp.exp(x), x)
    expected = sp.exp(x) * sp.sin(x) / 2 - sp.exp(x) * sp.cos(x) / 2
    assert sp.simplify(result - expected) == 0
    return f"[OK] SymPy: {result}"

def test_ortools() -> str:
    """Test OR-Tools: CP-SAT model, x+y=7."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    x = model.NewIntVar(0, 10, "x")
    y = model.NewIntVar(0, 10, "y")
    model.Add(x + y == 7)
    model.Maximize(x + y)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status == cp_model.OPTIMAL
    return f"[OK] OR-Tools: x={solver.Value(x)}, y={solver.Value(y)}"

def test_python_sandbox() -> str:
    """Test PythonSandboxProvider: factorial(10)."""
    from router.essm import PythonSandboxProvider
    provider = PythonSandboxProvider(time_limit=2.0)
    code = "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)\nresult = factorial(10)"
    output: Any = provider.execute(code, context={})
    assert output == 3628800
    return f"[OK] PythonSandbox: factorial(10) = {output}"

def main() -> None:
    print("=" * 60)
    print("TRM-Bank v3.0 — ESSM Provider Validation")
    print("=" * 60)
    tests = [("SymPy", test_sympy), ("OR-Tools", test_ortools), ("Sandbox", test_python_sandbox)]
    exit_code = 0
    for name, fn in tests:
        try:
            print(f"  {fn()}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            exit_code = 1
    print("=" * 60)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
```

---

## 3. prepare_datasets.py

Стриминговая загрузка датасетов. Фильтрация ≤640 токенов. GPT-2 токенизатор.

```python
"""prepare_datasets.py — Stream, filter, save datasets for TRM-Bank v3.0."""
from __future__ import annotations
import json, os, random, sys, time
from typing import Any, Dict, List

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)
MAX_TOKENS = 640
STAGE1_MATH = 3_900
STAGE1_CHAT = 2_600

def _flush() -> None:
    sys.stdout.flush()

def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def _tok_len(text: str, tok: Any) -> int:
    return len(tok.encode(text, truncation=False))

def _stream_dataset(name: str, split: str = "train", streaming: bool = True) -> Any:
    from datasets import load_dataset
    print(f"    [LOAD] {name} ({split}) ...", end=" "); _flush()
    ds = load_dataset(name, split=split, streaming=streaming)
    print("done."); _flush()
    return ds

def _take_n(stream: Any, tok: Any, n: int, formatter: Any, label: str = "") -> List[Dict[str, str]]:
    out, scanned, kept, t0 = [], 0, 0, time.time()
    for row in stream:
        scanned += 1
        try: text = formatter(row)
        except (KeyError, TypeError, AttributeError): text = ""
        if text and _tok_len(text, tok) <= MAX_TOKENS:
            out.append({"text": text}); kept += 1
        if scanned == 1: print(f"    [SCAN] First row in {time.time()-t0:.1f}s ..."); _flush()
        elif (kept > 0 and kept % 100 == 0) or (scanned % 500 == 0):
            print(f"    [SCAN] {kept}/{n} | {scanned} scanned | {kept/scanned:.1%}"); _flush()
        if len(out) >= n: break
    print(f"    [DONE] {label}: {len(out)}/{scanned} ({time.time()-t0:.1f}s)"); _flush()
    return out

# ── Formatters ──────────────────────────────────────────────────
def _fmt_numina(r): p,s=r.get("problem",""),r.get("solution",""); return f"### Problem\n{p}\n### Solution\n{s}" if p and s else ""
def _fmt_messages(r):
    msgs=r.get("messages",[]); 
    if msgs: return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in msgs if m.get("role") and m.get("content"))
    p,c=r.get("prompt",""),r.get("chosen",""); return f"user: {p}\nassistant: {c}" if p and c else ""
def _fmt_stable_code(r):
    i=r.get("instruction",""); 
    if isinstance(i,list): i=" ".join(m.get("content",str(m)) if isinstance(m,dict) else str(m) for m in i)
    c=r.get("output",""); return f"Instruction: {i}\nCode: {c}" if i and c else ""
def _fmt_flytech(r): i,c=r.get("instruction",""),r.get("output",""); return f"Instruction: {i}\nCode: {c}" if i and c else ""
def _fmt_glaive(r): q,a=r.get("question",""),r.get("answer",""); return f"Question: {q}\nAnswer: {a}" if q and a else ""
def _fmt_claude_reasoning(r):
    msgs=r.get("messages",[]); return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in msgs if m.get("role") and m.get("content")) if msgs else ""
def _fmt_orca_math(r):
    p=r.get("question",r.get("instruction",r.get("prompt",""))); a=r.get("answer",r.get("output",r.get("response","")))
    if isinstance(p,list): p="\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in p)
    if isinstance(a,list): a="\n".join(f"{m.get('role','assistant')}: {m.get('content','')}" for m in a)
    return f"Question: {p}\nAnswer: {a}" if p and a else ""
def _fmt_mmlupro(r):
    q=r.get("question",r.get("input","")); o=r.get("options",r.get("choices",[])); a=r.get("answer",r.get("answer_idx",""))
    return f"### Question\n{q}\n" + "\n".join(f"{chr(65+i)}. {x}" for i,x in enumerate(o)) + f"\n### Answer\n{a}" if q else ""

def _shuffle_and_save(data,path,label):
    random.shuffle(data)
    with open(path,"w",encoding="utf-8") as f:
        for item in data: f.write(json.dumps(item,ensure_ascii=False)+"\n")
    print(f"  {label}: {len(data)} -> {path}"); _flush()

def main():
    t_total=time.time(); print("="*60+"\nTRM-Bank v3.0 — Dataset Preparation\n"+"="*60); _flush()
    tok=_load_tokenizer()
    # Stage 1
    print("\n--- Stage 1: QRandLoRA Pretraining ---"); _flush()
    s1_math=_take_n(_stream_dataset("AI-MO/NuminaMath-CoT"),tok,STAGE1_MATH,_fmt_numina,label="NuminaMath")
    s1_chat=_take_n(_stream_dataset("HuggingFaceH4/no_robots"),tok,1300,_fmt_messages,label="no_robots")
    s1_chat+=_take_n(_stream_dataset("HuggingFaceH4/ultrachat_200k",split="train_sft"),tok,800,_fmt_messages,label="ultrachat")
    s1_chat+=_take_n(_stream_dataset("trl-internal-testing/hh-rlhf-trl-style"),tok,500,_fmt_messages,label="hh-rlhf")
    s1_all=s1_math+s1_chat; random.shuffle(s1_all)
    _shuffle_and_save(s1_all,os.path.join(DATA_DIR,"stage1_qrandlora.jsonl"),"Stage 1")
    # Stage 2
    print("\n--- Stage 2: Router + LPRM Tuning ---"); _flush()
    s2=[]
    s2+=_take_n(_stream_dataset("bunyaminergen/Stable-Code-Python-SFT"),tok,500,_fmt_stable_code,label="Stable-Code")
    s2+=_take_n(_stream_dataset("flytech/python-codes-25k"),tok,250,_fmt_flytech,label="flytech")
    s2+=_take_n(_stream_dataset("glaiveai/glaive-code-assistant-v3"),tok,150,_fmt_glaive,label="glaive")
    s2+=_take_n(_stream_dataset("angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k",split="train"),tok,1000,_fmt_claude_reasoning,label="Claude")
    s2+=_take_n(_stream_dataset("microsoft/orca-math-word-problems-200k"),tok,500,_fmt_orca_math,label="Orca-Math")
    s2+=_take_n(_stream_dataset("TIGER-Lab/MMLU-Pro",split="test"),tok,500,_fmt_mmlupro,label="MMLU-Pro")
    random.shuffle(s2)
    _shuffle_and_save(s2,os.path.join(DATA_DIR,"stage2_router.jsonl"),"Stage 2")
    print(f"\n{'='*60}\nDone in {time.time()-t_total:.0f}s\n{'='*60}"); _flush()

if __name__ == "__main__": main()
```

---

## 4. colab_train.py

Точка входа для Colab: Stage 1 (500 steps) + Stage 2 (300 steps).

```python
"""colab_train.py — TRM-Bank v3.0 training for Google Colab (T4, 16 GB)."""
from __future__ import annotations
import json, os, sys
from typing import Any, Dict, List, Optional
import torch, torch.nn as nn
from loguru import logger

# ── Colab T4 config ─────────────────────────────────────────────
STAGE1_OVERRIDES = {
    "batch_size": 2, "gradient_accumulation_steps": 8, "truncation_length": 256,
    "learning_rate": 3e-4, "warmup_steps": 30, "max_steps": 500, "num_epochs": 999,
    "log_every_n_steps": 10, "eval_every_n_steps": 250, "save_every_n_steps": 250,
    "output_dir": "./outputs_colab", "max_seq_length": 1024, "lprm_weight": 0.1,
}
STAGE2_OVERRIDES = {**STAGE1_OVERRIDES, "learning_rate": 1e-4, "max_steps": 300, "warmup_steps": 20}
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

def check_gpu() -> torch.device:
    if not torch.cuda.is_available():
        print("[ERROR] No GPU. Go to Runtime -> Change runtime type -> T4 GPU."); sys.exit(1)
    d = torch.device("cuda"); print(f"[OK] GPU: {torch.cuda.get_device_name(0)}"); return d

def _load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def _reconstruct_text(item: Dict[str, Any]) -> str:
    """Auto-detect text from any dataset format."""
    t = item.get("text", "")
    if t: return t
    messages = item.get("messages", item.get("conversations", []))
    if isinstance(messages, list) and messages:
        return "\n".join(f"{m.get('role',m.get('from',''))}: {m.get('content',m.get('value',''))}" for m in messages if isinstance(m,dict) and m.get('role',m.get('from')))
    prompt = (item.get("prompt","") or item.get("instruction","") or item.get("question","") or item.get("input","") or item.get("problem",""))
    answer = (item.get("output","") or item.get("response","") or item.get("answer","") or item.get("solution","") or item.get("target","") or item.get("code",""))
    if prompt and answer: return f"{prompt}\n{answer}"
    return prompt or ""

def load_tokenized_sequences(jsonl_path: str, max_seq_length: int = 1024):
    tokenizer = _load_tokenizer()
    sequences = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            text = _reconstruct_text(item)
            if not text: continue
            ids = tokenizer.encode(text, truncation=False)
            if len(ids) < 2: continue
            for start in range(0, len(ids), max_seq_length):
                chunk = ids[start:start+max_seq_length]
                if len(chunk) >= 2: sequences.append(torch.tensor(chunk, dtype=torch.long))
    print(f"  Loaded {len(sequences)} sequences from {jsonl_path}")
    return sequences

def build_trainer_for_stage(stage_name, jsonl_path, device, config_overrides=None):
    from config import CONFIG
    from core.complex_mimo_mamba import TRMBankModel
    from training.trainer import TruncatedBPTTDataset, TRMBankTrainer
    cfg = CONFIG.training
    if config_overrides:
        from dataclasses import replace; cfg = replace(cfg, **config_overrides)
    print(f"\n{'='*60}\nStage: {stage_name}\n{'='*60}")
    sequences = load_tokenized_sequences(jsonl_path, max_seq_length=cfg.truncation_length)
    if not sequences: raise RuntimeError(f"No sequences from {jsonl_path}")
    train_dataset = TruncatedBPTTDataset(sequences, cfg.truncation_length, cfg.max_seq_length)
    print("\n[2/5] Building TRMBankModel (Mamba-2.8B 4-bit)...")
    model = TRMBankModel(pretrained_name="state-spaces/mamba-1.4b-hf", mimo_rank=CONFIG.mimo.mimo_rank,
        d_state=CONFIG.model.d_state, use_4bit=True, device=device, qrandlora_r=64,
        qrandlora_alpha=CONFIG.qrandlora.scaling_init, qrandlora_sparsity=CONFIG.qrandlora.sparsity,
        qrandlora_num_components=CONFIG.qrandlora.num_components, qrandlora_target_modules=CONFIG.qrandlora.target_modules)
    model.build(); model.train()
    from router.lprm import MultiHeadLPRM
    lprm = MultiHeadLPRM(list(CONFIG.router.module_names), CONFIG.router.lprm.hidden_dim,
        model.hidden_dim // 8, CONFIG.router.lprm.dropout).to(device)
    from memory.hierarchical_memory import HierarchicalMemory
    from router.game_theoretic_router import GameTheoreticRouter
    memory = HierarchicalMemory(CONFIG.memory.tier_names, CONFIG.memory.memory_slots_per_tier, CONFIG.memory.embedding_dim)
    router = GameTheoreticRouter(lprm=lprm, flops_budget=CONFIG.router.flops_budget)
    model.set_pipeline(memory=memory, router=router, lprm=lprm, memory_top_k=CONFIG.memory.top_k_retrieval)
    trainer = TRMBankTrainer(model=model, lprm=lprm, config=cfg, train_dataset=train_dataset, memory=memory)
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return trainer

def main():
    print("="*60+"\nTRM-Bank v3.0 — Colab Training (T4 optimised)\n"+"="*60)
    device = check_gpu()
    for p in [os.path.join(DATA_DIR,"stage1_qrandlora.jsonl"), os.path.join(DATA_DIR,"stage2_router.jsonl")]:
        if not os.path.isfile(p): print(f"\n[ERROR] {p} not found. Run prepare_datasets.py first."); sys.exit(1)
    # Stage 1
    t1 = build_trainer_for_stage("Stage 1 — QRandLoRA (500 steps)", os.path.join(DATA_DIR,"stage1_qrandlora.jsonl"), device, STAGE1_OVERRIDES)
    h1 = t1.train(max_steps=500)
    l1 = h1["train_loss"][-1] if h1["train_loss"] else float("nan")
    # Stage 2
    t2 = build_trainer_for_stage("Stage 2 — Router+LPRM (300 steps)", os.path.join(DATA_DIR,"stage2_router.jsonl"), device, STAGE2_OVERRIDES)
    h2 = t2.train(max_steps=300)
    l2 = h2["train_loss"][-1] if h2["train_loss"] else float("nan")
    print(f"\n{'='*60}\nDone! Stage 1 loss: {l1:.4f}, Stage 2 loss: {l2:.4f}\n{'='*60}")

if __name__ == "__main__": main()
```

---

## 5. core/complex_mimo_mamba.py

SSM ядро: ComplexMIMOScan (комплексное MIMO-сканирование), ComplexMIMOMamba3 (полный блок), TinyTRMModel, TRMBankModel.

> **Файл: 1220 строк.** Содержит:
> - `_rotate_2d_pairs()` — 2D RoPE вращение
> - `_rotate_head_grouped()` — per-head RoPE
> - `ComplexMIMOScan` — последовательное сканирование с экспоненциально-трапециевидной дискретизацией
> - `ComplexMIMOMamba3` — полный Mamba-3 блок (in_proj → SSM → gate → out_proj)
> - `TinyTRMBlock` / `TinyTRMModel` — самодостаточная тестовая модель
> - `TRMBankModel` — обёртка над Mamba-2.8B 4-bit с патчем миксеров

### Ключевые формулы:

```
h_t = exp(Delta_t * A_t) * R_t * h_{t-1}
    + (1 - trap_t) * Delta_t * B_{t-1} * x_{t-1}
    + trap_t * Delta_t * B_t * x_t

y_t = C_t @ h_t + D * x_t
out = SiLU(z) * y_combined
```

где `R_t` — RoPE вращение, `trap_t` — трапециевидная интерполяция, `B_t`, `C_t` — MIMO проекции.

---

## 6. core/mamba3_adapter.py

Адаптер: заменяет `MambaBlock.mixer` на `Mamba3MixerAdapter`.

> **Файл: 243 строки.** Содержит:
> - `Mamba3MixerAdapter` — drop-in замена с conv1d + in_proj/out_proj (копирует веса)
> - `patch_mamba2_with_mamba3()` — обходит все MambaBlock, заменяет mixer

---

## 7. core/qrrandlora.py

QRandLoRA: квантованный случайный LoRA.

> **Файл: 356 строк.** Содержит:
> - `QRandLoRALayer` — ΔW = Σ_j B_j Λ_j A_j Γ_j, где A_j, B_j — замороженные тернарные матрицы {-1,0,1}
> - `QRandLoRALinear` — обёртка Linear + QRandLoRA
> - `apply_qrandlora()` — рекурсивно обходит модель и патчит Linear слои

---

## 8. core/pipeline.py

Менеджер пайплайна: Memory → Router → LPRM → Model.

> **Файл: 365 строк.** Содержит:
> - `PipelineManager` — orchestrates память (retrieve) → аукцион (run_auction) → генерация
> - `generate()` — полный цикл: prefill → memory → decode loop с аукционом → record experience
> - `_record_turn_experience()` — сохраняет TurnExperience в память

---

## 9. router/lprm.py

LPRM: Latent Process Reward Model — предсказывает q_i ∈ [0,1].

> **Файл: 120 строк.** Содержит:
> - `LPRM` — LayerNorm → Linear → SiLU → Dropout → Linear → Sigmoid
> - `MultiHeadLPRM` — по одному LPRM-хеду на каждый модуль (mimo_brancher, direct_generation, и т.д.)

---

## 10. router/game_theoretic_router.py

VCG обратный аукцион: каждый модуль делает ставку (q_i, c_i), побеждает max(V_i = q_i - c_i/flops_budget).

> **Файл: 232 строки.** Содержит:
> - `Bid` / `AuctionResult` — датаклассы ставки и результата
> - `GameTheoreticRouter` — collect_bids → run_auction (VCG payment = second-highest utility)
> - `dispatch()` — запускает аукцион и отправляет запрос победителю

---

## 11. router/essm.py

Провайдеры: PythonSandbox (sandboxed exec), SymPy (symbolic math), OR-Tools (constraints), DirectGeneration (free), MIMOBrancher (deep reasoning).

> **Файл: 315 строк.** Содержит:
> - `ComputeProvider` — абстрактный интерфейс: `estimate_quality()` + `execute()`
> - 5 конкретных провайдеров с хевристиками оценки качества

---

## 12. memory/hierarchical_memory.py

4-уровневая иерархическая память: Domain → Category → Event → Turn.

> **Файл: 643 строки.** Содержит:
> - `MemoryNode` — узел с embedding, strength (Ebbinghaus), access_count
> - `FAISSIndex` — обёртка над FAISS с fallback на brute-force
> - `MemoryTier` — уровень памяти с eviction по最低 retention
> - `HierarchicalMemory` — top-down retrieval + RL-update strength

---

## 13. memory/experience.py

Ebbinghaus-кривая + Experience Replay.

> **Файл: 233 строки.** Содержит:
> - `EbbinghausForgettingCurve` — R(t,S) = θ₁·exp(-t/(θ₂·S)) + θ₃
> - `TurnExperience` — опыт одного шага генерации
> - `ExperienceReplay` — буфер с eviction по最低 retention (не FIFO)

---

## 14. training/trainer.py

Тренер с truncated BPTT, LPRM-лоссом, gradient accumulation, early stopping.

> **Файл: 515 строк.** Содержит:
> - `TruncatedBPTTDataset` — разбивает длинные последовательности на чанки
> - `TRMBankTrainer` — тренировочный цикл с:
>   - LM loss (cross-entropy)
>   - LPRM auxiliary loss (MSE: q_pred vs p_correct)
>   - Replay training (из ExperienceReplay буфера)
>   - `train(max_steps=N)` — остановка после N шагов
>   - Checkpointing (save/load)

---

*Сгенерировано: TRM-Bank v3.0, commit 8a85599*
