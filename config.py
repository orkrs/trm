from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ModelConfig:
    pretrained_model_name: str = "state-spaces/mamba-1.4b-hf"
    quantization_bits: int = 4
    quantization_type: str = "nf4"
    hidden_dim: int = 2048
    d_state: int = 64
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: str = "auto"
    pretrained_dtype: torch.dtype = torch.bfloat16


@dataclass(frozen=True)
class ComplexMIMOConfig:
    mimo_rank: int = 2
    complex_state_dim: int = 32
    use_complex: bool = True
    discretization_order: int = 2
    lambda_init: float = 0.5
    rotation_dim: int = 64


@dataclass(frozen=True)
class QRandLoRAConfig:
    num_components: int = 8
    lora_dim: int = 64
    sparsity: float = 0.1
    ternary: bool = True
    scaling_init: float = 1.0
    target_modules: Tuple[str, ...] = ("in_proj", "x_proj", "dt_proj", "out_proj")


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


@dataclass(frozen=True)
class LPRMConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


@dataclass(frozen=True)
class RouterConfig:
    module_names: Tuple[str, ...] = (
        "mimo_brancher",
        "direct_generation",
        "symbolic_solver",
        "or_tools_solver",
        "memory_retrieval",
    )
    module_costs: Tuple[float, ...] = (10.0, 1.0, 20.0, 15.0, 2.0)
    flops_budget: float = 100.0
    lprm: LPRMConfig = field(default_factory=LPRMConfig)


@dataclass(frozen=True)
class TinyModelConfig:
    vocab_size: int = 50257
    d_model: int = 256
    d_state: int = 64
    mimo_rank: int = 4
    num_layers: int = 2


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
