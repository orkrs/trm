from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch

from config import CONFIG


def set_seed(seed: Optional[int] = None) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Seed value.  Defaults to CONFIG.seed.
    """
    seed = seed if seed is not None else CONFIG.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    """Resolve the target device from config."""
    return torch.device(CONFIG.device)


def resolve_dtype() -> torch.dtype:
    """Resolve the default dtype from config."""
    return CONFIG.model.pretrained_dtype


def count_params(
    module: torch.nn.Module,
    trainable_only: bool = True,
) -> Tuple[int, int]:
    """Count total and trainable parameters.

    Args:
        module: PyTorch module.
        trainable_only: If True, only count trainable params for
                        the second value.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


__all__ = [
    "set_seed",
    "resolve_device",
    "resolve_dtype",
    "count_params",
]
