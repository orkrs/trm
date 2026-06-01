from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPRM(nn.Module):
    """Latent Process Reward Model.

    A regression head that predicts the success probability
    q_i in [0, 1] for a given cognitive module based on the
    current latent hidden state.

    Architecture:
        hidden_state -> LayerNorm -> Linear(d, h) -> SiLU -> Dropout
        -> Linear(h, 1) -> Sigmoid -> q_i

    Args:
        hidden_dim: Input feature dimension from the backbone.
        num_heads: Number of hidden units in the regression MLP.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        num_heads: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim: int = hidden_dim
        self.num_heads: int = num_heads

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, num_heads, bias=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_heads, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict success probability from hidden state.

        Args:
            x: Hidden state tensor of shape (B, D) or (B, L, D).

        Returns:
            Predicted quality q_i of shape (B, 1) or (B, L, 1)
            with values in [0, 1].
        """
        # x: (..., D) -- supports both per-token and pooled
        x = self.norm(x)
        # x: (..., D) -> normalized
        x = self.fc1(x)
        # x: (..., H) -> first projection
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x: (..., 1) -> scalar quality
        q = torch.sigmoid(x)
        # q: (..., 1) -> success probability in [0, 1]
        return q


class MultiHeadLPRM(nn.Module):
    """Multi-head LPRM: one quality estimator per cognitive module.

    Maintains a separate LPRM head for each registered provider
    to predict module-specific success probabilities.

    Args:
        module_names: Names of the cognitive modules.
        hidden_dim: Input feature dimension.
        num_heads: Hidden units per head.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        module_names: List[str],
        hidden_dim: int = 2560,
        num_heads: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.module_names: List[str] = module_names
        self.num_modules: int = len(module_names)
        self.hidden_dim: int = hidden_dim

        self.heads = nn.ModuleList(
            [LPRM(hidden_dim, num_heads, dropout) for _ in range(self.num_modules)]
        )

    def forward(
        self,
        x: torch.Tensor,
        module_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict quality for one or all modules.

        Supports arbitrary batch dimensions via reshape.

        Args:
            x: Hidden state of shape (..., D) — 2D (B, D) or 3D (B, L, D).
            module_idx: If given, return quality for that module only.
                        If None, return qualities for all modules.

        Returns:
            Quality tensor: (..., 1) if module_idx is given,
            else (..., M) where M is the number of modules.
        """
        if module_idx is not None:
            return self.heads[module_idx](x)

        qualities = [head(x) for head in self.heads]
        # each head(x): (..., 1), cat at last dim -> (..., M)
        return torch.cat(qualities, dim=-1)
