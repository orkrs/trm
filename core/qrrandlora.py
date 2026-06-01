from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger


class QRandLoRALayer(nn.Module):
    """Quantized Random Low-Rank Adaptation layer.

    Implements the QRandLoRA update:

        Delta_W = sum_{j=1}^{n} B_j Lambda_j A_j Gamma_j

    where A_j and B_j are frozen sparse ternary matrices with entries
    from {-1, 0, 1} and sparsity s.  Lambda_j and Gamma_j are trainable
    diagonal scaling matrices.

    Reference: TRM-Bank v3.0 - Symbolic Knowledge Distillation via RandLoRA.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        num_components: Number of low-rank components n.
        lora_dim: Inner dimension r of each component.
        sparsity: Fraction of non-zero entries in A_j and B_j (s).
        device: Target device.
        dtype: Target dtype.

    Raises:
        ValueError: If in_features or out_features is non-positive.
        ValueError: If sparsity is outside (0, 1).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_components: int = 8,
        lora_dim: int = 64,
        sparsity: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in_features ({in_features}) and out_features ({out_features}) must be positive."
            )
        if not 0.0 < sparsity < 1.0:
            raise ValueError(f"sparsity must be in (0, 1), got {sparsity}.")

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.num_components: int = num_components
        self.lora_dim: int = lora_dim
        self.sparsity: float = sparsity

        factory = {"device": device, "dtype": dtype}

        # Frozen sparse ternary matrices A_j and B_j for each component.
        # A_j: (num_components, out_features, lora_dim)
        # B_j: (num_components, lora_dim, in_features)
        # Created on CPU to avoid GPU OOM, then moved to target device.
        A_data, B_data = self._init_ternary_matrices()
        if device is not None and device.type == "cuda":
            A_data = A_data.to(device)
            B_data = B_data.to(device)
        self.register_buffer("A_frozen", A_data)
        self.register_buffer("B_frozen", B_data)

        # Trainable diagonal scaling matrices Lambda_j and Gamma_j.
        # Lambda_j: (num_components, lora_dim)  (diagonal entries)
        # Gamma_j: (num_components, lora_dim)   (diagonal entries)
        self.Lambda = nn.Parameter(
            torch.ones(num_components, lora_dim, **factory)
        )
        self.Gamma = nn.Parameter(
            torch.ones(num_components, lora_dim, **factory)
        )

    def _init_ternary_matrices(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate frozen sparse ternary matrices A and B.

        Each matrix has entries from {-1, 0, 1} with sparsity s.
        Non-zero positions are randomly selected and assigned +1 or -1
        with equal probability.

        Tensors are created on CPU first to avoid GPU OOM, then moved
        to the target device (caller should handle the move).

        Args:
            device: Not used (kept for API compatibility).

        Returns:
            Tuple of (A, B) tensors on CPU.
        """
        n, r, d_in, d_out = (
            self.num_components,
            self.lora_dim,
            self.in_features,
            self.out_features,
        )
        num_nonzero = max(1, int(r * d_in * self.sparsity))
        mask = torch.zeros(n, r, d_in, dtype=torch.int8)
        for i in range(n):
            idx = torch.randperm(r * d_in)[:num_nonzero]
            mask.view(n, -1)[i, idx] = 1
        signs = torch.where(
            torch.rand(n, r, d_in) > 0.5,
            torch.tensor(1, dtype=torch.int8),
            torch.tensor(-1, dtype=torch.int8),
        )
        A = mask * signs
        del mask, signs

        # B: (n, d_out, r)
        num_nonzero_b = max(1, int(d_out * r * self.sparsity))
        mask_b = torch.zeros(n, d_out, r, dtype=torch.int8)
        for i in range(n):
            idx = torch.randperm(d_out * r)[:num_nonzero_b]
            mask_b.view(n, -1)[i, idx] = 1
        signs_b = torch.where(
            torch.rand(n, d_out, r) > 0.5,
            torch.tensor(1, dtype=torch.int8),
            torch.tensor(-1, dtype=torch.int8),
        )
        B = mask_b * signs_b
        del mask_b, signs_b

        return A, B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply QRandLoRA update to input.

        Computes:
            y = x + sum_j B_j Lambda_j A_j Gamma_j x

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).

        Raises:
            RuntimeError: If input shape is incompatible.
        """
        return self._apply(x)

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        """Inner forward: compute the LoRA additive update.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # A: (n, r, d_in), B: (n, d_out, r)
        A = self.A_frozen.to(x.dtype)
        B = self.B_frozen.to(x.dtype)
        L = self.Lambda     # (n, r)
        G = self.Gamma      # (n, r)

        # For each component j:
        #   h = A_j @ (Gamma_j * x)  -- wait, A_j maps d_in -> r
        # Actually: in forward order x -> Gamma_j -> A_j -> Lambda_j -> B_j -> out
        # But Gamma_j is (r,) diagonal, so it scales A_j output (r-dim)
        # So: h_j = A_j @ x  (r-dim), then Gamma_j * h_j (r-dim), then Lambda_j * that (r-dim)
        # Then: out = B_j @ (Lambda_j * Gamma_j * h_j) = sum over j

        # h: (..., n, r)  = A_j @ x for each component j
        h = torch.einsum("n r i, ... i -> ... n r", A, x)
        # h: (..., n, r) -> apply Gamma then Lambda scaling
        h = h * G * L       # (..., n, r)
        # out: (..., d_out) = sum_j B_j @ h_j
        out = torch.einsum("n d r, ... n r -> ... d", B, h)
        return out

    @torch.no_grad()
    def rebuild(self) -> torch.Tensor:
        """Materialize the full Delta_W matrix.

        Returns:
            The accumulated LoRA update of shape (out_features, in_features).
        """
        A = self.A_frozen.to(self.Lambda.dtype)
        B = self.B_frozen.to(self.Lambda.dtype)
        L = self.Lambda      # (n, r)
        G = self.Gamma       # (n, r)

        # A: (n, r, d_in), B: (n, d_out, r)
        # Apply Gamma scaling to A columns: A_scaled = A * G.unsqueeze(-1)
        A_scaled = A * G.unsqueeze(-1)        # (n, r, d_in)
        # Apply Lambda scaling to B columns: B_scaled = B * L.unsqueeze(1)
        B_scaled = B * L.unsqueeze(1)          # (n, d_out, r)

        # Delta_W_j = B_scaled_j @ A_scaled_j
        # (d_out, r) @ (r, d_in) -> (d_out, d_in)
        delta_W = torch.einsum("n o r, n r i -> o i", B_scaled, A_scaled)
        return delta_W

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_components={self.num_components}, "
            f"lora_dim={self.lora_dim}, "
            f"sparsity={self.sparsity}"
        )


class QRandLoRALinear(nn.Module):
    """Linear layer wrapped with QRandLoRA.

    Applies the base linear transformation (frozen) and adds the
    QRandLoRA adaptation on top:  y = W_base x + alpha * Delta_W x.

    Args:
        base_weight: Pretrained frozen weight tensor of shape (out_features, in_features).
        base_bias: Optional pretrained frozen bias tensor.
        num_components: Number of QRandLoRA components.
        lora_dim: Inner dimension of each low-rank component.
        sparsity: Fraction of non-zero entries in ternary matrices.
        alpha: Scaling factor for the LoRA additive update.
        device: Target device.
        dtype: Target dtype.
    """

    def __init__(
        self,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor] = None,
        num_components: int = 8,
        lora_dim: int = 64,
        sparsity: float = 0.1,
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        factory = {"device": device, "dtype": dtype}
        self.base_weight: torch.Tensor = base_weight.to(**factory)
        self.base_weight.requires_grad = False
        self.base_bias: Optional[torch.Tensor] = None
        if base_bias is not None:
            self.base_bias = base_bias.to(**factory)
            self.base_bias.requires_grad = False

        self.alpha: float = alpha

        out_features, in_features = base_weight.shape
        self.qrandlora = QRandLoRALayer(
            in_features=in_features,
            out_features=out_features,
            num_components=num_components,
            lora_dim=lora_dim,
            sparsity=sparsity,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base transformation + alpha * QRandLoRA update.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # x: (..., D_in) -> base: (..., D_out)
        y = F.linear(x, self.base_weight, self.base_bias)
        # y: (..., D_out) -> base output
        lora_out = self._apply_lora(x)
        # lora_out: (..., D_out) -> LoRA additive update
        return y + self.alpha * lora_out

    def _apply_lora(self, x: torch.Tensor) -> torch.Tensor:
        """Apply QRandLoRA adaptation to input.

        Computes:
            out = sum_j B_j @ diag(Lambda_j) @ A_j @ diag(Gamma_j) @ x

        where A_j and B_j are frozen sparse ternary matrices and
        Lambda_j, Gamma_j are trainable diagonal scaling matrices.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            LoRA additive update of shape (..., out_features).
        """
        A = self.qrandlora.A_frozen.to(x.dtype)     # (n, r, d_in)
        B = self.qrandlora.B_frozen.to(x.dtype)     # (n, d_out, r)
        L = self.qrandlora.Lambda                    # (n, r)
        G = self.qrandlora.Gamma                     # (n, r)

        h = torch.einsum("n r i, ... i -> ... n r", A, x)
        # h: (..., n, r) -> A_j @ x for each component j
        h = h * L * G
        # h: (..., n, r) -> scaled by Lambda_j and Gamma_j
        out = torch.einsum("n d r, ... n r -> ... d", B, h)
        # out: (..., d_out) -> sum over components j
        return out


class QRandLoRALinear4bit(nn.Module):
    """Linear layer wrapped with QRandLoRA for 4-bit quantized modules.

    Applies the original quantized linear layer and adds the QRandLoRA
    adaptation on top: y = base_layer(x) + alpha * Delta_W x.

    Args:
        base_layer: Quantized layer to wrap (e.g. bitsandbytes Linear4bit).
        num_components: Number of QRandLoRA components.
        lora_dim: Inner dimension of each low-rank component.
        sparsity: Fraction of non-zero entries in ternary matrices.
        alpha: Scaling factor for the LoRA additive update.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        num_components: int = 8,
        lora_dim: int = 64,
        sparsity: float = 0.1,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.alpha: float = alpha

        device = base_layer.weight.device
        dtype = getattr(base_layer, "compute_dtype", torch.bfloat16)

        self.qrandlora = QRandLoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            num_components=num_components,
            lora_dim=lora_dim,
            sparsity=sparsity,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base quantized forward + alpha * QRandLoRA update.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        y = self.base_layer(x)
        lora_out = self._apply_lora(x)
        return y + self.alpha * lora_out

    def _apply_lora(self, x: torch.Tensor) -> torch.Tensor:
        """Apply QRandLoRA adaptation to input."""
        A = self.qrandlora.A_frozen.to(x.dtype)     # (n, r, d_in)
        B = self.qrandlora.B_frozen.to(x.dtype)     # (n, d_out, r)
        L = self.qrandlora.Lambda                    # (n, r)
        G = self.qrandlora.Gamma                     # (n, r)

        h = torch.einsum("n r i, ... i -> ... n r", A, x)
        h = h * L * G
        out = torch.einsum("n d r, ... n r -> ... d", B, h)
        return out


@torch.no_grad()
def apply_qrandlora(
    model: nn.Module,
    r: int = 64,
    alpha: float = 1.0,
    sparsity: float = 0.1,
    num_components: int = 8,
    target_modules: Tuple[str, ...] = ("in_proj", "out_proj", "x_proj", "dt_proj"),
    _prefix: str = "",
) -> int:
    """Recursively wrap matching Linear layers with QRandLoRALinear.

    Traverses model tree, identifies ``nn.Linear`` children whose name
    contains any string in ``target_modules``, and replaces them with
    ``QRandLoRALinear`` (frozen base + trainable ternary LoRA).

    Args:
        model: Root module to traverse.
        r: Inner LoRA dimension.
        alpha: Scaling factor for the LoRA additive update.
        sparsity: Fraction of non-zero entries in ternary matrices.
        num_components: Number of LoRA components per layer.
        target_modules: Tuple of substring patterns to match layer names.
        _prefix: Internal recursion prefix (do not set manually).

    Returns:
        Number of layers patched.
    """
    import gc
    count: int = 0
    for name, child in list(model.named_children()):
        full_name: str = f"{_prefix}.{name}" if _prefix else name
        
        is_linear = isinstance(child, nn.Linear)
        is_bnb_linear = False
        if not is_linear:
            classname = child.__class__.__name__
            if classname in ("Linear4bit", "Linear8bitLt", "LinearParams4bit"):
                is_bnb_linear = True

        if (is_linear or is_bnb_linear) and any(t in name for t in target_modules):
            if is_linear:
                device: torch.device = child.weight.device
                dtype: torch.dtype = child.weight.dtype
                wrapper = QRandLoRALinear(
                    base_weight=child.weight.data,
                    base_bias=child.bias.data if child.bias is not None else None,
                    num_components=num_components,
                    lora_dim=r,
                    sparsity=sparsity,
                    alpha=alpha,
                    device=device,
                    dtype=dtype,
                )
            else:
                wrapper = QRandLoRALinear4bit(
                    base_layer=child,
                    num_components=num_components,
                    lora_dim=r,
                    sparsity=sparsity,
                    alpha=alpha,
                )
            setattr(model, name, wrapper)
            logger.info(
                f"QRandLoRA applied: {full_name} "
                f"({child.in_features}->{child.out_features}, "
                f"r={r}, components={num_components}, sparsity={sparsity}, "
                f"type={'bnb' if is_bnb_linear else 'standard'})"
            )
            count += 1
            gc.collect()
            torch.cuda.empty_cache()
        else:
            count += apply_qrandlora(
                child,
                r=r,
                alpha=alpha,
                sparsity=sparsity,
                num_components=num_components,
                target_modules=target_modules,
                _prefix=full_name,
            )
    return count
