from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from loguru import logger


def _rotate_2d_pairs(
    h: torch.Tensor,
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
) -> torch.Tensor:
    """Apply 2D rotation to pairs of adjacent elements in the last dim.

    Args:
        h: Tensor of shape (..., N) where N is even.
        cos_theta: Tensor of shape (..., N/2).
        sin_theta: Tensor of shape (..., N/2).

    Returns:
        Rotated tensor of same shape as h.

    Raises:
        ValueError: If N (last dim) is odd.
    """
    if h.shape[-1] % 2 != 0:
        raise ValueError(
            f"Last dimension must be even for 2D rotation, got {h.shape[-1]}."
        )
    h_2d = rearrange(h, "... (n two) -> ... n two", two=2)
    h_rot = torch.stack(
        [
            h_2d[..., 0] * cos_theta - h_2d[..., 1] * sin_theta,
            h_2d[..., 0] * sin_theta + h_2d[..., 1] * cos_theta,
        ],
        dim=-1,
    )
    return rearrange(h_rot, "... n two -> ... (n two)")


def _rotate_head_grouped(h_state: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor, nheads: int) -> torch.Tensor:
    """Apply per-head 2D RoPE rotation to SSM state.

    The state ``h_state`` of shape ``(B, D, N)`` is partitioned into ``nheads``
    groups along D.  Each group shares the same rotation angles from
    ``cos_t`` / ``sin_t``.

    Args:
        h_state: SSM state (B, D, N).  N must be even.
        cos_t: Cosine angles (B, H, N/2).
        sin_t: Sine angles (B, H, N/2).
        nheads: Number of heads H.

    Returns:
        Rotated state (B, D, N).

    Raises:
        ValueError: If N (state dim) is odd.
    """
    B, D, N = h_state.shape
    if N % 2 != 0:
        raise ValueError(
            f"State dimension N must be even for per-head rotation, got {N}."
        )
    H = nheads
    G = D // H
    h_head = rearrange(h_state, "b (h g) n -> b h g n", h=H, g=G)
    cos_e = cos_t.unsqueeze(2)
    sin_e = sin_t.unsqueeze(2)
    h_2d = rearrange(h_head, "b h g (n two) -> b h g n two", two=2)
    h_rot = torch.stack([
        h_2d[..., 0] * cos_e - h_2d[..., 1] * sin_e,
        h_2d[..., 0] * sin_e + h_2d[..., 1] * cos_e,
    ], dim=-1)
    return rearrange(h_rot, "b h g n two -> b (h g) (n two)")
    # output: (..., N)


class ComplexMIMOScan(nn.Module):
    """Complex-valued MIMO selective scan with Exponential-Trapezoidal
    discretization for TRM-Bank v3.0.

    Implements the recurrence (element-wise on D_in channels):

        h_t = exp(Delta_t * A_t) * h_{t-1}
            + (1 - lambda_t) * Delta_t * exp(Delta_t * A_t) * B_{t-1} * x_{t-1}
            + lambda_t * Delta_t * B_t * x_t

    The state h_t lives in C^{N/2} (stored as R^N with paired real dims).
    A_t rotates each complex pair by a data-dependent angle.
    B_t, C_t are rank-R matrix projections (MIMO branching).

    Args:
        d_in: Input channel dimension (D_in).
        d_state: Real-valued state dimension N (must be even).
        mimo_rank: Number of parallel MIMO branches R.
        device: Target device.
        dtype: Target dtype.
    """

    def __init__(
        self,
        d_in: int,
        d_state: int = 64,
        mimo_rank: int = 4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if d_state % 2 != 0:
            raise ValueError(f"d_state must be even, got {d_state}.")

        self.d_in: int = d_in
        self.d_state: int = d_state
        self.mimo_rank: int = mimo_rank

        factory = {"device": device, "dtype": dtype}

        # A_log: log damping coefficients for each complex pair per channel.
        # Shape: (d_in, d_state // 2)
        self.A_log = nn.Parameter(torch.randn(d_in, d_state // 2, **factory) * 0.01)

        # log_dt: log step size per channel.
        self.log_dt = nn.Parameter(torch.randn(d_in, **factory) * 0.01)

        # D: skip connection scaling per channel.
        self.D = nn.Parameter(torch.ones(d_in, **factory))

        # theta_proj: data-dependent rotation angles per complex pair.
        # Shared across channels; broadcasts automatically.
        self.theta_proj = nn.Linear(d_in, d_state // 2, bias=False, **factory)

        # lambda_proj: data-dependent interpolation factor (sigmoid output).
        self.lambda_proj = nn.Linear(d_in, 1, bias=False, **factory)

        # B_proj: MIMO input projection (B, L, N * R) -> (B, L, N, R).
        self.B_proj = nn.Linear(d_in, d_state * mimo_rank, bias=False, **factory)

        # C_proj: MIMO output projection (B, L, R * N) -> (B, L, R, N).
        self.C_proj = nn.Linear(d_in, mimo_rank * d_state, bias=False, **factory)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the complex MIMO scan over the input sequence.

        Args:
            x: Input tensor of shape (B, L, D_in).
            state: Optional initial state of shape (B, D_in, D_state).
                   Initialized to zeros if None.

        Returns:
            Tuple of:
                y: Output tensor of shape (B, L, D_in, R) where R is
                   the MIMO rank.
                final_state: Final state tensor of shape (B, D_in, D_state).

        Raises:
            ValueError: If x has fewer than 2 dimensions.
            RuntimeError: If scan encounters numerical instability.
        """
        # x: (B, L, D_in) -> input
        B, L, D_in = x.shape
        R = self.mimo_rank
        N = self.d_state
        device = x.device

        if D_in != self.d_in:
            raise ValueError(f"Expected d_in={self.d_in}, got {D_in}.")

        if state is None:
            state = torch.zeros(B, D_in, N, device=device, dtype=x.dtype)
        # state: (B, D_in, N)

        # --- Data-dependent parameter computation ---
        # theta: (B, L, N/2) - rotation angles (shared across D_in channels)
        theta = self.theta_proj(x)
        # theta: (B, L, N/2) -> expand for channel dimension
        theta = theta.unsqueeze(2)  # (B, L, 1, N/2)

        # lambda_t: (B, L, 1) - interpolation factor
        lam = torch.sigmoid(self.lambda_proj(x))  # (B, L, 1)

        # MIMO B projection
        B_proj = self.B_proj(x)                    # (B, L, N * R)
        B_t = rearrange(B_proj, "b l (n r) -> b l n r", n=N, r=R)
        # B_t: (B, L, N, R) - shared across D_in channels

        # MIMO C projection
        C_proj = self.C_proj(x)                    # (B, L, R * N)
        C_t = rearrange(C_proj, "b l (r n) -> b l r n", r=R, n=N)
        # C_t: (B, L, R, N)

        # Step size: (B, L, D_in) via learned log_dt
        dt = F.softplus(self.log_dt)               # (D_in,)
        dt = dt.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        # dt: (B, L, D_in)

        # Rotation angle per step: theta * dt (per channel)
        dt_exp = dt.unsqueeze(-1)                  # (B, L, D_in, 1)
        theta_step = theta * dt_exp                # (B, L, D_in, N/2)

        # Damping factor per step: exp(-exp(A_log) * dt)
        A_mag = torch.exp(self.A_log)              # (D_in, N/2)
        A_mag = A_mag.unsqueeze(0).unsqueeze(0)    # (1, 1, D_in, N/2)
        damping = torch.exp(-A_mag * dt_exp)       # (B, L, D_in, N/2)

        # --- Sequential scan over sequence length ---
        h = state                                  # (B, D_in, N)
        outputs: List[torch.Tensor] = []

        cos_theta_all = torch.cos(theta_step)      # (B, L, D_in, N/2)
        sin_theta_all = torch.sin(theta_step)      # (B, L, D_in, N/2)

        for t in range(L):
            # Current-step quantities
            x_t = x[:, t, :]                       # (B, D_in)
            cos_t = cos_theta_all[:, t, :, :]      # (B, D_in, N/2)
            sin_t = sin_theta_all[:, t, :, :]      # (B, D_in, N/2)
            damp_t = damping[:, t, :, :]           # (B, D_in, N/2)
            lam_t = lam[:, t, :]                   # (B, 1)

            # B_curr: (B, N, R), C_curr: (B, R, N)
            B_curr = B_t[:, t, :, :]               # (B, N, R)
            C_curr = C_t[:, t, :, :]               # (B, R, N)

            # --- Rotate state with damping ---
            h_rot = _rotate_2d_pairs(h, cos_t, sin_t)
            # h_rot: (B, D_in, N)
            damp_expanded = repeat(damp_t, "b d n -> b d (n two)", two=2)
            h_damped = h_rot * damp_expanded       # (B, D_in, N)

            # --- Current step input contribution ---
            B_x_curr = torch.einsum("b n r, b d -> b d n r", B_curr, x_t)
            # B_x_curr: (B, D_in, N, R) -> (B, D_in, N)
            delta_curr = B_x_curr.sum(dim=-1)      # (B, D_in, N)

            # --- Previous step input contribution (trapezoidal) ---
            if t > 0:
                x_prev = x[:, t - 1, :]            # (B, D_in)
                B_prev = B_t[:, t - 1, :, :]       # (B, N, R)
                cos_prev = cos_theta_all[:, t - 1, :, :]
                sin_prev = sin_theta_all[:, t - 1, :, :]
                damp_prev = damping[:, t - 1, :, :]

                B_x_prev = torch.einsum("b n r, b d -> b d n r", B_prev, x_prev)
                delta_prev = B_x_prev.sum(dim=-1)  # (B, D_in, N)

                # Rotate previous contribution
                delta_prev_rot = _rotate_2d_pairs(delta_prev, cos_prev, sin_prev)
                damp_prev_exp = repeat(damp_prev, "b d n -> b d (n two)", two=2)
                delta_prev_rot = delta_prev_rot * damp_prev_exp
            else:
                delta_prev_rot = torch.zeros_like(h_damped)

            # dt_t: (B, D_in, 1)
            dt_t = dt[:, t, :].unsqueeze(-1)       # (B, D_in, 1)

            # --- Exponential-Trapezoidal update ---
            # h_t = exp(Δt*A_t)*h_{t-1}
            #     + (1-λ)*Δt*exp(Δt*A_t)*B_{t-1}*x_{t-1}
            #     + λ*Δt*B_t*x_t
            h = (
                h_damped
                + (1.0 - lam_t.unsqueeze(-1)) * dt_t * delta_prev_rot
                + lam_t.unsqueeze(-1) * dt_t * delta_curr
            )
            # h: (B, D_in, N)

            # --- MIMO output ---
            # y_t = C_t @ h_t -> (B, R, N) @ (B, D_in, N) -> (B, D_in, R)
            y_t = torch.einsum("b r n, b d n -> b d r", C_curr, h)
            # Skip connection: y_t += D * x_t
            y_t = y_t + self.D.unsqueeze(0).unsqueeze(-1) * x_t.unsqueeze(-1)
            # y_t: (B, D_in, R)

            # Detach state for truncated BPTT compatibility.
            h = h.detach()

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        # y: (B, L, D_in, R)
        return y, h.detach()

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_state={self.d_state}, "
            f"mimo_rank={self.mimo_rank}"
        )


# ---------------------------------------------------------------------------
# ComplexMIMOMamba3 -- Full Mamba-3 block (drop-in replacement for Mamba-2 mixer)
# ---------------------------------------------------------------------------

class ComplexMIMOMamba3(nn.Module):
    """Full Mamba-3 block: in_proj -> complex MIMO SSM -> gate -> out_proj.

    Drop-in replacement for ``mamba_ssm.modules.mamba2.Mamba2`` mixer.
    All projections (z, x, dt, A, trap, angles, B, C) are computed from a
    single ``in_proj``, then a sequential scan applies:

        h_t = exp(dt*A)_t * R_t * h_{t-1}
            + (1 - trap_t) * dt_t * B_{t-1} * x_{t-1}
            + trap_t * dt_t * B_t * x_t

    with per-head RoPE rotation (R_t), per-head trap interpolation (trap_t),
    and MIMO rank-R input/output projections.  Output is gated as
    ``out = SiLU(z) * scan_out`` and fed through ``out_proj``.

    Replaces the native Mamba-2 scan with the complex MIMO scan from
    TRM-Bank v3.0 Section 2.

    Args:
        d_model: Model / embedding dimension.
        d_state: SSM state dimension (must be even).
        headdim: Dimension per head.
        mimo_rank: Number of parallel MIMO branches.
        device: Target device.
        dtype: Target dtype.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        headdim: int = 64,
        mimo_rank: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        precomputed_projections: bool = False,
        gradient_checkpointing: bool = False,
        cpu_offload_in_proj: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if d_state % 2 != 0:
            raise ValueError(f"d_state must be even, got {d_state}.")

        self.d_model: int = d_model
        self.d_state: int = d_state
        self.headdim: int = headdim
        self.mimo_rank: int = mimo_rank
        self.precomputed_projections: bool = precomputed_projections
        self.gradient_checkpointing: bool = gradient_checkpointing

        # SSM dimension = d_model (no expand factor for direct Mamba-3 style)
        d_in: int = d_model
        nheads: int = max(1, d_in // headdim)

        factory = {"device": device, "dtype": dtype}

        # ---- in_proj: d_model -> [B, C, dt, A, trap, angles] (and maybe z, x) ----
        if not precomputed_projections:
            in_total: int = (
                d_in * 2
                + d_state * nheads * mimo_rank * 2
                + nheads * 3
                + (d_state // 2) * nheads
            )
        else:
            in_total: int = (
                d_state * nheads * mimo_rank * 2
                + nheads * 3
                + (d_state // 2) * nheads
            )

        # Frozen in_proj: auto-offload to CPU on small VRAM GPUs.
        # Saves ~3.5 GB VRAM for 24 patched layers on T4 (16 GB).
        # On 2x T4 (~30 GB) or larger, weights stay on GPU.
        if cpu_offload_in_proj is None:
            self._in_proj_cpu_offload: bool = (
                precomputed_projections
                and device is not None
                and device.type == "cuda"
            )
        else:
            self._in_proj_cpu_offload = cpu_offload_in_proj

        if self._in_proj_cpu_offload:
            cpu_factory = {"device": torch.device("cpu"), "dtype": dtype}
            self.in_proj = nn.Linear(d_model, in_total, bias=False, **cpu_factory)
        else:
            self.in_proj = nn.Linear(d_model, in_total, bias=False, **factory)

        # Per-head learned parameters
        self.A_log = nn.Parameter(torch.randn(nheads, d_state, **factory) * 0.01)
        self.dt_bias = nn.Parameter(torch.randn(nheads, **factory) * 0.01)
        self.D = nn.Parameter(torch.ones(nheads, **factory))

        # MIMO per-branch scaling (learned, per head, per rank, per headdim)
        self.mimo_o = nn.Parameter(
            torch.ones(nheads, mimo_rank, headdim, **factory)
        )

        # Normalization for B and C states (per-state scalar)
        self.B_norm = nn.LayerNorm(d_state, **factory)
        self.C_norm = nn.LayerNorm(d_state, **factory)

        # Output projection + norm (only needed if not precomputed_projections)
        if not precomputed_projections:
            self.out_proj = nn.Linear(d_in, d_model, **factory)
            self.norm = nn.LayerNorm(d_model, **factory)

    @staticmethod
    def _scan_impl(
        x_ssm: torch.Tensor,
        h: torch.Tensor,
        cos_all: torch.Tensor,
        sin_all: torch.Tensor,
        dt_vals: torch.Tensor,
        A_vals: torch.Tensor,
        trap_vals: torch.Tensor,
        B_proj_vals: torch.Tensor,
        C_proj_vals: torch.Tensor,
        z_vals: torch.Tensor,
        D_head: torch.Tensor,
        mimo_o: torch.Tensor,
        N: int,
        R: int,
        H: int,
        G: int,
        P: int,
        precomputed: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x_ssm.shape
        outputs: List[torch.Tensor] = []
        prev_inp: Optional[torch.Tensor] = None
        # Accumulate state in float32 to prevent exp(A*dt) overflow in bfloat16.
        h = h.float()

        for t_idx in range(L):
            x_t = x_ssm[:, t_idx, :]
            cos_t = cos_all[:, t_idx, :, :]
            sin_t = sin_all[:, t_idx, :, :]
            dt_t = dt_vals[:, t_idx, :]
            A_t = A_vals[:, t_idx, :, :]
            trap_t = trap_vals[:, t_idx, :]
            B_t = B_proj_vals[:, t_idx, :, :, :]
            C_t = C_proj_vals[:, t_idx, :, :, :]

            A_t_exp = A_t.unsqueeze(1).expand(-1, G, -1, -1)
            A_t_exp = rearrange(A_t_exp, "b g h n -> b (g h) n")
            dt_t_exp = dt_t.unsqueeze(1).expand(-1, G, -1)
            dt_t_exp = rearrange(dt_t_exp, "b g h -> b (g h)")

            if prev_inp is not None:
                prev_rot = _rotate_head_grouped(prev_inp.float(), cos_t.float(), sin_t.float(), nheads=H)
                decay_prev = torch.exp(A_t_exp.float() * dt_t_exp.float().unsqueeze(-1))
                prev_inp_weighted = prev_rot * decay_prev
            else:
                prev_inp_weighted = torch.zeros_like(h)

            B_t_exp = B_t.unsqueeze(1).expand(-1, G, -1, -1, -1)
            B_t_exp = rearrange(B_t_exp, "b g h n r -> b (g h) n r")
            curr_inp = B_t_exp.float() * x_t.float().unsqueeze(-1).unsqueeze(-1)
            curr_inp = curr_inp.sum(dim=-1)

            h_rot = _rotate_head_grouped(h, cos_t.float(), sin_t.float(), nheads=H)
            decay = torch.exp(A_t_exp.float() * dt_t_exp.float().unsqueeze(-1))

            trap_channel = trap_t.unsqueeze(1).expand(-1, G, -1)
            trap_channel = rearrange(trap_channel, "b g h -> b (g h)")

            h = (
                decay * h_rot
                + (1.0 - trap_channel.float().unsqueeze(-1)) * dt_t_exp.float().unsqueeze(-1) * prev_inp_weighted
                + trap_channel.float().unsqueeze(-1) * dt_t_exp.float().unsqueeze(-1) * curr_inp
            )

            prev_inp = curr_inp.clone()

            C_t_exp = C_t.unsqueeze(1).expand(-1, G, -1, -1, -1)
            C_t_exp = rearrange(C_t_exp, "b g h r n -> b (g h) r n")
            y_t = torch.einsum("b d r n, b d n -> b d r", C_t_exp.float(), h).to(x_ssm.dtype)

            # Apply per-head per-branch MIMO output scaling mimo_o: (H, R, P)
            # y_t: (B, D, R) -> reshape to (B, H, G, R) -> scale -> back to (B, D, R)
            # mimo_o: (H, R, P) -> mean over P -> (H, R) -> expand to (1, H, 1, R) for broadcasting
            mimo_o_scale = mimo_o[:, :R, :P].mean(dim=-1)  # (H, R) - average over P dimension
            mimo_o_scale = mimo_o_scale.unsqueeze(0).unsqueeze(2)  # (1, H, 1, R)
            y_t_reshaped = rearrange(y_t, "b (h g) r -> b h g r", h=H, g=G)
            y_t_reshaped = y_t_reshaped * mimo_o_scale  # (B, H, G, R) * (1, H, 1, R) - broadcasts over G
            y_t = rearrange(y_t_reshaped, "b h g r -> b (h g) r")

            D_exp = D_head.unsqueeze(0).unsqueeze(1).expand(-1, G, -1)
            D_exp = rearrange(D_exp, "b g h -> b (g h)")
            y_t = y_t + D_exp.unsqueeze(-1) * x_t.unsqueeze(-1)

            if not precomputed:
                z_t = z_vals[:, t_idx, :]
                y_t = y_t * F.silu(z_t).unsqueeze(-1)

            outputs.append(y_t)

        return torch.stack(outputs, dim=1), h.detach()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_branches: bool = False,
        gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply ComplexMIMOMamba3 block.

        Args:
            x: Input of shape (B, L, D) where D = d_model.
            state: Optional initial state (B, D, d_state).
            return_branches: If True, keep per-branch outputs.
            gate: Optional precomputed gate z of shape (B, L, D).

        Returns:
            Tuple (output, final_state, branches_or_None):
                output: (B, L, D) after out_proj + residual + norm.
                final_state: (B, D, d_state).
                branches: (B, L, D, R) or None if return_branches=False.
        """
        B, L, D = x.shape
        N: int = self.d_state
        R: int = self.mimo_rank
        headdim_actual: int = min(self.headdim, D)
        H: int = max(1, D // headdim_actual)                     # nheads
        G: int = D // H                                          # channels per head
        P: int = headdim_actual
        device: torch.device = x.device
        dtpe: torch.dtype = x.dtype

        # ---- 1. in_proj and split (with CPU offload support) ----
        if self._in_proj_cpu_offload:
            # in_proj.weight is on CPU (frozen); move input to CPU, compute, result back to GPU
            x_cpu = x.to(device="cpu")
            # Detect weight dtype (handles both nn.Linear and QRandLoRALinear)
            if hasattr(self.in_proj, "weight"):
                w_dtype = self.in_proj.weight.dtype
            elif hasattr(self.in_proj, "base_weight"):
                w_dtype = self.in_proj.base_weight.dtype
            else:
                w_dtype = x.dtype
            proj = self.in_proj(x_cpu.to(dtype=w_dtype)).to(device=device, dtype=dtpe)
        else:
            proj = self.in_proj(x)
        # proj: (B, L, total)

        off: int = 0
        if not self.precomputed_projections:
            z = proj[:, :, off:off + D]
            off += D
            x_ssm = proj[:, :, off:off + D]
            off += D
        else:
            x_ssm = x
            if gate is not None:
                z = gate
            else:
                z = torch.ones_like(x_ssm)

        # B: (B, L, H, N, R)
        B_raw = proj[:, :, off:off + N * H * R]
        B_proj = rearrange(B_raw, "b l (h n r) -> b l h n r", h=H, n=N, r=R)
        off += N * H * R

        # C: (B, L, H, R, N)
        C_raw = proj[:, :, off:off + N * H * R]
        C_proj = rearrange(C_raw, "b l (h r n) -> b l h r n", h=H, r=R, n=N)
        off += N * H * R

        dt_raw = proj[:, :, off:off + H]          # (B, L, H)
        off += H
        A_raw = proj[:, :, off:off + H]           # (B, L, H)
        off += H
        trap_raw = proj[:, :, off:off + H]        # (B, L, H)
        off += H
        angles_raw = proj[:, :, off:off + (N // 2) * H]
        # angles_raw: (B, L, (N//2)*H)

        # ---- 2. SSM parameters ----
        dt = F.softplus(dt_raw + self.dt_bias.unsqueeze(0).unsqueeze(0))
        # dt: (B, L, H)

        A_base = torch.exp(self.A_log)             # (H, N)
        A_mod = torch.exp(A_raw)                   # (B, L, H)
        A = -A_base.unsqueeze(0).unsqueeze(0) * A_mod.unsqueeze(-1)
        # A: (B, L, H, N)

        trap = torch.sigmoid(trap_raw)             # (B, L, H)

        # RoPE angles
        angles = rearrange(angles_raw, "b l (h p) -> b l h p", h=H, p=N // 2)
        # angles: (B, L, H, N/2)

        # ---- 3. State initialization ----
        h = torch.zeros(B, D, N, device=device, dtype=dtpe)
        if state is not None:
            h = state.to(device=device, dtype=dtpe)
        # h: (B, D, N) -- each channel has its own state

        # ---- 4. SSM recurrence (sequential scan) ----
        # Reshape computations are per-channel; parameters broadcast over D_in.
        # B_proj shares across channels: B_proj[t] (H, N, R) applies to all channels
        # via expand: B_proj[t].unsqueeze(1) -> (1, H, N, R) broadcasts over D.

        cos_all = torch.cos(angles)
        sin_all = torch.sin(angles)

        if self.gradient_checkpointing and self.training:
            y, h = torch.utils.checkpoint.checkpoint(
                self._scan_impl,
                x_ssm, h, cos_all, sin_all, dt, A, trap, B_proj, C_proj,
                z if not self.precomputed_projections else z,
                self.D, self.mimo_o, N, R, H, G, headdim_actual, self.precomputed_projections,
                use_reentrant=False,
            )
        else:
            y, h = self._scan_impl(
                x_ssm, h, cos_all, sin_all, dt, A, trap, B_proj, C_proj,
                z if not self.precomputed_projections else z,
                self.D, self.mimo_o, N, R, H, G, headdim_actual, self.precomputed_projections,
            )
        # y: (B, L, D, R)

        branches_out: Optional[torch.Tensor] = y if return_branches else None

        # Default branch combination: average over R
        y_combined = y.mean(dim=-1)
        # y_combined: (B, L, D)

        if self.precomputed_projections:
            return y_combined, h.detach(), branches_out

        # out_proj: D -> d_model
        out = self.out_proj(y_combined)
        # out: (B, L, d_model)

        # Residual + norm
        out = self.norm(x + out)
        # out: (B, L, d_model)

        return out, h.detach(), branches_out

    def allocate_inference_cache(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero-initialized state for autoregressive decoding."""
        return torch.zeros(batch_size, self.d_model, self.d_state, device=device)

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with explicit state return (for generation prefill)."""
        out, final_state, _ = self.forward(x, state=state, return_branches=False)
        return out, final_state

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward (for decode loop).

        Args:
            x_t: Single token embedding of shape (B, D).
            state: Current SSM state of shape (B, D, N).

        Returns:
            Tuple (output, next_state).
        """
        # Add length dimension
        out, next_state, _ = self.forward(
            x_t.unsqueeze(1), state=state, return_branches=False
        )
        return out.squeeze(1), next_state

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"headdim={self.headdim}, mimo_rank={self.mimo_rank}"
        )


class TinyTRMBlock(nn.Module):
    """Single block of TinyTRMModel: Mamba-3 block with residual + norm.

    Each block wraps ComplexMIMOMamba3 which includes in_proj, complex
    MIMO scan with trapezoidal discretization, z-gating, and out_proj.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension (must be even).
        headdim: Dimension per head.
        mimo_rank: Number of parallel MIMO branches.
        device: Target device.
        dtype: Target dtype.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        headdim: int = 64,
        mimo_rank: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.mamba3 = ComplexMIMOMamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            mimo_rank=mimo_rank,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Mamba-3 block.

        Args:
            x: Input of shape (B, L, D).
            state: Optional initial state (B, D, N).

        Returns:
            Tuple of (output, final_state).
        """
        out, next_state, _ = self.mamba3(x, state=state)
        return out, next_state


class TinyTRMModel(nn.Module):
    """Self-contained tiny TRM-Bank model using ComplexMIMOScan.

    Architecture:
        embed(vocab, D) -> [TinyTRMBlock x N] -> norm -> lm_head(D, vocab)

    Does NOT require mamba_ssm or bitsandbytes.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model / embedding dimension.
        d_state: SSM state dimension.
        mimo_rank: Number of parallel MIMO branches R.
        num_layers: Number of TinyTRMBlocks.
        device: Target device.
        dtype: Target dtype.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        d_state: int = 64,
        mimo_rank: int = 4,
        num_layers: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.vocab_size: int = vocab_size
        self.d_model: int = d_model
        self.d_state: int = d_state
        self.mimo_rank: int = mimo_rank
        self.num_layers: int = num_layers
        self.headdim: int = d_state  # headdim = d_state for Mamba-3 alignment

        factory = {"device": device, "dtype": dtype}

        self.embed = nn.Embedding(vocab_size, d_model, **factory)
        self.blocks = nn.ModuleList([
            TinyTRMBlock(
                d_model, d_state=d_state, headdim=self.headdim,
                mimo_rank=mimo_rank, **factory,
            )
            for _ in range(num_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model, **factory)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory)

        self._pipeline: Any = None

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through the tiny model.

        Args:
            input_ids: Token indices of shape (B, L).
            states: Optional list of initial states, one per block.
                    Each state is (B, D, N).
            return_states: If True, return final states per block.

        Returns:
            Dictionary with keys:
                logits: Output logits of shape (B, L, V).
                states: (Optional) list of final states.
        """
        B, L = input_ids.shape
        h = self.embed(input_ids)

        next_states: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            h, ns = block(h, s)
            next_states.append(ns.detach())

        h = self.norm_f(h)
        logits = self.lm_head(h)

        result: Dict[str, Any] = {"logits": logits, "hidden_states": h}
        if return_states:
            result["states"] = next_states
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_pipeline: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Prefill run forward() once, then decode token-by-token
        carrying SSM states.

        When a ``PipelineManager`` is attached via ``set_pipeline()``
        and ``use_pipeline=True``, the full memory+router pipeline runs.

        Args:
            input_ids: Prompt of shape (B, L).
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k filter threshold.
            top_p: Nucleus sampling threshold.
            use_pipeline: Whether to use the memory+router pipeline.

        Returns:
            Generated token ids of shape (B, L + max_new_tokens).
        """
        if use_pipeline and self._pipeline is not None:
            return self._pipeline.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        B = input_ids.shape[0]
        device = input_ids.device
        seq = input_ids

        out = self.forward(seq, states=None, return_states=True)
        logits = out["logits"]
        states: List[torch.Tensor] = out["states"]

        for _ in range(max_new_tokens):
            logit = logits[:, -1, :]

            if temperature == 0.0:
                next_id = logit.argmax(dim=-1, keepdim=True)
            else:
                if top_k > 0 and top_k < logit.size(-1):
                    vals, _ = logit.topk(top_k, dim=-1)
                    logit[logit < vals[:, -1:]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_idx = logit.sort(dim=-1, descending=True)
                    cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    sorted_logits[cum_probs > top_p] = float("-inf")
                    logit = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
                probs = torch.softmax(logit / max(temperature, 1e-8), dim=-1)
                next_id = torch.multinomial(probs, 1)

            seq = torch.cat([seq, next_id], dim=-1)

            h = self.embed(next_id)
            next_states: List[torch.Tensor] = []
            for i, block in enumerate(self.blocks):
                h, ns = block(h, states[i])
                next_states.append(ns.detach())
            states = next_states

            h = self.norm_f(h)
            logits = self.lm_head(h)

        return seq

    def set_pipeline(
        self,
        memory: Any = None,
        router: Any = None,
        lprm: Any = None,
        memory_top_k: int = 5,
        router_interval: int = 0,
    ) -> None:
        """Attach memory, router, and LPRM for the full generation pipeline.

        Args:
            memory: ``HierarchicalMemory`` instance.
            router: ``GameTheoreticRouter`` instance.
            lprm: ``MultiHeadLPRM`` instance (attached to router).
            memory_top_k: Number of memory nodes to retrieve.
            router_interval: Run auction every N steps (0 = once).
        """
        from core.pipeline import PipelineManager

        if lprm is not None and router is not None:
            router.lprm = lprm

        self._pipeline = PipelineManager(
            model=self,
            memory=memory,
            router=router,
            memory_top_k=memory_top_k,
            router_interval=router_interval,
        )

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"d_state={self.d_state}, mimo_rank={self.mimo_rank}, "
            f"num_layers={self.num_layers}"
        )


class TRMBankModel(nn.Module):
    """TRM-Bank v3.0: wrapper over 4-bit quantized Mamba-1.4B.

    Loads the pretrained Mamba-1.4B model in 4-bit NF4 format via
    bitsandbytes, freezes all backbone weights, and enriches hidden
    states with ComplexMIMOScan branch features.

    Requires mamba_ssm and bitsandbytes (for full mode).
    For a self-contained alternative see TinyTRMModel.

    Args:
        pretrained_name: HuggingFace model name for Mamba-1.4B.
        mimo_rank: Number of parallel MIMO branches (R).
        d_state: Real-valued SSM state dimension (N, must be even).
        use_4bit: Whether to load in 4-bit NF4 quantization.
        device: Target device override.
    """

    def __init__(
        self,
        pretrained_name: str = "state-spaces/mamba-1.4b-hf",
        mimo_rank: int = 4,
        d_state: int = 64,
        use_4bit: bool = True,
        device: Optional[torch.device] = None,
        qrandlora_r: int = 0,
        qrandlora_alpha: float = 1.0,
        qrandlora_sparsity: float = 0.1,
        qrandlora_num_components: int = 8,
        qrandlora_target_modules: Optional[Tuple[str, ...]] = None,
        cpu_offload_in_proj: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.pretrained_name: str = pretrained_name
        self.mimo_rank: int = mimo_rank
        self.d_state: int = d_state
        self.use_4bit: bool = use_4bit
        self._device: Optional[torch.device] = device
        self._cpu_offload_in_proj: Optional[bool] = cpu_offload_in_proj

        self._qrandlora_r: int = qrandlora_r
        self._qrandlora_alpha: float = qrandlora_alpha
        self._qrandlora_sparsity: float = qrandlora_sparsity
        self._qrandlora_num_components: int = qrandlora_num_components
        self._qrandlora_target_modules: Tuple[str, ...] = (
            qrandlora_target_modules
            if qrandlora_target_modules is not None
            else ("in_proj", "out_proj")
        )

        self.backbone: Any = None
        self.hidden_dim: int = 0
        self.num_layers: int = 0
        self._cache_params: Any = None
        self._qr_patched_count: int = 0

        # Pipeline components (memory + router + LPRM)
        self._pipeline: Any = None
        self._memory: Any = None
        self._router: Any = None

    def _load_pretrained(self) -> Any:
        """Load the pretrained Mamba model with 4-bit quantization.

        When ``self._device`` is ``None`` and ``use_4bit`` is True, the
        model is loaded on CPU (``device_map=None``) so that
        ``accelerator.prepare()`` can handle DDP wrapping and device
        placement.  This avoids the PCIe bottleneck of
        ``device_map="auto"`` (pipeline parallelism).

        Returns:
            Loaded HuggingFace model.

        Raises:
            ImportError: If transformers or bitsandbytes is missing.
        """
        try:
            import transformers
        except ImportError:
            raise ImportError("transformers is required to load Mamba models.")

        if self.use_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                raise ImportError(
                    "bitsandbytes required for 4-bit quantization."
                )

            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            # device=None: load on CPU for DDP (accelerator.prepare handles placement).
            # device=<cuda>: load on specific device (for device_map="auto" single-process).
            if self._device is None:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.pretrained_name,
                    quantization_config=quantization_config,
                    device_map=None,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
            else:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.pretrained_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.pretrained_name,
                device_map=None,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            if self._device is not None:
                model = model.to(self._device)

        return model

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters except trainable adapters."""
        for name, param in self.backbone.named_parameters():
            if any(k in name for k in ("Lambda", "Gamma", "mimo_o", "A_log", "dt_bias", "D")):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _log_mem(self, tag: str) -> None:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"[MEM {tag} GPU{i}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")

    @torch.no_grad()
    def build(self) -> None:
        """Load the pretrained backbone and replace all SSM mixers.

        Loads Mamba-1.4B in 4-bit NF4, then replaces every ``MambaBlock.mixer``
        with ``Mamba3MixerAdapter`` (which wraps ``ComplexMIMOMamba3``), applies
        QRandLoRA, and freezes backbone weights.

        Call this once before forward() or generate().
        """
        from core.mamba3_adapter import patch_mamba2_with_mamba3

        self._log_mem("before_load")
        model = self._load_pretrained()
        self._log_mem("after_load")

        # Tie lm_head to embeddings if missing from checkpoint.
        # Mamba checkpoints do not include lm_head weights.
        # Copy weights instead of tying to avoid device mismatch with
        # device_map="auto" where lm_head and embeddings may be on different GPUs.
        if hasattr(model, "lm_head") and hasattr(model, "backbone"):
            emb = getattr(model.backbone, "embeddings", None)
            if emb is not None:
                with torch.no_grad():
                    model.lm_head.weight.copy_(emb.weight.data.to(model.lm_head.weight.dtype))
                logger.info("lm_head.weight copied from backbone.embeddings.weight")

        self.backbone = model
        self._log_mem("after_tie")

        config = model.config
        self.hidden_dim = getattr(config, "hidden_size", 2560)
        self.num_layers = getattr(config, "num_hidden_layers", 64)

        self._log_mem("before_patch")
        n_patched = patch_mamba2_with_mamba3(
            self.backbone,
            d_state=self.d_state,
            headdim=min(self.d_state, 64),
            mimo_rank=self.mimo_rank,
            device=self._device,
            dtype=torch.bfloat16,
            gradient_checkpointing=True,
            cpu_offload_in_proj=self._cpu_offload_in_proj,
        )
        self._log_mem("after_patch")
        logger.info(f"Patched {n_patched} mixers with ComplexMIMOMamba3")

        self._qr_patched_count = self._apply_qrandlora()
        self._log_mem("after_qrandlora")

        # Freeze the backbone parameters, keeping only the adapter parameters trainable
        self._freeze_backbone()

        # 4-bit models are already on the correct device via device_map="auto".
        # Only move non-quantized models explicitly.
        if not self.use_4bit and self._device is not None:
            self.backbone = self.backbone.to(self._device)

    @torch.no_grad()
    def _apply_qrandlora(self) -> int:
        """Apply QRandLoRA to the adapter's Linear projections.

        Traverses the patched backbone and wraps every
        ``nn.Linear`` matching ``self._qrandlora_target_modules``
        with ``QRandLoRALinear``.  Only runs if
        ``self._qrandlora_r > 0`` (disabled by default).

        Returns:
            Number of layers patched.
        """
        if self._qrandlora_r <= 0:
            return 0

        from core.qrrandlora import apply_qrandlora

        n = apply_qrandlora(
            self.backbone,
            r=self._qrandlora_r,
            alpha=self._qrandlora_alpha,
            sparsity=self._qrandlora_sparsity,
            num_components=self._qrandlora_num_components,
            target_modules=self._qrandlora_target_modules,
        )
        return n

    def set_pipeline(
        self,
        memory: Any = None,
        router: Any = None,
        lprm: Any = None,
        memory_top_k: int = 5,
        router_interval: int = 0,
    ) -> None:
        """Attach memory, router, and LPRM for the full generation pipeline.

        Creates a ``PipelineManager`` that wires memory retrieval and
        game-theoretic routing into ``generate()``.

        Args:
            memory: ``HierarchicalMemory`` instance.
            router: ``GameTheoreticRouter`` instance.
            lprm: ``MultiHeadLPRM`` instance (attached to router).
            memory_top_k: Number of memory nodes to retrieve.
            router_interval: Run auction every N steps (0 = once).
        """
        from core.pipeline import PipelineManager

        if lprm is not None and router is not None:
            router.lprm = lprm

        self._memory = memory
        self._router = router

        self._pipeline = PipelineManager(
            model=self,
            memory=memory,
            router=router,
            memory_top_k=memory_top_k,
            router_interval=router_interval,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_branches: bool = False,
        return_states: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Forward pass through the patched TRM-Bank backbone.

        The backbone's mixers have been replaced with
        ``Mamba3MixerAdapter`` (ComplexMIMOMamba3), so the SSM
        replacement runs as part of the normal forward pass --
        no separate enrichment needed.

        Args:
            input_ids: Token indices of shape (B, L).
            attention_mask: Optional mask of shape (B, L).
            return_branches: Not supported in this mode (always False).

        Returns:
            Dictionary with keys:
                logits: Output logits of shape (B, L, V).
                hidden_states: Hidden states from each layer.

        Raises:
            RuntimeError: If build() has not been called.
        """
        if self.backbone is None:
            raise RuntimeError("Call .build() before forward().")

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = outputs.logits
        # Guard against NaN/Inf from Mamba slow_forward numerical instability.
        if logits is not None and (logits.isnan().any() or logits.isinf().any()):
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        result: Dict[str, Any] = {
            "logits": logits,
            "hidden_states": outputs.hidden_states,
        }
        if return_states:
            result["states"] = None
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_pipeline: bool = True,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with custom SSM state carry.

        When a ``PipelineManager`` is attached via ``set_pipeline()``
        and ``use_pipeline=True``, the full pipeline runs: memory
        retrieval at prefill, game-theoretic router auction at each
        decode step, and dispatch to the winning compute provider.

        When no pipeline is attached, falls back to the custom SSM
        autoregressive loop (``ComplexMIMOMamba3`` recurrence with
        RoPE + trapezoidal integration at every step).  Never calls
        ``backbone.generate()``.

        Args:
            input_ids: Prompt of shape (B, L).
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k filter threshold.
            top_p: Nucleus sampling threshold.
            use_pipeline: Whether to use the memory+router pipeline
                          (ignored if no pipeline is configured).

        Returns:
            Generated token ids of shape (B, L + max_new_tokens).
        """
        if self.backbone is None:
            raise RuntimeError("Call .build() before generate().")

        # -- Pipeline path --
        if use_pipeline and self._pipeline is not None:
            return self._pipeline.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # -- SSM-only autoregressive path (no pipeline) --
        B: int = input_ids.shape[0]
        device: torch.device = input_ids.device

        outputs = self.backbone(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        logits: torch.Tensor = outputs.logits
        cache_params: Any = outputs.cache_params
        seq: torch.Tensor = input_ids

        for _ in range(max_new_tokens):
            logit: torch.Tensor = logits[:, -1, :]

            if temperature == 0.0:
                next_id: torch.Tensor = logit.argmax(dim=-1, keepdim=True)
            else:
                if 0 < top_k < logit.size(-1):
                    vals, _ = logit.topk(top_k, dim=-1)
                    logit[logit < vals[:, -1:]] = float("-inf")
                if top_p < 1.0:
                    sorted_l, sorted_idx = logit.sort(dim=-1, descending=True)
                    cum_probs = sorted_l.softmax(dim=-1).cumsum(dim=-1)
                    sorted_l[cum_probs > top_p] = float("-inf")
                    logit = sorted_l.scatter(-1, sorted_idx, sorted_l)
                probs = torch.softmax(
                    logit / max(temperature, 1e-8), dim=-1
                )
                next_id = torch.multinomial(probs, 1)

            seq = torch.cat([seq, next_id], dim=-1)

            outputs = self.backbone(
                input_ids=next_id,
                cache_params=cache_params,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits
            cache_params = outputs.cache_params

        return seq

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"mimo_rank={self.mimo_rank}, "
            f"4bit={self.use_4bit}"
        )
