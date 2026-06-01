from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.mamba.modeling_mamba import MambaBlock, MambaConfig

from core.complex_mimo_mamba import ComplexMIMOMamba3


class Mamba3MixerAdapter(nn.Module):
    """Drop-in replacement for ``MambaMixer`` that uses ``ComplexMIMOMamba3``.

    Preserves the conv1d, in_proj / out_proj interface of the original
    MambaMixer so the parent ``MambaBlock`` continues to work.  The SSM
    recurrence (the core of the mixer) is delegated to
    ``ComplexMIMOMamba3`` which provides RoPE rotation, trapezoidal
    discretization, and multi-branch MIMO output.

    Args:
        config: HuggingFace ``MambaConfig``.
        layer_idx: Index of this layer in the model.
        d_state: SSM state dimension (must be even).
        headdim: Head dimension for per-head RoPE.
        mimo_rank: Number of parallel MIMO branches.
        device: Target device.
        dtype: Target dtype.
    """

    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        d_state: int = 64,
        headdim: int = 64,
        mimo_rank: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        orig_mixer: Optional[nn.Module] = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.config: MambaConfig = config
        self.layer_idx: int = layer_idx
        self.hidden_size: int = config.hidden_size
        self.ssm_state_size: int = d_state
        self.intermediate_size: int = config.intermediate_size
        self.conv_kernel_size: int = config.conv_kernel
        self.use_conv_bias: bool = config.use_conv_bias
        self.activation: str = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        factory = {"device": device, "dtype": dtype}

        # ---- Preservation of Mamba conv1d + projections ----
        if orig_mixer is not None:
            self.conv1d = orig_mixer.conv1d
            self.in_proj = orig_mixer.in_proj
            self.out_proj = orig_mixer.out_proj
        else:
            self.conv1d = nn.Conv1d(
                in_channels=self.intermediate_size,
                out_channels=self.intermediate_size,
                bias=config.use_conv_bias,
                kernel_size=config.conv_kernel,
                groups=self.intermediate_size,
                padding=config.conv_kernel - 1,
                **factory,
            )
            self.in_proj = nn.Linear(
                self.hidden_size, self.intermediate_size * 2, bias=config.use_bias, **factory
            )
            self.out_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias=config.use_bias, **factory
            )

        # Our custom SSM core -- replaces the selective scan
        self.ssm = ComplexMIMOMamba3(
            d_model=self.intermediate_size,
            d_state=d_state,
            headdim=headdim,
            mimo_rank=mimo_rank,
            device=device,
            dtype=dtype,
            precomputed_projections=True,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through conv1d, then ComplexMIMOMamba3 SSM.

        Args:
            hidden_states: Post-norm input of shape (B, L, hidden_size).
            cache_params: Optional Cache for conv / SSM state.
            attention_mask: Optional mask of shape (B, L).

        Returns:
            Output of shape (B, L, hidden_size).
        """
        B, L, D = hidden_states.shape

        # 1. in_proj -> split -> [hidden_states, gate]
        proj = self.in_proj(hidden_states).transpose(1, 2)
        # proj: (B, 2 * intermediate_size, L)
        hidden_states, gate = proj.chunk(2, dim=1)
        # hidden_states: (B, intermediate_size, L)
        # gate: (B, intermediate_size, L)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Conv1d sequence transformation (with cache for decode)
        if cache_params is not None:
            if cache_params.has_previous_state(self.layer_idx):
                # Decode: use cached conv state for single-step conv1d
                conv_state = cache_params.update_conv_state(hidden_states, self.layer_idx)
                conv_state = conv_state.to(self.conv1d.weight.device)
                hidden_states = torch.sum(
                    conv_state * self.conv1d.weight[:, 0, :], dim=-1
                )
                if self.use_conv_bias:
                    hidden_states = hidden_states + self.conv1d.bias
                hidden_states = self.act(hidden_states).unsqueeze(-1)
            else:
                # First call with cache: init conv state, then run conv1d normally
                if L <= self.conv_kernel_size:
                    conv_state = F.pad(hidden_states, (self.conv_kernel_size - L, 0))
                else:
                    conv_state = hidden_states[..., -self.conv_kernel_size:]
                cache_params.update_conv_state(conv_state, self.layer_idx)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :L])
        else:
            hidden_states = self.act(self.conv1d(hidden_states)[..., :L])
            # hidden_states: (B, intermediate_size, L)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. SSM: ComplexMIMOMamba3
        # hidden_states: (B, intermediate_size, L) -> (B, L, intermediate_size)
        ssm_in = hidden_states.transpose(1, 2).contiguous()
        gate_t = gate.transpose(1, 2).contiguous()

        state: Optional[torch.Tensor] = None
        if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
            state = cache_params.layers[self.layer_idx].recurrent_states
            # state: (B, intermediate_size, d_state) matches ComplexMIMOMamba3 format

        ssm_out, next_state, _ = self.ssm(ssm_in, state=state, gate=gate_t)
        # ssm_out: (B, L, intermediate_size)

        # 4. Gate
        # ssm_out is gated using gate_t
        ssm_out = ssm_out * F.silu(gate_t)

        # 5. Output projection
        out = self.out_proj(ssm_out)
        # out: (B, L, hidden_size)

        if cache_params is not None:
            cache_params.update_recurrent_state(next_state, self.layer_idx)

        return out

    def extra_repr(self) -> str:
        return (
            f"layer_idx={self.layer_idx}, hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"conv_kernel={self.conv_kernel_size}"
        )


def patch_mamba2_with_mamba3(
    model: nn.Module,
    d_state: int = 64,
    headdim: int = 64,
    mimo_rank: int = 2,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    gradient_checkpointing: bool = False,
) -> int:
    """Replace every ``MambaBlock.mixer`` with ``Mamba3MixerAdapter``.

    Iterates over the named modules of ``model``, finds all
    ``MambaBlock`` instances, and replaces their ``.mixer`` attribute
    with a ``Mamba3MixerAdapter``.  Weights for the conv1d, in_proj,
    and out_proj are copied from the original mixer.  The
    ``ComplexMIMOMamba3`` inside the adapter is freshly initialised.

    Args:
        model: A HuggingFace ``MambaForCausalLM`` or any module that
               contains ``MambaBlock`` submodules.
        d_state: SSM state dimension for ComplexMIMOMamba3.
        headdim: Head dimension for ComplexMIMOMamba3.
        mimo_rank: Number of parallel MIMO branches.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Number of mixers patched.

    Raises:
        RuntimeError: If a ``MambaBlock`` is found but its ``.mixer``
            is missing or of an unexpected type.
    """
    count: int = 0

    for _name, child in model.named_modules():
        if not isinstance(child, MambaBlock):
            continue

        if not hasattr(child, "mixer"):
            raise RuntimeError(f"{type(child).__name__} at '{_name}' has no .mixer.")

        orig_mixer = child.mixer

        # Checkerboard hybrid optimization for Tesla T4 OOM protection
        # (patch only even layers: 24 out of 48)
        if getattr(orig_mixer, "layer_idx", 0) % 2 != 0:
            continue

        config: MambaConfig = orig_mixer.config
        layer_idx: int = orig_mixer.layer_idx

        adapter = Mamba3MixerAdapter(
            config, layer_idx,
            d_state=d_state, headdim=headdim, mimo_rank=mimo_rank,
            device=device, dtype=dtype,
            orig_mixer=orig_mixer,
            gradient_checkpointing=gradient_checkpointing,
        )

        child.mixer = adapter
        count += 1

    return count
