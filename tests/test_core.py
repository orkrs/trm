"""Unit tests for QRandLoRA and ComplexMIMOScan modules."""

from __future__ import annotations

import torch
import pytest

from core.qrrandlora import QRandLoRALayer, QRandLoRALinear
from core.complex_mimo_mamba import (
    ComplexMIMOScan,
    TinyTRMBlock,
    TinyTRMModel,
    _rotate_2d_pairs,
)


class TestQRandLoRA:
    """Verification suite for QRandLoRA layer."""

    def test_qrandlora_forward_shape(self) -> None:
        """Output shape matches (B, D_out) for (B, D_in) input."""
        layer = QRandLoRALayer(
            in_features=64,
            out_features=128,
            num_components=4,
            lora_dim=16,
            sparsity=0.2,
        )
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"

    def test_qrandlora_linear_forward_shape(self) -> None:
        """QRandLoRALinear output shape matches (B, D_out)."""
        base_w = torch.randn(128, 64)
        base_b = torch.randn(128)
        linear = QRandLoRALinear(base_w, base_b, num_components=4, lora_dim=16)
        x = torch.randn(2, 64)
        out = linear(x)
        assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"

    def test_qrandlora_requires_grad(self) -> None:
        """Only Lambda and Gamma should be trainable; A, B are buffers."""
        layer = QRandLoRALayer(
            in_features=64,
            out_features=128,
            num_components=4,
            lora_dim=16,
            sparsity=0.2,
        )
        assert layer.Lambda.requires_grad
        assert layer.Gamma.requires_grad
        assert not layer.A_frozen.requires_grad
        assert not layer.B_frozen.requires_grad

    def test_qrandlora_rebuild(self) -> None:
        """rebuild() produces a matrix of shape (D_out, D_in)."""
        layer = QRandLoRALayer(
            in_features=64,
            out_features=128,
            num_components=4,
            lora_dim=16,
            sparsity=0.2,
        )
        delta_W = layer.rebuild()
        assert delta_W.shape == (128, 64), f"Expected (128, 64), got {delta_W.shape}"

    def test_qrandlora_ternary_entries(self) -> None:
        """A and B matrices contain only values from {-1, 0, 1}."""
        layer = QRandLoRALayer(
            in_features=32,
            out_features=32,
            num_components=2,
            lora_dim=8,
            sparsity=0.3,
        )
        for name, mat in [("A", layer.A_frozen), ("B", layer.B_frozen)]:
            vals = mat.unique().tolist()
            assert all(v in (-1, 0, 1) for v in vals), (
                f"{name} contains unexpected values: {vals}"
            )

    def test_qrandlora_no_inplace_ops(self) -> None:
        """Forward pass does not use in-place ops that break autograd."""
        layer = QRandLoRALinear(
            base_weight=torch.randn(64, 32),
            num_components=2,
            lora_dim=8,
            sparsity=0.3,
        )
        x = torch.randn(2, 32, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.qrandlora.Lambda.grad is not None
        assert layer.qrandlora.Gamma.grad is not None


class TestComplexMIMOScan:
    """Verification suite for ComplexMIMOScan."""

    def test_rotate_2d_pairs_shape(self) -> None:
        """_rotate_2d_pairs preserves shape (..., N)."""
        h = torch.randn(2, 4, 6)  # (B, D_in, N=6)
        cos_t = torch.cos(torch.randn(2, 4, 3))
        sin_t = torch.sin(torch.randn(2, 4, 3))
        out = _rotate_2d_pairs(h, cos_t, sin_t)
        assert out.shape == (2, 4, 6), f"Expected (2, 4, 6), got {out.shape}"

    def test_rotate_2d_pairs_raises_on_odd(self) -> None:
        """_rotate_2d_pairs raises ValueError for odd last dim."""
        h = torch.randn(2, 5)
        cos_t = torch.randn(2, 2)
        sin_t = torch.randn(2, 2)
        with pytest.raises(ValueError, match="even"):
            _rotate_2d_pairs(h, cos_t, sin_t)

    def test_scan_forward_shape(self) -> None:
        """ComplexMIMOScan returns (B, L, D, R) and (B, D, N)."""
        B, L, D_in, N, R = 2, 8, 16, 8, 4
        scan = ComplexMIMOScan(d_in=D_in, d_state=N, mimo_rank=R)
        x = torch.randn(B, L, D_in)
        y, final_state = scan(x)
        assert y.shape == (B, L, D_in, R), f"y shape: {y.shape}"
        assert final_state.shape == (B, D_in, N), f"state shape: {final_state.shape}"

    def test_scan_output_finite(self) -> None:
        """All outputs are finite (no NaN / Inf)."""
        B, L, D_in, N, R = 2, 16, 32, 16, 4
        scan = ComplexMIMOScan(d_in=D_in, d_state=N, mimo_rank=R)
        x = torch.randn(B, L, D_in)
        y, final_state = scan(x)
        assert torch.isfinite(y).all(), "y contains NaN or Inf"
        assert torch.isfinite(final_state).all(), "state contains NaN or Inf"

    def test_scan_with_initial_state(self) -> None:
        """Scan accepts and propagates an initial state."""
        B, L, D_in, N, R = 1, 4, 8, 4, 2
        scan = ComplexMIMOScan(d_in=D_in, d_state=N, mimo_rank=R)
        x = torch.randn(B, L, D_in)
        state = torch.randn(B, D_in, N)
        y, final_state = scan(x, state=state)
        assert torch.isfinite(y).all()
        assert final_state.shape == state.shape

    def test_scan_no_inplace_ops(self) -> None:
        """Forward pass supports backward (no in-place on autograd graph)."""
        B, L, D_in, N, R = 1, 4, 8, 4, 2
        scan = ComplexMIMOScan(d_in=D_in, d_state=N, mimo_rank=R)
        x = torch.randn(B, L, D_in, requires_grad=True)
        y, _ = scan(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_scan_mimo_rank_output(self) -> None:
        """Different MIMO ranks produce correct output dimensions."""
        D_in, N = 16, 8
        for R in [1, 2, 4, 8]:
            scan = ComplexMIMOScan(d_in=D_in, d_state=N, mimo_rank=R)
            x = torch.randn(1, 4, D_in)
            y, _ = scan(x)
            assert y.shape[-1] == R, f"Expected last dim {R}, got {y.shape[-1]}"


class TestTinyTRMBlock:
    """Verification suite for TinyTRMBlock."""

    def test_forward_shape(self) -> None:
        """TinyTRMBlock preserves (B, L, D) shape."""
        B, L, D, N, R = 2, 8, 32, 16, 4
        block = TinyTRMBlock(d_model=D, d_state=N, mimo_rank=R)
        x = torch.randn(B, L, D)
        y, state = block(x)
        assert y.shape == (B, L, D), f"y shape: {y.shape}"
        assert state.shape == (B, D, N), f"state shape: {state.shape}"

    def test_output_finite(self) -> None:
        """Output and state are finite."""
        block = TinyTRMBlock(d_model=16, d_state=8, mimo_rank=2)
        x = torch.randn(1, 4, 16)
        y, state = block(x)
        assert torch.isfinite(y).all()
        assert torch.isfinite(state).all()

    def test_backward(self) -> None:
        """Gradients flow through block."""
        block = TinyTRMBlock(d_model=16, d_state=8, mimo_rank=2)
        x = torch.randn(1, 4, 16, requires_grad=True)
        y, _ = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestTinyTRMModel:
    """Verification suite for TinyTRMModel (integration-level)."""

    def test_forward_shape(self) -> None:
        """TinyTRMModel returns (B, L, V) logits."""
        model = TinyTRMModel(
            vocab_size=100, d_model=32, d_state=16,
            mimo_rank=2, num_layers=2,
        )
        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        logits = out["logits"]
        assert logits.shape == (2, 8, 100), f"logits: {logits.shape}"

    def test_forward_with_states(self) -> None:
        """States can be passed between calls."""
        model = TinyTRMModel(
            vocab_size=50, d_model=16, d_state=8,
            mimo_rank=2, num_layers=2,
        )
        x = torch.randint(0, 50, (1, 8))
        out1 = model(x, return_states=True)
        states = out1["states"]
        assert len(states) == 2
        assert states[0].shape == (1, 16, 8)

        out2 = model(x, states=states)
        assert torch.isfinite(out2["logits"]).all()

    def test_generate_greedy(self) -> None:
        """Greedy generation returns longer sequence."""
        model = TinyTRMModel(
            vocab_size=50, d_model=16, d_state=8,
            mimo_rank=2, num_layers=2,
        )
        x = torch.randint(0, 50, (1, 4))
        gen = model.generate(x, max_new_tokens=5, temperature=0.0)
        assert gen.shape == (1, 9), f"gen: {gen.shape}"

    def test_generate_sampling(self) -> None:
        """Sampling generation works with temperature > 0."""
        model = TinyTRMModel(
            vocab_size=50, d_model=16, d_state=8,
            mimo_rank=2, num_layers=2,
        )
        x = torch.randint(0, 50, (1, 4))
        gen = model.generate(x, max_new_tokens=5, temperature=0.8)
        assert gen.shape == (1, 9)

    def test_backward(self) -> None:
        """Full forward + backward works."""
        model = TinyTRMModel(
            vocab_size=50, d_model=16, d_state=8,
            mimo_rank=2, num_layers=2,
        )
        x = torch.randint(0, 50, (2, 8))
        out = model(x)
        loss = out["logits"].sum()
        loss.backward()
        assert model.lm_head.weight.grad is not None
        assert model.embed.weight.grad is not None
