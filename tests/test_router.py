"""Unit tests for Game-Theoretic Router, LPRM, and ESSM modules."""

from __future__ import annotations

import torch
import pytest

from router.lprm import LPRM, MultiHeadLPRM
from router.essm import (
    PythonSandboxProvider,
    SymPyProvider,
    ORToolsProvider,
    DirectGenerationProvider,
    MIMOBrancherProvider,
)
from router.game_theoretic_router import (
    GameTheoreticRouter,
    Bid,
    AuctionResult,
)


class TestLPRM:
    """Verification suite for Latent Process Reward Model."""

    def test_lprm_forward_shape(self) -> None:
        """LPRM produces (B, 1) for (B, D) input."""
        model = LPRM(hidden_dim=64, num_heads=16)
        x = torch.randn(4, 64)
        q = model(x)
        assert q.shape == (4, 1), f"Expected (4, 1), got {q.shape}"

    def test_lprm_output_range(self) -> None:
        """LPRM outputs are in [0, 1] (sigmoid-bounded)."""
        model = LPRM(hidden_dim=64, num_heads=16)
        x = torch.randn(10, 64)
        q = model(x)
        assert q.min() >= 0.0 and q.max() <= 1.0, f"Range: [{q.min()}, {q.max()}]"

    def test_lprm_backward(self) -> None:
        """LPRM supports gradient flow."""
        model = LPRM(hidden_dim=64, num_heads=16)
        x = torch.randn(2, 64, requires_grad=True)
        q = model(x)
        loss = q.sum()
        loss.backward()
        assert x.grad is not None

    def test_multihead_lprm_shape(self) -> None:
        """MultiHeadLPRM with M modules returns (B, M)."""
        model = MultiHeadLPRM(
            module_names=["a", "b", "c"],
            hidden_dim=64,
            num_heads=16,
        )
        x = torch.randn(4, 64)
        q = model(x)
        assert q.shape == (4, 3), f"Expected (4, 3), got {q.shape}"

    def test_multihead_lprm_batched(self) -> None:
        """MultiHeadLPRM with batched (B, L, D) input returns (B, L, M)."""
        model = MultiHeadLPRM(
            module_names=["a", "b", "c"],
            hidden_dim=64,
            num_heads=16,
        )
        x = torch.randn(2, 8, 64)
        q = model(x)
        assert q.shape == (2, 8, 3), f"Expected (2, 8, 3), got {q.shape}"
        # Verify single-module path still works with batched input.
        q1 = model(x, module_idx=0)
        assert q1.shape == (2, 8, 1), f"Expected (2, 8, 1), got {q1.shape}"

    def test_multihead_single_module(self) -> None:
        """MultiHeadLPRM with module_idx returns (B, 1)."""
        model = MultiHeadLPRM(
            module_names=["a", "b"],
            hidden_dim=64,
            num_heads=16,
        )
        x = torch.randn(4, 64)
        q = model(x, module_idx=1)
        assert q.shape == (4, 1), f"Expected (4, 1), got {q.shape}"


class TestProviders:
    """Verification suite for ESSM compute providers."""

    def test_python_sandbox_valid_code(self) -> None:
        """PythonSandbox returns result for valid code."""
        provider = PythonSandboxProvider()
        code = "result = 2 + 2"
        output = provider.execute(code)
        assert output == 4, f"Expected 4, got {output}"

    def test_python_sandbox_syntax_error(self) -> None:
        """PythonSandbox returns error dict for invalid syntax."""
        provider = PythonSandboxProvider()
        code = "result = 2 @invalid_syntax_here 2"
        output = provider.execute(code)
        assert isinstance(output, dict) and "error" in output

    def test_python_sandbox_quality_estimate(self) -> None:
        """PythonSandbox quality heuristic returns [0, 1]."""
        provider = PythonSandboxProvider()
        q = provider.estimate_quality("def f(): return 42")
        assert 0.0 <= q <= 1.0

    def test_direct_generation_quality_range(self) -> None:
        """DirectGeneration quality is in [0, 1]."""
        provider = DirectGenerationProvider()
        q = provider.estimate_quality("What is 2+2?")
        assert 0.0 <= q <= 1.0

    def test_mimo_brancher_quality_range(self) -> None:
        """MIMOBrancher quality is in [0, 1]."""
        provider = MIMOBrancherProvider()
        q = provider.estimate_quality("Solve this complex equation:")
        assert 0.0 <= q <= 1.0

    def test_sympy_quality_range(self) -> None:
        """SymPy quality is in [0, 1]."""
        provider = SymPyProvider()
        q = provider.estimate_quality("solve x**2 - 4")
        assert 0.0 <= q <= 1.0

    def test_ortools_quality_range(self) -> None:
        """OR-Tools quality is in [0, 1]."""
        provider = ORToolsProvider()
        q = provider.estimate_quality("NewIntVar(0, 10, 'x')")
        assert 0.0 <= q <= 1.0


class TestGameTheoreticRouter:
    """Verification suite for the VCG auction router."""

    def test_collect_bids_returns_sorted(self) -> None:
        """Bids are sorted by utility descending."""
        router = GameTheoreticRouter()
        hidden = torch.randn(1, 8)
        bids = router.collect_bids("test query", hidden)
        assert len(bids) > 0
        utilities = [b.utility for b in bids]
        assert all(utilities[i] >= utilities[i + 1] for i in range(len(utilities) - 1))

    def test_run_auction_returns_winner(self) -> None:
        """Auction identifies a valid winner."""
        router = GameTheoreticRouter()
        hidden = torch.randn(1, 8)
        result = router.run_auction("test", hidden)
        assert isinstance(result, AuctionResult)
        assert isinstance(result.winner_name, str)
        assert len(result.winner_name) > 0

    def test_auction_payment_non_negative(self) -> None:
        """VCG payment is non-negative."""
        router = GameTheoreticRouter()
        hidden = torch.randn(1, 8)
        result = router.run_auction("test", hidden)
        assert result.payment >= 0.0

    def test_auction_social_welfare(self) -> None:
        """Social welfare is sum of all utilities."""
        router = GameTheoreticRouter()
        hidden = torch.randn(1, 8)
        result = router.run_auction("test", hidden)
        expected_sw = sum(b.utility for b in result.all_bids)
        assert abs(result.social_welfare - expected_sw) < 1e-6

    def test_dispatch_returns_tuple(self) -> None:
        """dispatch() returns (name, result, auction_result)."""
        router = GameTheoreticRouter()
        hidden = torch.randn(1, 8)
        name, exec_result, auction = router.dispatch("test query", hidden)
        assert isinstance(name, str)
        assert auction.winner_name == name

    def test_auction_with_lprm(self) -> None:
        """Router works with MultiHeadLPRM attached."""
        lprm = MultiHeadLPRM(
            module_names=["direct_generation", "python_sandbox"],
            hidden_dim=64,
            num_heads=16,
        )
        providers = [DirectGenerationProvider(), PythonSandboxProvider()]
        router = GameTheoreticRouter(providers=providers, lprm=lprm)
        hidden = torch.randn(1, 64)
        result = router.run_auction("test", hidden)
        assert result.winner_name in ("direct_generation", "python_sandbox")

    def test_bid_dataclass(self) -> None:
        """Bid dataclass stores attributes correctly."""
        bid = Bid(provider_name="test", quality=0.8, cost=5.0, utility=0.3)
        assert bid.provider_name == "test"
        assert bid.quality == 0.8
        assert bid.cost == 5.0
        assert abs(bid.utility - 0.3) < 1e-6
