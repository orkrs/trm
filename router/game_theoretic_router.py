from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from config import CONFIG
from router.lprm import MultiHeadLPRM
from router.essm import (
    ComputeProvider,
    DirectGenerationProvider,
    MIMOBrancherProvider,
    ORToolsProvider,
    PythonSandboxProvider,
    SymPyProvider,
)


@dataclass
class Bid:
    """A bid submitted by a compute provider in the reverse auction.

    Attributes:
        provider_name: Name of the bidding module.
        quality: Estimated success probability q_i in [0, 1].
        cost: Requested computational budget c_i (abstract FLOP units).
        utility: Computed net utility V_i = q_i - c_i.
    """
    provider_name: str
    quality: float
    cost: float
    utility: float


@dataclass
class AuctionResult:
    """Result of a VCG reverse second-price auction round.

    Attributes:
        winner_name: Name of the winning provider.
        winner_bid: The Bid object of the winner.
        payment: Amount the winner must pay (second-highest utility).
        all_bids: List of all submitted bids.
        social_welfare: Total social welfare of the allocation.
    """
    winner_name: str
    winner_bid: Bid
    payment: float
    all_bids: List[Bid]
    social_welfare: float


class GameTheoreticRouter:
    """VCG reverse second-price auction router for cognitive modules.

    Formalizes computation allocation as a market mechanism.
    Each module (provider) bids with (q_i, c_i).  The router selects
    the provider maximizing utility V_i = q_i - c_i.  Payment follows
    the Vickrey-Clarke-Groves mechanism: the winner pays the
    opportunity cost (second-highest utility).

    Reference: TRM-Bank v3.0 Section 4 (Theoretical-Game Routing).

    Args:
        providers: List of registered ComputeProvider instances.
        lprm: Multi-head LPRM for quality estimation.
        flops_budget: Global FLOPs budget limit.
    """

    def __init__(
        self,
        providers: Optional[List[ComputeProvider]] = None,
        lprm: Optional[MultiHeadLPRM] = None,
        flops_budget: float = 100.0,
    ) -> None:
        self.flops_budget: float = flops_budget

        if providers is not None:
            self.providers: List[ComputeProvider] = providers
        else:
            self.providers = self._default_providers()

        self.lprm: Optional[MultiHeadLPRM] = lprm
        self._provider_map: Dict[str, ComputeProvider] = {
            p.name: p for p in self.providers
        }

    @staticmethod
    def _default_providers() -> List[ComputeProvider]:
        """Create the standard set of cognitive compute providers."""
        return [
            DirectGenerationProvider(cost=1.0),
            PythonSandboxProvider(cost=10.0),
            ORToolsProvider(cost=15.0),
            SymPyProvider(cost=20.0),
            MIMOBrancherProvider(cost=30.0),
        ]

    def _compute_utility(self, quality: float, cost: float) -> float:
        """Compute net utility V_i = q_i - normalized_cost.

        The cost is normalized by the global FLOPs budget so that
        both quantities are in approximately the same range.

        Args:
            quality: Predicted success probability q_i in [0, 1].
            cost: Computational cost c_i.

        Returns:
            Utility V_i.
        """
        norm_cost = cost / self.flops_budget
        return quality - norm_cost

    def collect_bids(
        self,
        query: str,
        hidden_state: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Bid]:
        """Collect utility bids from all registered providers.

        Args:
            query: The input query string.
            hidden_state: Latent state tensor of shape (B, D).
            context: Optional context dict for quality estimation.

        Returns:
            List of Bids sorted by utility descending.
        """
        bids: List[Bid] = []

        for idx, provider in enumerate(self.providers):
            q_i: float = 0.5  # default fallback

            if self.lprm is not None and idx < len(self.lprm.heads):
                with torch.no_grad():
                    q_tensor = self.lprm(hidden_state, module_idx=idx)
                    q_i = float(q_tensor.mean().item())
            else:
                q_i = provider.estimate_quality(query, context)

            c_i = provider.cost
            v_i = self._compute_utility(q_i, c_i)

            bid = Bid(
                provider_name=provider.name,
                quality=q_i,
                cost=c_i,
                utility=v_i,
            )
            bids.append(bid)

        bids.sort(key=lambda b: b.utility, reverse=True)
        return bids

    def run_auction(
        self,
        query: str,
        hidden_state: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuctionResult:
        """Run a full VCG reverse second-price auction round.

        Args:
            query: The input query string.
            hidden_state: Latent state tensor of shape (B, D).
            context: Optional context dict.

        Returns:
            AuctionResult containing the winner, payment, and all bids.

        Raises:
            RuntimeError: If no bids can be collected.
        """
        bids = self.collect_bids(query, hidden_state, context)

        if not bids:
            raise RuntimeError("No bids collected; no providers registered.")

        winner = bids[0]
        second_best = bids[1] if len(bids) > 1 else bids[0]

        # VCG payment: winner pays the opportunity cost (second-highest utility).
        # In a reverse auction, the payment is the second-best utility value.
        payment = max(0.0, second_best.utility)

        # Social welfare: sum of utilities of all participants.
        social_welfare = sum(b.utility for b in bids)

        logger.info(
            f"Auction winner: {winner.provider_name} "
            f"(q={winner.quality:.3f}, c={winner.cost:.1f}, "
            f"V={winner.utility:.3f}, payment={payment:.3f})"
        )

        return AuctionResult(
            winner_name=winner.provider_name,
            winner_bid=winner,
            payment=payment,
            all_bids=bids,
            social_welfare=social_welfare,
        )

    def dispatch(
        self,
        query: str,
        hidden_state: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any, AuctionResult]:
        """Run the auction and dispatch the query to the winning provider.

        Args:
            query: The input query string.
            hidden_state: Latent state tensor.
            context: Optional context for execution.

        Returns:
            Tuple of (winner_name, execution_result, auction_result).
        """
        result = self.run_auction(query, hidden_state, context)
        provider = self._provider_map[result.winner_name]
        execution_result = provider.execute(query, context)
        return result.winner_name, execution_result, result

    def extra_repr(self) -> str:
        return (
            f"num_providers={len(self.providers)}, "
            f"flops_budget={self.flops_budget}"
        )
