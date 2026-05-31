from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from memory.experience import ExperienceReplay, TurnExperience
from memory.hierarchical_memory import HierarchicalMemory, RetrievalResult
from router.game_theoretic_router import (
    AuctionResult,
    Bid,
    GameTheoreticRouter,
)
from router.lprm import MultiHeadLPRM


class PipelineManager:
    """Orchestrates Memory -> Router -> LPRM -> Model for token generation.

    Wires together the three standalone subsystems:
        1. **HierarchicalMemory** -- retrieves relevant context from the
           4-tier memory graph before generation.
        2. **GameTheoreticRouter** -- runs a VCG reverse auction at each
           decode step to decide which compute provider to use (SSM
           backbone, Python sandbox, SymPy, OR-Tools, or MIMO brancher).
        3. **MultiHeadLPRM** -- predicts module-specific success
           probabilities from the latent hidden state, providing
           quality estimates q_i for the router.

    The manager wraps a patched model (TinyTRMModel or TRMBankModel)
    and overrides its ``generate()`` method with the full pipeline.

    Args:
        model: The patched language model (must have ``forward()``
               returning ``{"logits": ..., "states": ...}``).
        memory: Initialized HierarchicalMemory instance.
        router: Initialized GameTheoreticRouter with LPRM attached.
        memory_top_k: Number of memory nodes to retrieve.
        router_interval: Run the auction every N decode steps
                         (0 = only at prefill).
    """

    def __init__(
        self,
        model: nn.Module,
        memory: Optional[HierarchicalMemory] = None,
        router: Optional[GameTheoreticRouter] = None,
        memory_top_k: int = 5,
        router_interval: int = 0,
    ) -> None:
        self.model: nn.Module = model
        self.memory: Optional[HierarchicalMemory] = memory
        self.router: Optional[GameTheoreticRouter] = router
        self.memory_top_k: int = memory_top_k
        self.router_interval: int = router_interval

        self._last_retrieval: Optional[RetrievalResult] = None
        self._last_auction: Optional[AuctionResult] = None

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_for_memory(self, input_ids: torch.Tensor) -> np.ndarray:
        """Encode the prompt into a flat embedding for memory retrieval.

        Uses the model's pre-logit hidden state mean-pooled over the
        sequence length.  Falls back to mean-pooled logits if hidden
        states are not available.

        Args:
            input_ids: Prompt of shape (B, L).

        Returns:
            Numpy embedding vector of shape (embedding_dim,).
        """
        out = self.model(input_ids)
        if out is None:
            return np.zeros(256, dtype=np.float32)

        if isinstance(out, dict):
            # Try hidden_states first (pre-logit activations)
            hs: Any = out.get("hidden_states", None)
            if hs is not None:
                if isinstance(hs, (tuple, list)):
                    hs = hs[-1]  # last layer
                pooled: torch.Tensor = hs.mean(dim=1)
                # pooled: (B, D)
            else:
                logits = out.get("logits", None)
                if logits is None:
                    return np.zeros(256, dtype=np.float32)
                pooled = logits.mean(dim=1)
                # pooled: (B, V)
        else:
            pooled = out.mean(dim=1) if isinstance(out, torch.Tensor) else torch.zeros(256)

        return pooled[0].cpu().numpy().flatten().astype(np.float32)

    def retrieve_memory(
        self, query_emb: np.ndarray, top_k: Optional[int] = None
    ) -> Optional[RetrievalResult]:
        """Query the hierarchical memory.

        Args:
            query_emb: Query embedding of shape (D,).
            top_k: Override default top_k.

        Returns:
            RetrievalResult or None if memory is not configured.
        """
        if self.memory is None:
            return None
        k = top_k if top_k is not None else self.memory_top_k
        result: RetrievalResult = self.memory.retrieve(query_emb, top_k=k)
        self._last_retrieval = result
        return result

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def run_auction(
        self,
        query_text: str,
        hidden_state: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AuctionResult]:
        """Run the game-theoretic router auction.

        Args:
            query_text: Decoded query string.
            hidden_state: Latent state of shape (B, D).
            context: Optional context dict (e.g., with memory results).

        Returns:
            AuctionResult or None if router is not configured.
        """
        if self.router is None:
            return None
        result: AuctionResult = self.router.run_auction(
            query_text, hidden_state, context=context
        )
        self._last_auction = result
        return result

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        retrieve_memory: bool = True,
        use_router: bool = True,
        record_experience: bool = True,
    ) -> torch.Tensor:
        """Generate tokens with the full pipeline.

        Pipeline:
            1. Run the model prefill forward pass.
            2. Retrieve relevant context from hierarchical memory.
            3. For each decode step (or at ``router_interval`` intervals):
               a. Collect LPRM-based quality bids from all providers.
               b. Run the VCG auction to select the winning module.
               c. Dispatch the generation step to the winner.
            4. Optionally record turn experience for offline training.
            5. Return the full generated sequence.

        Args:
            input_ids: Prompt of shape (B, L).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filter.
            top_p: Nucleus sampling threshold.
            retrieve_memory: Whether to query memory before generation.
            use_router: Whether to run the router at decode steps.
            record_experience: Whether to store the turn experience in
                               memory for offline LPRM training.

        Returns:
            Generated token ids of shape (B, L + max_new_tokens).
        """
        B: int = input_ids.shape[0]
        device: torch.device = input_ids.device
        seq: torch.Tensor = input_ids

        # ---- 1. Prefill ----
        try:
            prefill_out = self.model(input_ids, return_states=True)
        except TypeError:
            prefill_out = self.model(input_ids)

        logits: torch.Tensor = prefill_out.get("logits", prefill_out) if isinstance(prefill_out, dict) else prefill_out
        states: Any = prefill_out.get("states", None) if isinstance(prefill_out, dict) else None
        step_out: Any = prefill_out

        # ---- 2. Memory retrieval ----
        memory_context: Optional[Dict[str, Any]] = None
        if retrieve_memory and self.memory is not None:
            query_emb: np.ndarray = self.encode_for_memory(input_ids)
            mem_result: Optional[RetrievalResult] = self.retrieve_memory(query_emb)
            if mem_result is not None and mem_result.turn_ids:
                memory_context = {
                    "turn_ids": mem_result.turn_ids,
                    "scores": mem_result.scores,
                    "paths": mem_result.paths,
                }

        # ---- 3. Decode loop ----
        step_module_map: Dict[int, str] = {}
        _use_ssm: bool = True

        for step_idx in range(max_new_tokens):
            logit: torch.Tensor = logits[:, -1, :]

            # Sample next token
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

            # ---- 4. Router dispatch (if enabled and interval hit) ----
            _use_ssm = True
            if use_router and self.router is not None:
                if self.router_interval == 0 or step_idx % self.router_interval == 0:
                    last_hidden: torch.Tensor = self._extract_hidden_for_router(
                        logits, step_out
                    )
                    auction_ctx: Dict[str, Any] = {}
                    if memory_context is not None:
                        auction_ctx["memory"] = memory_context
                    auction_result: Optional[AuctionResult] = self.run_auction(
                        query_text="",
                        hidden_state=last_hidden,
                        context=auction_ctx,
                    )
                    if auction_result is not None:
                        winner: str = auction_result.winner_name
                        step_module_map[step_idx] = winner
                        _use_ssm = winner in (
                            "direct_generation",
                            "mimo_brancher",
                        )

            # ---- 5. Model step ----
            if _use_ssm:
                if hasattr(self.model, "forward") and states is not None:
                    try:
                        step_out = self.model(next_id, states=states, return_states=True)
                        logits = step_out.get("logits", step_out) if isinstance(step_out, dict) else step_out
                        states = step_out.get("states", None) if isinstance(step_out, dict) else None
                    except (TypeError, NotImplementedError):
                        step_out = self.model(next_id)
                        logits = step_out.get("logits", step_out) if isinstance(step_out, dict) else step_out
                else:
                    step_out = self.model(next_id)
                    logits = step_out.get("logits", step_out) if isinstance(step_out, dict) else step_out
            else:
                logits = torch.zeros(
                    B, 1, self._vocab_size(logits), device=device
                )

        # ---- 6. Experience recording ----
        if record_experience and self.memory is not None:
            self._record_turn_experience(
                input_ids=input_ids,
                seq=seq,
                logits=logits,
                step_out=step_out,
                step_module_map=step_module_map,
                use_router=use_router,
            )

        return seq

    def _record_turn_experience(
        self,
        input_ids: torch.Tensor,
        seq: torch.Tensor,
        logits: torch.Tensor,
        step_out: Any,
        step_module_map: Dict[int, str],
        use_router: bool,
    ) -> None:
        """Build a TurnExperience and push it to memory."""
        prompt_emb: np.ndarray = self.encode_for_memory(input_ids)

        last_hidden: torch.Tensor = self._extract_hidden_for_router(logits, step_out)

        generated_ids: torch.Tensor = seq[:, input_ids.shape[-1]:]
        if generated_ids.numel() > 0:
            probs = F.softmax(logits[:, -1:, :], dim=-1)
            log_probs = torch.log(probs.gather(-1, generated_ids[:, -1:].unsqueeze(-1)) + 1e-10)
            reward: float = float(log_probs.mean().item())
        else:
            reward = 0.0

        module_name: str = "direct_generation"
        if use_router and self.router is not None and step_module_map:
            last_step: int = max(step_module_map.keys())
            module_name = step_module_map.get(last_step, "direct_generation")

        exp = TurnExperience(
            prompt_embedding=prompt_emb,
            hidden_for_lprm=last_hidden.detach().cpu(),
            module_name=module_name,
            reward=reward,
            prompt_ids=input_ids.cpu(),
            generated_ids=generated_ids.cpu(),
            step_module_map=step_module_map,
        )
        self.memory.record_experience(exp)

    def _extract_hidden_for_router(
        self,
        logits: torch.Tensor,
        out: Any,
    ) -> torch.Tensor:
        """Extract a flat hidden state vector for the router.

        Tries ``hidden_states`` from the model output (last layer, last
        token), falls back to the last-token logits.
        """
        if isinstance(out, dict):
            hs = out.get("hidden_states", None)
            if hs is not None:
                if isinstance(hs, (tuple, list)):
                    hs = hs[-1]
                return hs[:, -1, :]
        return logits[:, -1, :].float()

    @staticmethod
    def _vocab_size(logits: torch.Tensor) -> int:
        return logits.shape[-1]

    @property
    def last_retrieval(self) -> Optional[RetrievalResult]:
        return self._last_retrieval

    @property
    def last_auction(self) -> Optional[AuctionResult]:
        return self._last_auction
