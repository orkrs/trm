"""Integration tests for the full TRM-Bank v3.0 pipeline.

Tests the end-to-end wiring of:
- TinyTRMModel (with ComplexMIMOMamba3 backbone)
- QRandLoRA adaptation
- HierarchicalMemory
- MultiHeadLPRM
- GameTheoreticRouter
- PipelineManager
- Experience replay
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import pytest

from core.complex_mimo_mamba import TinyTRMModel
from core.qrrandlora import apply_qrandlora
from core.pipeline import PipelineManager
from memory.hierarchical_memory import HierarchicalMemory
from router.game_theoretic_router import GameTheoreticRouter
from router.lprm import MultiHeadLPRM


@pytest.fixture
def tiny_model() -> TinyTRMModel:
    model = TinyTRMModel(
        vocab_size=100, d_model=64, d_state=16,
        mimo_rank=2, num_layers=2,
    )
    return model


@pytest.fixture
def model_with_lora(tiny_model: TinyTRMModel) -> TinyTRMModel:
    apply_qrandlora(
        tiny_model,
        r=8, alpha=4.0, sparsity=0.3,
        num_components=4,
        target_modules=("in_proj", "out_proj"),
    )
    return tiny_model


@pytest.fixture
def memory() -> HierarchicalMemory:
    mem = HierarchicalMemory(
        tier_names=("Domain", "Category", "Event", "Turn"),
        tier_slots=(4, 8, 16, 32),
        embedding_dim=64,
    )
    # Pre-populate a few memory nodes.
    for _ in range(5):
        mem.add_memory(
            tier=0,
            embedding=np.random.randn(64).astype(np.float32),
            summary="test turn",
        )
    mem.rebuild_all_indexes()
    return mem


@pytest.fixture
def lprm() -> MultiHeadLPRM:
    return MultiHeadLPRM(
        module_names=[
            "direct_generation", "python_sandbox",
            "sympy", "or_tools", "mimo_brancher",
        ],
        hidden_dim=64,
        num_heads=16,
    )


@pytest.fixture
def router(lprm: MultiHeadLPRM) -> GameTheoreticRouter:
    return GameTheoreticRouter(lprm=lprm, flops_budget=100.0)


class TestModelWithQRandLoRA:
    """Verify QRandLoRA is correctly applied."""

    def test_qrandlora_applied(self, model_with_lora: TinyTRMModel) -> None:
        """QRandLoRA wrappers exist on target modules."""
        qrandlora_count = 0
        for mod in model_with_lora.modules():
            if hasattr(mod, "qrandlora") and hasattr(mod.qrandlora, "Lambda"):
                qrandlora_count += 1
        assert qrandlora_count > 0, "No QRandLoRA wrappers found"

        # Verify frozen base weights.
        for name, param in model_with_lora.named_parameters():
            if "base_weight" in name:
                assert not param.requires_grad, (
                    f"base_weight {name} is not frozen"
                )

    def test_qrandlora_forward_shape(self, model_with_lora: TinyTRMModel) -> None:
        """Forward pass produces correct shape after QRandLoRA."""
        x = torch.randint(0, 100, (2, 8))
        out = model_with_lora(x)
        assert out["logits"].shape == (2, 8, 100), (
            f"Expected (2, 8, 100), got {out['logits'].shape}"
        )

    def test_qrandlora_gradients_flow(self, model_with_lora: TinyTRMModel) -> None:
        """Gradients flow through QRandLoRA params."""
        x = torch.randint(0, 100, (1, 4))
        logits = model_with_lora(x)["logits"]
        loss = logits.mean()
        loss.backward()

        grad_found = False
        for name, param in model_with_lora.named_parameters():
            if "Lambda" in name or "Gamma" in name:
                if param.grad is not None and param.grad.abs().sum().item() > 0:
                    grad_found = True
                    break
        assert grad_found, "No gradient found in QRandLoRA params"


class TestPipelineComponents:
    """Verify each pipeline component works in isolation."""

    def test_memory_retrieval(
        self, model_with_lora: TinyTRMModel, memory: HierarchicalMemory
    ) -> None:
        """Memory retrieves nodes using encoded prompt."""
        pm = PipelineManager(model=model_with_lora, memory=memory, router=None)
        x = torch.randint(0, 100, (1, 8))
        emb = pm.encode_for_memory(x)
        assert emb.shape == (64,), f"Expected (64,), got {emb.shape}"

        result = pm.retrieve_memory(emb, top_k=3)
        assert result is not None
        assert len(result.turn_ids) > 0, "No memory nodes retrieved"

    def test_router_auction(
        self, model_with_lora: TinyTRMModel,
        router: GameTheoreticRouter, lprm: MultiHeadLPRM,
    ) -> None:
        """Router auction produces a valid winner."""
        pm = PipelineManager(
            model=model_with_lora, memory=None, router=router,
        )
        x = torch.randint(0, 100, (1, 8))
        out = model_with_lora(x)
        hidden = out["hidden_states"][:, -1, :]  # (1, 64)

        result = pm.run_auction("", hidden)
        assert result is not None
        assert result.winner_name in [p.name for p in router.providers]
        assert result.payment >= 0.0

    def test_lprm_with_hidden_states(
        self, lprm: MultiHeadLPRM, model_with_lora: TinyTRMModel,
    ) -> None:
        """LPRM predicts quality from hidden states correctly."""
        x = torch.randint(0, 100, (1, 8))
        out = model_with_lora(x)
        hidden = out["hidden_states"]  # (1, 8, 64)

        q_all = lprm(hidden)
        assert q_all.shape == (1, 8, 5), (
            f"Expected (1, 8, 5) for 5 modules, got {q_all.shape}"
        )
        assert torch.all(q_all >= 0) and torch.all(q_all <= 1), (
            "LPRM quality outside [0, 1]"
        )


class TestPipelineEndToEnd:
    """Full pipeline: generation with memory + router + LPRM."""

    def test_generate_with_memory(
        self, model_with_lora: TinyTRMModel, memory: HierarchicalMemory,
    ) -> None:
        """Generation works with memory retrieval (no router)."""
        pm = PipelineManager(
            model=model_with_lora, memory=memory, router=None,
            memory_top_k=3,
        )
        x = torch.randint(0, 100, (1, 4))
        seq = pm.generate(x, max_new_tokens=5, temperature=0.0)
        assert seq.shape == (1, 9), f"Expected (1, 9), got {seq.shape}"

    def test_generate_with_router(
        self, model_with_lora: TinyTRMModel,
        router: GameTheoreticRouter,
    ) -> None:
        """Generation works with router (no memory)."""
        pm = PipelineManager(
            model=model_with_lora, memory=None, router=router,
            router_interval=0,
        )
        x = torch.randint(0, 100, (1, 4))
        seq = pm.generate(
            x, max_new_tokens=5, temperature=0.0, retrieve_memory=False,
        )
        assert seq.shape == (1, 9), f"Expected (1, 9), got {seq.shape}"

    def test_generate_full_pipeline(
        self, model_with_lora: TinyTRMModel, memory: HierarchicalMemory,
        router: GameTheoreticRouter,
    ) -> None:
        """Full pipeline: memory + router + LPRM together."""
        pm = PipelineManager(
            model=model_with_lora, memory=memory, router=router,
            memory_top_k=3, router_interval=0,
        )
        x = torch.randint(0, 100, (1, 4))
        seq = pm.generate(
            x, max_new_tokens=5, temperature=0.0,
            retrieve_memory=True, use_router=True,
        )
        assert seq.shape == (1, 9), f"Expected (1, 9), got {seq.shape}"
        # Verify auction ran.
        assert pm.last_auction is not None, "Auction did not run"
        # Verify memory was queried.
        assert pm.last_retrieval is not None, "Memory was not queried"

    def test_experience_recorded(
        self, model_with_lora: TinyTRMModel, memory: HierarchicalMemory,
        router: GameTheoreticRouter,
    ) -> None:
        """Experiences are stored during generation."""
        pm = PipelineManager(
            model=model_with_lora, memory=memory, router=router,
            memory_top_k=3, router_interval=0,
        )
        x = torch.randint(0, 100, (1, 4))

        initial_count = len(memory.replay)
        pm.generate(
            x, max_new_tokens=5, temperature=0.0,
            retrieve_memory=True, use_router=True,
            record_experience=True,
        )
        assert len(memory.replay) > initial_count, (
            "Experience was not stored in replay buffer"
        )

    def test_experience_replay_training(
        self, model_with_lora: TinyTRMModel, memory: HierarchicalMemory,
        router: GameTheoreticRouter, lprm: MultiHeadLPRM,
    ) -> None:
        """LPRM can be trained on stored experiences."""
        from training.trainer import TRMBankTrainer

        pm = PipelineManager(
            model=model_with_lora, memory=memory, router=router,
            memory_top_k=3, router_interval=0,
        )
        x = torch.randint(0, 100, (1, 4))

        # Generate and store several experiences.
        for _ in range(3):
            pm.generate(
                x, max_new_tokens=3, temperature=0.0,
                retrieve_memory=True, use_router=True,
                record_experience=True,
            )

        assert len(memory.replay) >= 3, "Not enough experiences stored"

        # LPRM replay training.
        trainer = TRMBankTrainer(
            model=model_with_lora, lprm=lprm, memory=memory,
        )
        loss = trainer.train_lprm_on_replay(
            batch_size=2, lr=1e-4, num_steps=8,
        )
        assert loss > 0, f"Expected positive loss, got {loss}"
        assert torch.isfinite(torch.tensor(loss)), (
            f"Loss is not finite: {loss}"
        )

    def test_model_generate_with_pipeline(
        self, model_with_lora: TinyTRMModel, memory: HierarchicalMemory,
        router: GameTheoreticRouter,
    ) -> None:
        """Model.generate() with set_pipeline() works end-to-end."""
        model_with_lora.set_pipeline(
            memory=memory, router=router, lprm=router.lprm,
            memory_top_k=3, router_interval=0,
        )
        x = torch.randint(0, 100, (1, 4))
        seq = model_with_lora.generate(
            x, max_new_tokens=5, temperature=0.0, use_pipeline=True,
        )
        assert seq.shape == (1, 9), f"Expected (1, 9), got {seq.shape}"
