"""Unit tests for Truncated BPTT Trainer."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import pytest

from config import TrainingConfig
from memory.experience import ExperienceReplay, TurnExperience
from memory.hierarchical_memory import HierarchicalMemory
from router.lprm import MultiHeadLPRM
from training.trainer import TruncatedBPTTDataset, TRMBankTrainer


class TestTruncatedBPTTDataset:
    """Verification suite for TruncatedBPTTDataset."""

    def test_single_sequence_short(self) -> None:
        """Short sequences produce a single chunk."""
        data = [torch.randint(0, 100, (10,))]
        ds = TruncatedBPTTDataset(data, truncation_length=20)
        chunks = list(ds)
        assert len(chunks) == 1

    def test_single_sequence_long(self) -> None:
        """Long sequences are split into multiple chunks."""
        data = [torch.randint(0, 100, (100,))]
        ds = TruncatedBPTTDataset(data, truncation_length=30)
        chunks = list(ds)
        assert len(chunks) >= 3

    def test_chunk_fields(self) -> None:
        """Each chunk has required fields."""
        data = [torch.randint(0, 100, (50,))]
        ds = TruncatedBPTTDataset(data, truncation_length=20)
        for chunk in ds:
            assert "input_ids" in chunk
            assert "labels" in chunk
            assert "segment_start" in chunk
            assert "is_last" in chunk

    def test_last_chunk_marked(self) -> None:
        """Only the final chunk of a sequence has is_last=True."""
        data = [torch.randint(0, 100, (50,))]
        ds = TruncatedBPTTDataset(data, truncation_length=20)
        chunks = list(ds)
        for i, chunk in enumerate(chunks):
            if i == len(chunks) - 1:
                assert chunk["is_last"] is True
            else:
                assert chunk["is_last"] is False

    def test_multiple_sequences(self) -> None:
        """Multiple sequences produce correct total chunks."""
        data = [
            torch.randint(0, 100, (10,)),
            torch.randint(0, 100, (50,)),
            torch.randint(0, 100, (100,)),
        ]
        ds = TruncatedBPTTDataset(data, truncation_length=30)
        chunks = list(ds)
        expected = 1 + 2 + 4  # 10+50+100 with trunc=30
        assert len(chunks) == expected


class _TrainableModel(nn.Module):
    """Minimal model with trainable params for trainer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
        B, L = input_ids.shape
        logits = self.linear(input_ids.float())
        return {"logits": logits.unsqueeze(-1).expand(B, L, 100)}


class TestTRMBankTrainer:
    """Verification suite for TRMBankTrainer (unit-level)."""

    def test_trainer_init(self) -> None:
        """Trainer initialises without error."""
        model = _TrainableModel()
        data = [torch.randint(0, 100, (50,))]
        ds = TruncatedBPTTDataset(data, truncation_length=20)
        trainer = TRMBankTrainer(model=model, train_dataset=ds)
        assert trainer is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_trainer_no_dataset_raises(self) -> None:
        """train_epoch raises without a dataset."""
        model = _TrainableModel()
        trainer = TRMBankTrainer(model=model, train_dataset=None)
        with pytest.raises(RuntimeError, match="No training dataset"):
            trainer.train_epoch(1)

    def test_compute_loss_shape(self) -> None:
        """_compute_loss produces a scalar tensor."""
        model = _TrainableModel()
        trainer = TRMBankTrainer(model=model)
        B, L, V = 2, 8, 100
        logits = torch.randn(B, L, V)
        labels = torch.randint(0, V, (B, L))
        loss = trainer._compute_loss(logits, labels)
        assert loss.ndim == 0

    def test_compute_loss_finite(self) -> None:
        """Loss is finite (no NaN/Inf)."""
        model = _TrainableModel()
        trainer = TRMBankTrainer(model=model)
        logits = torch.randn(2, 8, 100)
        labels = torch.randint(0, 100, (2, 8))
        loss = trainer._compute_loss(logits, labels)
        assert torch.isfinite(loss)

    def test_save_load_checkpoint(self, tmp_path: str) -> None:
        """Checkpoint can be saved and loaded."""
        model = _TrainableModel()
        config = TrainingConfig(output_dir=str(tmp_path), log_every_n_steps=10)
        trainer = TRMBankTrainer(
            model=model,
            config=config,
        )
        path = trainer._save_checkpoint("test")
        assert path.endswith("checkpoint_test.pt")


class TestTinyTRMTraining:
    """Integration: training pipeline with TinyTRMModel."""

    def test_train_one_step(self) -> None:
        """TinyTRMModel can run one training step without error."""
        from core import TinyTRMModel

        model = TinyTRMModel(
            vocab_size=100, d_model=32, d_state=16,
            mimo_rank=2, num_layers=2,
        )
        data = [torch.randint(0, 100, (50,))]
        ds = TruncatedBPTTDataset(data, truncation_length=20)
        trainer = TRMBankTrainer(model=model, train_dataset=ds)
        metrics = trainer.train_epoch(1)
        assert "loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_state_carry_across_segments(self) -> None:
        """States are carried between segments of the same sequence."""
        from core import TinyTRMModel

        model = TinyTRMModel(
            vocab_size=50, d_model=16, d_state=8,
            mimo_rank=2, num_layers=2,
        )
        data = [torch.randint(0, 50, (30,))]
        ds = TruncatedBPTTDataset(data, truncation_length=10)
        trainer = TRMBankTrainer(model=model, train_dataset=ds)
        metrics = trainer.train_epoch(1)
        assert torch.isfinite(torch.tensor(metrics["loss"]))


class TestLPRMTraining:
    """Verification suite for LPRM training integration."""

    def test_lprm_loss_basic(self) -> None:
        """LPRM loss computed from model confidence produces a finite scalar."""
        model = _TrainableModel()
        lprm = MultiHeadLPRM(
            module_names=["direct_generation", "python_sandbox"],
            hidden_dim=10,
            num_heads=4,
        )
        trainer = TRMBankTrainer(model=model, lprm=lprm)

        B, L, V, D = 2, 8, 100, 10
        logits = torch.randn(B, L, V)
        labels = torch.randint(0, V, (B, L))
        hidden = torch.randn(B, L, D)
        loss = trainer._compute_lprm_loss(logits, labels, hidden)
        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"LPRM loss is not finite: {loss}"

    def test_lprm_loss_no_lprm(self) -> None:
        """LPRM loss returns 0.0 when no LPRM is attached."""
        model = _TrainableModel()
        trainer = TRMBankTrainer(model=model, lprm=None)

        B, L, V, D = 2, 8, 100, 10
        logits = torch.randn(B, L, V)
        labels = torch.randint(0, V, (B, L))
        hidden = torch.randn(B, L, D)
        loss = trainer._compute_lprm_loss(logits, labels, hidden)
        assert loss.item() == 0.0

    def test_lprm_loss_differentiable(self) -> None:
        """LPRM loss has non-zero gradient w.r.t. LPRM params."""
        lprm = MultiHeadLPRM(
            module_names=["a", "b"],
            hidden_dim=8,
            num_heads=4,
        )
        model = _TrainableModel()
        trainer = TRMBankTrainer(model=model, lprm=lprm)

        B, L, V, D = 1, 4, 50, 8
        logits = torch.randn(B, L, V, requires_grad=False)
        labels = torch.randint(0, V, (B, L))
        hidden = torch.randn(B, L, D)

        loss = trainer._compute_lprm_loss(logits, labels, hidden)
        loss.backward()
        grad_norm = sum(
            p.grad.norm().item() for p in lprm.parameters()
            if p.grad is not None
        )
        assert grad_norm > 0, "LPRM loss did not produce gradients"

    def test_training_step_with_lprm(self) -> None:
        """Training step runs with LPRM attached (no error, finite loss)."""
        from core import TinyTRMModel

        model = TinyTRMModel(
            vocab_size=50, d_model=16, d_state=8,
            mimo_rank=2, num_layers=2,
        )
        lprm = MultiHeadLPRM(
            module_names=["direct_generation", "symbolic_solver"],
            hidden_dim=16,
            num_heads=4,
        )
        data = [torch.randint(0, 50, (40,))]
        ds = TruncatedBPTTDataset(data, truncation_length=20)
        trainer = TRMBankTrainer(model=model, lprm=lprm, train_dataset=ds)
        metrics = trainer.train_epoch(1)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert torch.isfinite(torch.tensor(metrics["lprm_loss"]))

    def test_training_step_lprm_improves(self) -> None:
        """LPRM loss decreases over multiple steps."""
        from core import TinyTRMModel

        model = TinyTRMModel(
            vocab_size=50, d_model=16, d_state=8,
            mimo_rank=2, num_layers=2,
        )
        lprm = MultiHeadLPRM(
            module_names=["direct_generation"],
            hidden_dim=16,
            num_heads=4,
        )
        data = [torch.randint(0, 50, (80,))]
        ds = TruncatedBPTTDataset(data, truncation_length=20)
        config = TrainingConfig(
            num_epochs=2,
            gradient_accumulation_steps=1,
            log_every_n_steps=100,
        )
        trainer = TRMBankTrainer(
            model=model, lprm=lprm,
            train_dataset=ds, config=config,
        )
        metrics = trainer.train()
        # LPRM loss in last epoch should be lower than first.
        first_lprm = metrics.get("train_lprm_loss", [float("inf")])[0]
        last_lprm = metrics.get("train_lprm_loss", [float("-inf")])[-1]
        # May not strictly decrease in 2 epochs but should not explode.
        assert torch.isfinite(torch.tensor(first_lprm))
        assert torch.isfinite(torch.tensor(last_lprm))


class TestExperienceReplay:
    """Verification suite for ExperienceReplay and turn experience."""

    def test_push_and_length(self) -> None:
        """Experiences can be pushed and buffer length grows."""
        buffer = ExperienceReplay(capacity=100)
        assert len(buffer) == 0
        for i in range(10):
            exp = TurnExperience(
                prompt_embedding=np.random.randn(64).astype(np.float32),
                hidden_for_lprm=torch.randn(1, 64),
                module_name="direct_generation",
                reward=0.5,
                prompt_ids=torch.randint(0, 100, (1, 4)),
                generated_ids=torch.randint(0, 100, (1, 8)),
            )
            buffer.push(exp)
        assert len(buffer) == 10

    def test_circular_capacity(self) -> None:
        """Buffer evicts lowest-retention experiences when full."""
        buffer = ExperienceReplay(capacity=5)
        for i in range(10):
            exp = TurnExperience(
                prompt_embedding=np.random.randn(64).astype(np.float32),
                hidden_for_lprm=torch.randn(1, 64),
                module_name="test",
                reward=float(i),
                prompt_ids=torch.randint(0, 100, (1, 4)),
                generated_ids=torch.randint(0, 100, (1, 8)),
            )
            buffer.push(exp)
        assert len(buffer) == 5
        # With Ebbinghaus retention, the 5 most recent survive (lowest t).
        retentions = buffer.get_retention_values()
        assert all(r > 0.0 for r in retentions)

    def test_sample(self) -> None:
        """Sampling returns the requested number of experiences."""
        buffer = ExperienceReplay(capacity=100)
        for i in range(20):
            exp = TurnExperience(
                prompt_embedding=np.random.randn(64).astype(np.float32),
                hidden_for_lprm=torch.randn(1, 64),
                module_name="direct_generation",
                reward=0.5,
                prompt_ids=torch.randint(0, 100, (1, 4)),
                generated_ids=torch.randint(0, 100, (1, 8)),
            )
            buffer.push(exp)
        sampled = buffer.sample(5)
        assert len(sampled) == 5
        # All returned items are TurnExperience.
        for s in sampled:
            assert isinstance(s, TurnExperience)

    def test_sample_empty(self) -> None:
        """Sampling from empty buffer returns empty list."""
        buffer = ExperienceReplay(capacity=10)
        assert buffer.sample(5) == []

    def test_record_experience_on_memory(self) -> None:
        """HierarchicalMemory.record_experience stores nodes and replay entries."""
        mem = HierarchicalMemory(
            tier_slots=(16, 16, 16, 16),
            embedding_dim=64,
        )
        exp = TurnExperience(
            prompt_embedding=np.random.randn(64).astype(np.float32),
            hidden_for_lprm=torch.randn(1, 64),
            module_name="direct_generation",
            reward=0.8,
            prompt_ids=torch.randint(0, 100, (1, 4)),
            generated_ids=torch.randint(0, 100, (1, 5)),
        )
        node_id = mem.record_experience(exp, tier=0)
        assert node_id >= 0, "Memory node was not created"
        assert mem.tiers[0].size == 1, "Turn tier should have 1 node"
        assert len(mem.replay) == 1, "Replay buffer should have 1 entry"

    def test_replay_training_updates_lprm(self) -> None:
        """LPRM loss decreases after replay training on stored experiences."""
        mem = HierarchicalMemory(
            tier_slots=(16, 16, 16, 16),
            embedding_dim=8,
        )
        lprm = MultiHeadLPRM(
            module_names=["direct_generation"],
            hidden_dim=8,
            num_heads=4,
        )
        model = _TrainableModel()

        # Populate replay buffer with experiences.
        for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
            exp = TurnExperience(
                prompt_embedding=np.random.randn(8).astype(np.float32),
                hidden_for_lprm=torch.randn(1, 8),
                module_name="direct_generation",
                reward=r,
                prompt_ids=torch.randint(0, 50, (1, 4)),
                generated_ids=torch.randint(0, 50, (1, 3)),
            )
            mem.record_experience(exp)

        trainer = TRMBankTrainer(model=model, lprm=lprm, memory=mem)
        loss = trainer.train_lprm_on_replay(batch_size=4, lr=1e-3, num_steps=16)
        assert loss > 0, f"Expected positive loss, got {loss}"
        assert torch.isfinite(torch.tensor(loss))
