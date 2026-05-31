"""Unit tests for Hierarchical Memory module."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from memory.hierarchical_memory import (
    FAISSIndex,
    MemoryTier,
    HierarchicalMemory,
    SemanticPredictor,
    RetrievalResult,
    MemoryNode,
)


class TestFAISSIndex:
    """Verification suite for FAISSIndex wrapper."""

    def test_build_and_search(self) -> None:
        """FAISSIndex returns correct number of results."""
        index = FAISSIndex(dim=16)
        emb = np.random.randn(100, 16).astype(np.float32)
        index.build(emb)
        query = np.random.randn(16).astype(np.float32)
        dist, idx = index.search(query, top_k=5)
        assert len(idx[0]) == 5, f"Expected 5 results, got {len(idx[0])}"

    def test_add_and_search(self) -> None:
        """Adding embeddings incrementally works."""
        index = FAISSIndex(dim=8)
        for i in range(10):
            emb = np.random.randn(8).astype(np.float32)
            index.add(emb)
        query = np.random.randn(8).astype(np.float32)
        dist, idx = index.search(query, top_k=3)
        assert len(idx[0]) == 3

    def test_search_without_build(self) -> None:
        """Search on empty index returns empty arrays."""
        index = FAISSIndex(dim=8)
        query = np.random.randn(8).astype(np.float32)
        dist, idx = index.search(query, top_k=5)
        assert len(dist) == 0 or len(idx[0]) == 0


class TestMemoryTier:
    """Verification suite for MemoryTier."""

    def test_add_node_and_search(self) -> None:
        """MemoryTier returns nodes from search."""
        tier = MemoryTier(tier_level=0, embedding_dim=16, max_slots=100)
        for i in range(20):
            emb = np.random.randn(16).astype(np.float32)
            tier.add_node(emb, summary=f"node_{i}")
        query = np.random.randn(16).astype(np.float32)
        results = tier.search(query, top_k=3)
        assert len(results) == 3
        assert all(isinstance(n, MemoryNode) for n in results)

    def test_add_node_full(self) -> None:
        """Adding to a full tier evicts the lowest-retention node."""
        tier = MemoryTier(tier_level=0, embedding_dim=4, max_slots=2)
        tier.add_node(np.random.randn(4).astype(np.float32))
        tier.add_node(np.random.randn(4).astype(np.float32))
        result = tier.add_node(np.random.randn(4).astype(np.float32))
        assert result >= 0, "Expected eviction and successful addition"
        assert tier.size == 2, "Tier should stay at capacity after eviction"

    def test_size_property(self) -> None:
        """size reflects the number of nodes."""
        tier = MemoryTier(tier_level=0, embedding_dim=4, max_slots=10)
        assert tier.size == 0
        tier.add_node(np.random.randn(4).astype(np.float32))
        assert tier.size == 1
        tier.add_node(np.random.randn(4).astype(np.float32))
        assert tier.size == 2

    def test_get_children(self) -> None:
        """get_children returns the correct child indices."""
        tier = MemoryTier(tier_level=1, embedding_dim=4, max_slots=10)
        tier.add_node(np.random.randn(4).astype(np.float32), children=[0, 1, 2])
        tier.add_node(np.random.randn(4).astype(np.float32), children=[3, 4])
        children = tier.get_children(0)
        assert children == [0, 1, 2]
        children = tier.get_children(1)
        assert children == [3, 4]

    def test_get_children_out_of_range(self) -> None:
        """get_children returns empty list for invalid node."""
        tier = MemoryTier(tier_level=0, embedding_dim=4, max_slots=5)
        assert tier.get_children(99) == []


class TestHierarchicalMemory:
    """Verification suite for HierarchicalMemory."""

    def test_memory_structure(self) -> None:
        """Memory has 4 tiers by default."""
        memory = HierarchicalMemory()
        assert memory.num_tiers == 4
        assert memory.tier_names == ("Domain", "Category", "Event", "Turn")

    def test_add_and_retrieve(self) -> None:
        """Add nodes to all tiers and perform retrieval."""
        memory = HierarchicalMemory(embedding_dim=8, tier_slots=(4, 8, 16, 32))

        # Add domain nodes (tier 3)
        for d in range(4):
            memory.add_memory(3, np.random.randn(8).astype(np.float32),
                              summary=f"domain_{d}")

        # Add category nodes (tier 2) with children pointing to domain
        for c in range(8):
            domain_id = c % 4
            memory.add_memory(2, np.random.randn(8).astype(np.float32),
                              summary=f"cat_{c}", children=[domain_id])

        # Rebuild indexes after adding
        memory.rebuild_all_indexes()

        query = np.random.randn(8).astype(np.float32)
        result = memory.retrieve(query, top_k=3)
        assert isinstance(result, RetrievalResult)
        assert len(result.turn_ids) <= 3

    def test_retrieve_returns_scores(self) -> None:
        """Retrieval result includes scores."""
        memory = HierarchicalMemory(embedding_dim=8, tier_slots=(2, 4, 8, 16))

        for d in range(2):
            memory.add_memory(3, np.random.randn(8).astype(np.float32))
        for c in range(4):
            memory.add_memory(2, np.random.randn(8).astype(np.float32))
        for e in range(8):
            memory.add_memory(1, np.random.randn(8).astype(np.float32))

        memory.rebuild_all_indexes()
        query = np.random.randn(8).astype(np.float32)
        result = memory.retrieve(query, top_k=2)
        assert len(result.scores) == len(result.turn_ids)

    def test_retrieve_empty_memory(self) -> None:
        """Retrieval from empty memory returns empty results."""
        memory = HierarchicalMemory(embedding_dim=8, tier_slots=(2, 4, 8, 16))
        query = np.random.randn(8).astype(np.float32)
        result = memory.retrieve(query, top_k=3)
        assert len(result.turn_ids) == 0

    def test_stats_report(self) -> None:
        """stats() returns counts per tier."""
        memory = HierarchicalMemory(embedding_dim=8, tier_slots=(2, 4, 8, 16))
        memory.add_memory(3, np.random.randn(8).astype(np.float32))
        memory.add_memory(2, np.random.randn(8).astype(np.float32))
        memory.add_memory(2, np.random.randn(8).astype(np.float32))
        memory.add_memory(0, np.random.randn(8).astype(np.float32))
        stats = memory.stats()
        assert stats["Domain"] == 1
        assert stats["Category"] == 2
        assert stats["Turn"] == 1

    def test_set_predictor(self) -> None:
        """SemanticPredictor can be attached to memory."""
        memory = HierarchicalMemory(embedding_dim=16, tier_slots=(2, 4, 8, 16))
        predictor = SemanticPredictor(embedding_dim=16)
        memory.set_predictor(predictor)
        assert memory.predictor is not None


class TestSemanticPredictor:
    """Verification suite for SemanticPredictor."""

    def test_forward_shape(self) -> None:
        """Predictor outputs (B, N) relevance scores."""
        predictor = SemanticPredictor(embedding_dim=16, summary_encoder_dim=8)
        query = torch.randn(2, 16)
        nodes = torch.randn(2, 5, 16)
        scores = predictor(query, nodes)
        assert scores.shape == (2, 5), f"Expected (2, 5), got {scores.shape}"

    def test_backward(self) -> None:
        """Predictor supports gradient flow."""
        predictor = SemanticPredictor(embedding_dim=16)
        query = torch.randn(1, 16, requires_grad=True)
        nodes = torch.randn(1, 3, 16)
        scores = predictor(query, nodes)
        loss = scores.sum()
        loss.backward()
        assert query.grad is not None

    def test_forward_with_summaries(self) -> None:
        """Predictor accepts optional summary features."""
        predictor = SemanticPredictor(embedding_dim=16, summary_encoder_dim=8)
        query = torch.randn(1, 16)
        nodes = torch.randn(1, 3, 16)
        summaries = torch.randn(1, 3, 8)
        scores = predictor(query, nodes, node_summaries=summaries)
        assert scores.shape == (1, 3)

    def test_node_dataclass(self) -> None:
        """MemoryNode dataclass stores attributes correctly."""
        node = MemoryNode(node_id=5, tier=1,
                          embedding=np.array([1.0, 2.0]),
                          summary="test", children=[0, 1])
        assert node.node_id == 5
        assert node.tier == 1
        assert node.summary == "test"
        assert node.children == [0, 1]
