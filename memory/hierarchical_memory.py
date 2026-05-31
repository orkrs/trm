from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from memory.experience import EbbinghausForgettingCurve, ExperienceReplay, TurnExperience


@dataclass
class MemoryNode:
    """A single node in the hierarchical memory graph.

    Attributes:
        node_id: Unique identifier within its tier.
        tier: Level index (0=Turn, 1=Event, 2=Category, 3=Domain).
        embedding: Dense semantic embedding vector.
        summary: Optional text summary (used for prediction).
        children: Indices of child nodes in the next (lower) tier.
        strength: Ebbinghaus forgetting-curve strength (increases on
                  successful retrieval).
        last_access_time: Global step counter value of last retrieval.
        access_count: Number of times this node has been retrieved.
    """
    node_id: int
    tier: int
    embedding: np.ndarray
    summary: str = ""
    children: List[int] = None
    strength: float = 1.0
    last_access_time: int = 0
    access_count: int = 0

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []


@dataclass
class RetrievalResult:
    """Result of a hierarchical memory retrieval.

    Attributes:
        turn_ids: Retrieved leaf node indices from the Turn tier.
        scores: Relevance scores for each retrieved leaf.
        paths: Full paths through the hierarchy for each result.
    """
    turn_ids: List[int]
    scores: List[float]
    paths: List[List[int]]


class FAISSIndex:
    """Wrapper around FAISS for approximate nearest neighbor search.

    Falls back to brute-force torch search when FAISS is unavailable.

    Args:
        dim: Embedding dimension.
        index_type: FAISS index type (e.g., "Flat", "IVF").
    """

    def __init__(self, dim: int, index_type: str = "Flat") -> None:
        self.dim: int = dim
        self.index_type: str = index_type
        self._faiss_available: bool = False
        self._index = None
        self._storage: List[np.ndarray] = []

    def build(self, embeddings: np.ndarray) -> None:
        """Build or rebuild the index from a matrix of embeddings.

        Args:
            embeddings: Array of shape (N, D).
        """
        self._storage = [embeddings[i] for i in range(len(embeddings))]

        try:
            import faiss
            self._faiss_available = True
            if self.index_type == "Flat":
                self._index = faiss.IndexFlatIP(self.dim)
            elif self.index_type.startswith("IVF"):
                nlist = min(int(np.sqrt(len(embeddings))), 256)
                quantizer = faiss.IndexFlatIP(self.dim)
                self._index = faiss.IndexIVFFlat(quantizer, self.dim, nlist,
                                                  faiss.METRIC_INNER_PRODUCT)
                self._index.train(embeddings.astype(np.float32))
            else:
                self._index = faiss.IndexFlatIP(self.dim)

            self._index.add(embeddings.astype(np.float32))
            logger.info(
                f"FAISS index built: {self.index_type}, N={len(embeddings)}, D={self.dim}"
            )
        except ImportError:
            self._faiss_available = False
            logger.warning("FAISS not available, using brute-force torch search.")

    def search(self, query: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k nearest neighbors.

        Args:
            query: Query vector of shape (D,) or (Q, D).
            top_k: Number of nearest neighbors.

        Returns:
            Tuple of (distances, indices) arrays.
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self._faiss_available and self._index is not None:
            distances, indices = self._index.search(query.astype(np.float32), top_k)
            return distances, indices

        return self._brute_force_search(query, top_k)

    def _brute_force_search(
        self, query: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: brute-force cosine similarity via torch.

        Args:
            query: Query array of shape (Q, D).
            top_k: Number of results.

        Returns:
            Tuple of (distances, indices).
        """
        if not self._storage:
            return np.array([]), np.array([])

        storage = np.stack(self._storage, axis=0)            # (N, D)
        q = torch.from_numpy(query.astype(np.float32))       # (Q, D)
        s = torch.from_numpy(storage.astype(np.float32))     # (N, D)

        sim = torch.mm(q, s.t())                              # (Q, N)
        sim_values, sim_indices = sim.topk(
            min(top_k, sim.size(1)), dim=1, sorted=True
        )
        return sim_values.numpy(), sim_indices.numpy()

    def add(self, embedding: np.ndarray) -> int:
        """Add a single embedding to the index.

        Args:
            embedding: Vector of shape (D,).

        Returns:
            The assigned index.
        """
        idx = len(self._storage)
        self._storage.append(embedding)

        if self._faiss_available and self._index is not None:
            self._index.add(embedding.reshape(1, -1).astype(np.float32))

        return idx

    @property
    def size(self) -> int:
        return len(self._storage)


class MemoryTier:
    """A single level in the 4-tier hierarchical memory.

    Each tier stores dense embeddings paired with child pointers to
    the next lower tier.  A FAISS index provides fast similarity search.
    When the tier is full, the node with the lowest Ebbinghaus retention
    is evicted to make room for new nodes.

    Args:
        tier_level: Index (0=Turn, 1=Event, 2=Category, 3=Domain).
        embedding_dim: Dimensionality of the embedding space.
        max_slots: Maximum number of nodes in this tier.
        faiss_index_type: FAISS index type ("Flat" or "IVF").
        ebbinghaus: EbbinghausForgettingCurve instance (created
                    automatically if not provided).
    """

    def __init__(
        self,
        tier_level: int,
        embedding_dim: int = 256,
        max_slots: int = 4096,
        faiss_index_type: str = "Flat",
        ebbinghaus: Optional[EbbinghausForgettingCurve] = None,
    ) -> None:
        self.tier_level: int = tier_level
        self.embedding_dim: int = embedding_dim
        self.max_slots: int = max_slots

        self.nodes: List[MemoryNode] = []
        self.index = FAISSIndex(embedding_dim, faiss_index_type)
        self._children_matrix: List[List[int]] = []
        self._ebbinghaus: EbbinghausForgettingCurve = (
            ebbinghaus or EbbinghausForgettingCurve()
        )
        self._step: int = 0

    @property
    def size(self) -> int:
        return len(self.nodes)

    def add_node(
        self,
        embedding: np.ndarray,
        summary: str = "",
        children: Optional[List[int]] = None,
    ) -> int:
        """Add a memory node to this tier.

        If the tier is full, the node with the lowest Ebbinghaus
        retention is evicted before adding the new node.

        Args:
            embedding: Dense vector of shape (D,).
            summary: Optional text summary.
            children: Child indices in the next tier.

        Returns:
            Node ID (index within this tier).
        """
        if self.size >= self.max_slots:
            evict_id: int = self._find_eviction_candidate()
            self._evict_node(evict_id)

        node_id = self.size
        node = MemoryNode(
            node_id=node_id,
            tier=self.tier_level,
            embedding=embedding,
            summary=summary,
            children=children or [],
        )
        self.nodes.append(node)
        self._children_matrix.append(node.children)
        self.index.add(embedding)
        return node_id

    def _find_eviction_candidate(self) -> int:
        """Return the index of the node with lowest Ebbinghaus retention."""
        current_step: int = self._step
        min_ret: float = float("inf")
        min_idx: int = 0
        for i, node in enumerate(self.nodes):
            t: int = current_step - node.last_access_time
            ret: float = self._ebbinghaus.retention(float(t), node.strength)
            if ret < min_ret:
                min_ret = ret
                min_idx = i
        return min_idx

    def _evict_node(self, idx: int) -> None:
        """Evict the node at index ``idx`` and rebuild the FAISS index.

        Note: eviction is O(N) because the FAISS index is rebuilt
        from scratch after removing the node.
        """
        evicted: MemoryNode = self.nodes.pop(idx)
        self._children_matrix.pop(idx)
        logger.debug(
            f"Tier {self.tier_level}: evicted node {evicted.node_id} "
            f"(strength={evicted.strength:.3f}, accesses={evicted.access_count})"
        )
        self.rebuild_index()

    def search(self, query: np.ndarray, top_k: int = 5) -> List[MemoryNode]:
        """Retrieve top-k most similar nodes from this tier.

        Updates node access stats (strength, last_access_time,
        access_count) for the retrieved nodes.

        Args:
            query: Query embedding of shape (D,).
            top_k: Number of results.

        Returns:
            List of retrieved MemoryNode objects.
        """
        if self.size == 0:
            return []

        distances, indices = self.index.search(query, top_k)
        results: List[MemoryNode] = []
        for idx_arr in indices:
            for idx in idx_arr:
                if 0 <= idx < len(self.nodes):
                    node: MemoryNode = self.nodes[int(idx)]
                    node.access_count += 1
                    node.last_access_time = self._step
                    results.append(node)
        self._step += 1
        return results

    def get_embeddings_matrix(self) -> np.ndarray:
        """Return all embeddings as a (N, D) matrix."""
        if not self.nodes:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.stack([n.embedding for n in self.nodes], axis=0).astype(np.float32)

    def rebuild_index(self) -> None:
        """Rebuild the FAISS index from all stored nodes."""
        matrix = self.get_embeddings_matrix()
        if matrix.shape[0] > 0:
            self.index.build(matrix)

    def get_children(self, node_id: int) -> List[int]:
        """Return child indices for a given node."""
        if 0 <= node_id < len(self._children_matrix):
            return self._children_matrix[node_id]
        return []

    @property
    def ebbinghaus(self) -> EbbinghausForgettingCurve:
        return self._ebbinghaus


class SemanticPredictor(nn.Module):
    """Lightweight neural predictor for top-down retrieval guidance.

    Given a query embedding and a set of candidate node summaries,
    predicts which nodes are relevant (binary relevance score).

    Used to narrow the search space before dense FAISS retrieval,
    as described in the TRM-Bank v3.0 memory design.

    Args:
        embedding_dim: Dimensionality of node embeddings.
        summary_encoder_dim: Size of the summary encoding projection.
        hidden_dim: Hidden layer size.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        summary_encoder_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.query_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.node_proj = nn.Linear(embedding_dim + summary_encoder_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        query_emb: torch.Tensor,
        node_embs: torch.Tensor,
        node_summaries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict relevance scores for candidate nodes.

        When node_summaries is None, the query_proj output is used
        directly for the node projection (the node_proj layer is not
        called).

        Args:
            query_emb: Query embedding of shape (B, D_q).
            node_embs: Candidate node embeddings of shape (B, N, D_n).
            node_summaries: Optional summary features of shape (B, N, S).

        Returns:
            Relevance logits of shape (B, N).
        """
        q = self.query_proj(query_emb).unsqueeze(1)   # (B, 1, H)
        if node_summaries is not None:
            combined = torch.cat([node_embs, node_summaries], dim=-1)
            n = self.node_proj(combined)               # (B, N, H)
        else:
            n = self.query_proj(node_embs)              # (B, N, H)
        scores = self.score_proj(torch.tanh(q + n))    # (B, N, 1)
        return scores.squeeze(-1)                      # (B, N)


class HierarchicalMemory:
    """4-tier hierarchical memory with FAISS-accelerated top-down retrieval.

    Structure:
        Tier 3 (Domain):   Abstract domain categories.
        Tier 2 (Category): Metadata and topic categories.
        Tier 1 (Event):    Summaries of events.
        Tier 0 (Turn):     Raw facts and tokens.

    Retrieval proceeds top-down: query -> Domain -> Category -> Event -> Turn.
    At each level, the most relevant nodes are selected via FAISS search.
    A SemanticPredictor can optionally pre-filter candidates using node
    summaries before dense search.

    Memory node weights are managed by an Ebbinghaus forgetting curve
    with RL-based updates: nodes that are successfully retrieved (high
    reward) get their strength reinforced, while nodes with low reward
    decay, making them eviction candidates when the tier is full.

    Reference: TRM-Bank v3.0 Section 3 (Hierarchical Neuro-Symbolic Memory).

    Args:
        tier_names: Names for each level (top to bottom).
        tier_slots: Max nodes per tier (top to bottom).
        embedding_dim: Dimension of node embeddings.
        faiss_index_type: FAISS index type.
        ebbinghaus: EbbinghausForgettingCurve instance (created
                    automatically if not provided).
    """

    def __init__(
        self,
        tier_names: Tuple[str, ...] = ("Domain", "Category", "Event", "Turn"),
        tier_slots: Tuple[int, ...] = (64, 256, 1024, 4096),
        embedding_dim: int = 256,
        faiss_index_type: str = "Flat",
        ebbinghaus: Optional[EbbinghausForgettingCurve] = None,
    ) -> None:
        if len(tier_names) != len(tier_slots):
            raise ValueError(
                f"tier_names ({len(tier_names)}) and tier_slots "
                f"({len(tier_slots)}) must have the same length."
            )

        self.tier_names: Tuple[str, ...] = tier_names
        self.num_tiers: int = len(tier_names)
        self.embedding_dim: int = embedding_dim

        curve: EbbinghausForgettingCurve = ebbinghaus or EbbinghausForgettingCurve()

        # Tiers in top-to-bottom order (index 0 = top, index -1 = bottom).
        self.tiers: List[MemoryTier] = [
            MemoryTier(
                tier_level=i,
                embedding_dim=embedding_dim,
                max_slots=tier_slots[self.num_tiers - 1 - i],
                faiss_index_type=faiss_index_type,
                ebbinghaus=curve,
            )
            for i in range(self.num_tiers)
        ]
        # tiers[0] = Turn (bottom, most detailed)
        # tiers[1] = Event
        # tiers[2] = Category
        # tiers[3] = Domain (top, most abstract)

        self.predictor: Optional[SemanticPredictor] = None
        self.replay: ExperienceReplay = ExperienceReplay(capacity=10000, ebbinghaus=curve)
        self._last_retrieval: Optional[RetrievalResult] = None

    def record_experience(
        self,
        exp: TurnExperience,
        tier: int = 0,
    ) -> int:
        """Store a turn experience as a memory node and in the replay buffer.

        The experience embedding is added as a Turn-tier (tier 0) memory
        node for future retrieval.  The full experience is also pushed
        to the internal ``ExperienceReplay`` buffer for offline training.
        After storage, the Ebbinghaus strength of nodes from the last
        retrieval is updated based on the experience reward (RL update).

        Args:
            exp: The TurnExperience to record.
            tier: Target memory tier (default 0 = Turn).

        Returns:
            Node ID in the target tier, or -1 on failure.
        """
        node_id: int = self.add_memory(
            tier=tier,
            embedding=exp.prompt_embedding,
            summary=f"module={exp.module_name}, reward={exp.reward:.4f}",
        )
        self.replay.push(exp)
        # RL-based Ebbinghaus weight update for retrieved nodes.
        self._update_node_strengths(exp.reward)
        return node_id

    def _update_node_strengths(self, reward: float) -> None:
        """Apply RL-based Ebbinghaus strength updates to the last retrieved nodes.

        Nodes retrieved in the most recent memory query get their
        forgetting-curve strength updated according to the task reward.
        High reward -> strength increase (slower decay).  Low reward ->
        strength decrease (faster decay, eviction candidate).

        Args:
            reward: Reward signal from the generation turn.
        """
        if self._last_retrieval is None:
            return
        for turn_id in self._last_retrieval.turn_ids:
            if 0 <= turn_id < self.tiers[0].size:
                node: MemoryNode = self.tiers[0].nodes[turn_id]
                old_strength: float = node.strength
                node.strength = self.tiers[0].ebbinghaus.update_strength(
                    node.strength, reward
                )
                logger.debug(
                    f"RL update: node {turn_id} strength "
                    f"{old_strength:.3f} -> {node.strength:.3f} "
                    f"(reward={reward:.4f})"
                )

    def set_predictor(self, predictor: SemanticPredictor) -> None:
        """Attach a semantic predictor for guided retrieval."""
        self.predictor = predictor

    def add_memory(
        self,
        tier: int,
        embedding: np.ndarray,
        summary: str = "",
        children: Optional[List[int]] = None,
    ) -> int:
        """Add a memory node to a specific tier.

        Args:
            tier: Tier index (0=Turn, 1=Event, 2=Category, 3=Domain).
            embedding: Embedding vector of shape (D,).
            summary: Text summary of the node.
            children: Child node indices in the next lower tier.

        Returns:
            Node ID within the tier, or -1 on failure.
        """
        if tier < 0 or tier >= self.num_tiers:
            raise ValueError(f"Invalid tier index {tier}. Must be in [0, {self.num_tiers - 1}].")
        return self.tiers[tier].add_node(embedding, summary, children)

    def retrieve(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        use_predictor: bool = False,
    ) -> RetrievalResult:
        """Top-down hierarchical memory retrieval.

        Starts from the top (Domain) tier and narrows down to the
        bottom (Turn) tier through child pointer traversal.

        Args:
            query_emb: Query embedding of shape (D,).
            top_k: Number of results to return.
            use_predictor: Whether to use the semantic predictor for
                           candidate pre-filtering.

        Returns:
            RetrievalResult with leaf node ids, scores, and paths.
        """
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # Start from the top tier (Domain).
        candidates: List[Tuple[int, float, List[int]]] = []  # (node_id, score, path)

        for tier_idx in range(self.num_tiers - 1, -1, -1):
            tier = self.tiers[tier_idx]

            if tier.size == 0:
                continue

            if tier_idx == self.num_tiers - 1:
                # Top tier: search via FAISS.
                nodes = tier.search(query_emb.flatten(), top_k=top_k)
                candidates = [
                    (n.node_id, 1.0, [n.node_id])
                    for n in nodes
                ]
            else:
                # Lower tiers: follow children of current candidates.
                new_candidates: List[Tuple[int, float, List[int]]] = []

                for node_id, score, path in candidates:
                    child_ids = tier.get_children(node_id)
                    for cid in child_ids:
                        if 0 <= cid < tier.size:
                            child_node = tier.nodes[cid]
                            # Score: cosine similarity with query.
                            cossim = self._cosine_sim(
                                query_emb.flatten(), child_node.embedding
                            )
                            combined_score = score * max(0.0, cossim)
                            new_candidates.append(
                                (cid, combined_score, path + [cid])
                            )

                # Re-rank by combined score and keep top_k.
                new_candidates.sort(key=lambda x: x[1], reverse=True)
                candidates = new_candidates[:top_k]

        # Extract leaf (Turn tier) nodes.
        turn_ids: List[int] = []
        scores: List[float] = []
        paths: List[List[int]] = []

        for node_id, score, path in candidates:
            turn_ids.append(node_id)
            scores.append(score)
            paths.append(path)

        # If no candidates, fall back to direct search on Turn tier.
        if not turn_ids and self.tiers[0].size > 0:
            nodes = self.tiers[0].search(query_emb.flatten(), top_k=top_k)
            for n in nodes:
                turn_ids.append(n.node_id)
                scores.append(1.0)
                paths.append([n.node_id])

        result: RetrievalResult = RetrievalResult(turn_ids=turn_ids, scores=scores, paths=paths)
        self._last_retrieval = result
        return result

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def rebuild_all_indexes(self) -> None:
        """Rebuild FAISS indexes for all tiers."""
        for tier in self.tiers:
            tier.rebuild_index()

    def stats(self) -> Dict[str, int]:
        """Return size statistics for all tiers."""
        return {
            self.tier_names[i]: self.tiers[self.num_tiers - 1 - i].size
            for i in range(self.num_tiers)
        }

    def extra_repr(self) -> str:
        return (
            f"tiers={self.num_tiers}, "
            f"embedding_dim={self.embedding_dim}, "
            f"sizes={self.stats()}"
        )
