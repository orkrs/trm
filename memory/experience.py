from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger


class EbbinghausForgettingCurve:
    """Modified Ebbinghaus forgetting curve for memory node weight management.

    The retention (forgetting curve) value for a memory node:

        R(t, S) = theta1 * exp(-t / (theta2 * S)) + theta3

    where:
        t = elapsed steps since last access
        S = node strength (increases with each successful retrieval)
        theta1, theta2, theta3 = curve parameters

    Higher R means the memory is more retained and less likely to be
    evicted.  On successful retrieval (high reward), the node strength
    is increased, making it decay more slowly.  On low reward, strength
    is decreased.

    This creates an RL feedback loop: useful memories are reinforced
    and retained longer; irrelevant memories decay and become eviction
    candidates.

    Reference: TRM-Bank v3.0 Section 3 (Ebbinghaus RL Memory).
    """

    def __init__(
        self,
        theta1: float = 1.0,
        theta2: float = 1.0,
        theta3: float = 0.05,
        strength_increment: float = 0.1,
        strength_decay: float = 0.05,
    ) -> None:
        self.theta1: float = theta1
        self.theta2: float = theta2
        self.theta3: float = theta3
        self.strength_increment: float = strength_increment
        self.strength_decay: float = strength_decay

    def retention(self, t: float, strength: float) -> float:
        """Compute the retention value R(t, S).

        Args:
            t: Elapsed steps since last access.
            strength: Current node strength S.

        Returns:
            Retention value in [theta3, theta1 + theta3].
        """
        if strength < 1e-12:
            return self.theta3
        return float(self.theta1 * np.exp(-t / (self.theta2 * strength)) + self.theta3)

    def update_strength(self, strength: float, reward: float, threshold: float = 0.5) -> float:
        """RL-based strength update based on retrieval reward.

        Args:
            strength: Current node strength.
            reward: Reward signal from the retrieval (e.g. avg log-prob).
            threshold: Reward threshold for positive reinforcement.

        Returns:
            Updated strength value.
        """
        if reward > threshold:
            return strength + self.strength_increment
        return max(0.0, strength - self.strength_decay)

    def __repr__(self) -> str:
        return (
            f"EbbinghausForgettingCurve(theta1={self.theta1}, "
            f"theta2={self.theta2}, theta3={self.theta3}, "
            f"inc={self.strength_increment}, dec={self.strength_decay})"
        )


@dataclass
class TurnExperience:
    """Experience from one inference generation turn.

    Attributes:
        prompt_embedding: Pooled embedding of the prompt (D,) for memory storage.
        hidden_for_lprm: The hidden state (1, D) used at the LPRM auction step.
        module_name: Name of the winning cognitive module.
        reward: Actual outcome signal (e.g. average log-prob of generated tokens).
        prompt_ids: Full prompt token ids for offline training.
        generated_ids: Generated token ids for offline training.
        step_module_map: Optional mapping of decode step -> module name.
        strength: Forgetting-curve strength (increases with useful retrievals).
        push_step: Global step counter value when added to the replay buffer.
    """
    prompt_embedding: np.ndarray
    hidden_for_lprm: torch.Tensor
    module_name: str
    reward: float
    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor
    step_module_map: Dict[int, str] = field(default_factory=dict)
    strength: float = 1.0
    push_step: int = 0


class ExperienceReplay:
    """Ebbinghaus-forgetting-curve replay buffer for turn experiences.

    Stores recent generation experiences and provides sampling for
    offline LPRM training.  Eviction and sampling are governed by
    the Ebbinghaus forgetting curve rather than raw FIFO.

    Args:
        capacity: Maximum number of experiences to store.
        ebbinghaus: EbbinghausForgettingCurve instance (created
                    automatically if not provided).
    """

    def __init__(
        self,
        capacity: int = 10000,
        ebbinghaus: Optional[EbbinghausForgettingCurve] = None,
    ) -> None:
        self.capacity: int = capacity
        self._ebbinghaus: EbbinghausForgettingCurve = (
            ebbinghaus or EbbinghausForgettingCurve()
        )
        self._buffer: List[TurnExperience] = []
        self._position: int = 0
        self._global_step: int = 0

    def push(self, exp: TurnExperience) -> None:
        """Add an experience to the buffer.

        If the buffer is full, the experience with the lowest Ebbinghaus
        retention value is evicted.
        """
        exp.push_step = self._global_step
        self._global_step += 1

        if len(self._buffer) < self.capacity:
            self._buffer.append(exp)
        else:
            evict_idx: int = self._find_lowest_retention()
            self._buffer[evict_idx] = exp

    def sample(self, batch_size: int) -> List[TurnExperience]:
        """Sample experiences weighted by Ebbinghaus retention.

        Higher-retention experiences (more recently / frequently accessed)
        are more likely to be sampled.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            List of sampled TurnExperience objects.
        """
        if not self._buffer:
            return []
        k = min(batch_size, len(self._buffer))
        weights: np.ndarray = self._compute_retention_weights()
        indices = np.random.choice(
            len(self._buffer), size=k, replace=False, p=weights
        )
        return [self._buffer[i] for i in indices]

    def _find_lowest_retention(self) -> int:
        """Find the index of the experience with the lowest retention."""
        current_step: int = self._global_step
        min_ret: float = float("inf")
        min_idx: int = 0
        for i, exp in enumerate(self._buffer):
            t: int = current_step - exp.push_step
            ret: float = self._ebbinghaus.retention(t, exp.strength)
            if ret < min_ret:
                min_ret = ret
                min_idx = i
        return min_idx

    def _compute_retention_weights(self) -> np.ndarray:
        """Compute normalised retention weights for all buffer entries."""
        current_step: int = self._global_step
        weights: List[float] = []
        for exp in self._buffer:
            t: int = current_step - exp.push_step
            ret: float = self._ebbinghaus.retention(t, exp.strength)
            weights.append(max(ret, 0.0))
        w: np.ndarray = np.array(weights, dtype=np.float64)
        w_sum: float = w.sum()
        if w_sum < 1e-12:
            return np.ones_like(w) / len(w)
        return w / w_sum

    def update_experience_strength(self, idx: int, reward: float) -> None:
        """Apply an RL strength update to the experience at index ``idx``.

        Args:
            idx: Index of the experience in the buffer.
            reward: Reward signal from the task.
        """
        if 0 <= idx < len(self._buffer):
            exp: TurnExperience = self._buffer[idx]
            exp.strength = self._ebbinghaus.update_strength(exp.strength, reward)

    def get_retention_values(self) -> List[float]:
        """Return current retention values for all buffer entries.

        Useful for debugging / monitoring forgetting curve behaviour.
        """
        current_step: int = self._global_step
        return [
            self._ebbinghaus.retention(
                current_step - exp.push_step, exp.strength
            )
            for exp in self._buffer
        ]

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, idx: int) -> TurnExperience:
        return self._buffer[idx]

    @property
    def ebbinghaus(self) -> EbbinghausForgettingCurve:
        return self._ebbinghaus
