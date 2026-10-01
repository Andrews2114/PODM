"""
Multi-Armed Bandit Environment Classes

This module provides various bandit environment implementations with different
reward distributions and configurations.

Note: The environments use a convention where `timestep` counts the number of pulls so that
pre-specified reward matrices are indexed at 0 for the first pull; most other
environments sample first and then increment `timestep`.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BanditEnvironment(ABC):
    """
    Abstract base class for multi-armed bandit environments.
    """

    def __init__(self, n_arms: int, random_seed: Optional[int] = None):
        """
        Initialize bandit environment.

        Args:
            n_arms: Number of arms in the bandit
            random_seed: Random seed for reproducibility
        """
        self.n_arms = n_arms
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.timestep = 0  # Current time step
        self.history = []  # Store (arm, reward) pairs

    @abstractmethod
    def pull_arm(self, arm: int) -> float:
        """
        Pull an arm and return reward.

        Args:
            arm: Arm index to pull (0-indexed)

        Returns:
            Reward from pulling the arm
        """
        pass

    @abstractmethod
    def get_optimal_arm(self) -> int:
        """Return the index of the optimal arm."""
        pass

    @abstractmethod
    def get_arm_means(self) -> np.ndarray:
        """Return the true mean rewards for all arms."""
        pass

    def get_regret(self, arm: int) -> float:
        """
        Get instantaneous regret for pulling given arm.

        Args:
            arm: Arm that was pulled

        Returns:
            Instantaneous regret
        """
        means = self.get_arm_means()
        optimal_mean = np.max(means)
        return optimal_mean - means[arm]

    def reset(self):
        """Reset environment to initial state."""
        self.timestep = 0
        self.history = []
        self.rng = np.random.RandomState(self.random_seed)


class BernoulliBandit(BanditEnvironment):
    """Bernoulli bandit with different success probabilities for each arm."""

    def __init__(self, probabilities: List[float], random_seed: Optional[int] = None):
        """
        Initialize Bernoulli bandit.

        Args:
            probabilities: Success probability for each arm
            random_seed: Random seed for reproducibility
        """
        super().__init__(len(probabilities), random_seed)
        self.probabilities = np.array(probabilities)
        if np.any(self.probabilities < 0) or np.any(self.probabilities > 1):
            raise ValueError('Bernoulli probabilities must be in [0,1]')

    def pull_arm(self, arm: int) -> float:
        """Pull arm and return Bernoulli reward."""
        reward = float(self.rng.binomial(1, self.probabilities[arm]))
        self.history.append((arm, reward))
        self.timestep += 1
        return reward

    def get_optimal_arm(self) -> int:
        return int(np.argmax(self.probabilities))

    def get_arm_means(self) -> np.ndarray:
        return self.probabilities.copy()
