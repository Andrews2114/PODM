"""
Multi-Armed Bandit Algorithm Implementations
"""


from abc import ABC, abstractmethod
from typing import Optional
import copy

import numpy as np


class BanditAlgorithm(ABC):
    """
    Abstract base class for multi-armed bandit algorithms.
    """

    def __init__(self, n_arms: int, random_seed: Optional[int] = None):
        """
        Initialize bandit algorithm.

        Args:
            n_arms: Number of arms
            random_seed: Random seed for reproducibility
        """
        self.n_arms = n_arms
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.reset()

    def reset(self):
        """Reset algorithm to initial state."""
        self.timestep = 0
        self.arms_pulls = np.zeros(self.n_arms)  # Number of pulls for each arm
        self.arms_estimates_history = [np.zeros(self.n_arms)]
        self.rewards_history = [[]
                                # Store all rewards
                                for _ in range(self.n_arms)]
        self.arm_history = []  # Store arm choices

    @abstractmethod
    def select_arm(self) -> int:
        """
        Select an arm to pull.

        Returns:
            Index of selected arm
        """
        pass

    def update(self, arm: int, reward: float):
        """
        Update algorithm with observed reward.

        Args:
            arm: Arm that was pulled
            reward: Observed reward
        """
        self.timestep += 1
        self.arms_pulls[arm] += 1
        self.rewards_history[arm].append(reward)
        self.arm_history.append(arm)

        # Update estimated value using incremental average
        n = self.arms_pulls[arm]
        new_estimates = copy.copy(self.arms_estimates_history[-1])
        new_estimates[arm] = ((((n - 1) * new_estimates[arm]) + reward) / n)
        self.arms_estimates_history.append(new_estimates)

    def get_arms_estimates_history(self) -> np.ndarray:
        """Get current arm value estimates."""
        return np.array(self.arms_estimates_history)[1:]

    def get_confidence_intervals(self) -> Optional[np.ndarray]:
        """
        Get current confidence intervals. Optional for all algorithms.

        # Reason for not making this an abstract method:
        # Not all bandit algorithms have a standard or useful concept of a confidence
        # interval (e.g., Thompson Sampling), so it's a useful, but not mandatory,
        # method to implement. Providing a default non-abstract method allows for
        # optional implementation by subclasses that need it, without forcing it
        # on those that don't.
        """
        return None


class ExploreThenCommit(BanditAlgorithm):
    """
    Explore-then-Commit algorithm that first explores each arm and then
    commits to the best one.
    """

    def __init__(self, n_arms: int, T_explore: int,
                 random_seed: Optional[int] = None):
        """
        Initialize Explore-then-Commit.

        Args:
            n_arms: Number of arms
            T_explore: Number of exploration pulls for each arm
            random_seed: Random seed for reproducibility
        """
        self.T_explore = T_explore
        super().__init__(n_arms, random_seed)

    def select_arm(self) -> int:
        """Select an arm based on exploration or commitment phase."""
        if self.timestep < self.n_arms * self.T_explore:
            # Exploration phase: cycle through arms
            return self.timestep % self.n_arms
        else:
            # Commitment phase: choose the best arm at the end of exploration
            return int(np.argmax(self.arms_estimates_history[self.n_arms * self.T_explore - 1]))


class EpsilonGreedy(BanditAlgorithm):
    """
    Epsilon-Greedy algorithm with a constant epsilon for exploration.
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1,
                 random_seed: Optional[int] = None):
        """
        Initialize Epsilon-Greedy.

        Args:
            n_arms: Number of arms
            epsilon: Probability of exploration
            random_seed: Random seed for reproducibility
        """
        self.epsilon = epsilon
        self.exploration_history = []
        super().__init__(n_arms, random_seed)

    def select_arm(self) -> int:
        """Select an arm based on epsilon-greedy policy."""
        if self.rng.random() < self.epsilon:
            # Explore: choose a random arm
            self.exploration_history.append(self.timestep)
            return int(self.rng.integers(self.n_arms))
        else:
            # Exploit: choose the best-performing arm
            return int(np.argmax(self.arms_estimates_history[-1]))
