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
        self.reward_interval = (None, None)  # min and max possible rewards

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
        self.interval = (0, 1)
        if np.any(self.probabilities < 0) or np.any(self.probabilities > 1):
            raise ValueError("Bernoulli probabilities must be in [0,1]")

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


class GaussianBandit(BanditEnvironment):
    """Gaussian bandit with different means and variances for each arm."""

    def __init__(
        self,
        means: List[float] | np.ndarray,
        variances: List[float] | np.ndarray,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Gaussian bandit.

        Args:
            means: Mean reward for each arm
            variances: Variance of rewards for each arm
            random_seed: Random seed for reproducibility
        """
        super().__init__(len(means), random_seed)
        self.means = np.array(means)
        self.variances = np.array(variances)
        self.interval = (None, None)
        if np.any(self.variances < 0):
            raise ValueError("Variances must be non-negative")

    def pull_arm(self, arm: int) -> float:
        """Pull arm and return Gaussian reward."""
        reward = ...  # PODMexercise
        self.history.append((arm, reward))
        self.timestep += 1
        return reward

    def get_optimal_arm(self) -> int:
        return ...  # PODMexercise

    def get_arm_means(self) -> np.ndarray:
        return self.means.copy()


class UniformBandit(BanditEnvironment):
    """Uniform bandit with different [a,b] intervals for each arm."""

    def __init__(self, intervals: List[tuple], random_seed: Optional[int] = None):
        """
        Initialize Uniform bandit.

        Args:
            intervals: List of (low, high) tuples for each arm
            random_seed: Random seed for reproducibility
        """
        super().__init__(len(intervals), random_seed)
        self.intervals = intervals
        self.means = np.array([(high + low) / 2 for low, high in intervals])
        self.interval = (
            min([interv[0] for interv in intervals]),
            max([interv[1] for interv in intervals]),
        )

    def pull_arm(self, arm: int) -> float:
        """Pull arm and return Uniform reward."""
        self.timestep += 1
        low, high = self.intervals[arm]
        reward = ...  # PODMexercise
        self.history.append((arm, reward))
        return reward

    def get_optimal_arm(self) -> int:
        return ...  # PODMexercise

    def get_arm_means(self) -> np.ndarray:
        return self.means.copy()


class ParetoBandit(BanditEnvironment):
    """Heavy-tailed Pareto bandit with different shape parameters for each arm."""

    def __init__(
        self,
        shapes: List[float],
        scales: List[float],
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Pareto bandit.

        Args:
            shapes: Shape parameter (alpha) for each arm
            scales: Scale parameter (x_m) for each arm
            random_seed: Random seed for reproducibility
        """
        super().__init__(len(shapes), random_seed)
        self.shapes = np.array(shapes)
        self.scales = np.array(scales)
        if np.any(self.shapes <= 0) or np.any(self.scales <= 0):
            raise ValueError("Pareto shapes and scales must be positive")
        # Mean exists only if alpha > 1
        self.means = np.array(
            [
                scale * shape / (shape - 1) if shape > 1 else np.inf
                for shape, scale in zip(shapes, scales)
            ]
        )
        self.interval = (None, None)

    def pull_arm(self, arm: int) -> float:
        """Pull arm and return Pareto reward."""
        reward = self.rng.pareto(self.shapes[arm]) * self.scales[arm]
        self.history.append((arm, reward))
        self.timestep += 1
        return reward

    def get_optimal_arm(self) -> int:
        finite_means = self.means[np.isfinite(self.means)]
        if len(finite_means) == 0:
            return 0  # Arbitrary choice when all means are infinite
        return int(np.argmax(np.where(np.isfinite(self.means), self.means, -np.inf)))

    def get_arm_means(self) -> np.ndarray:
        return self.means.copy()


class StudentTBandit(BanditEnvironment):
    """Heavy-tailed Student-t bandit with different degrees of freedom for each arm."""

    def __init__(
        self,
        dfs: List[float],
        locs: List[float],
        scales: List[float],
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Student-t bandit.

        Args:
            dfs: Degrees of freedom for each arm
            locs: Location parameter for each arm
            scales: Scale parameter for each arm
            random_seed: Random seed for reproducibility
        """
        super().__init__(len(dfs), random_seed)
        self.dfs = np.array(dfs)
        self.locs = np.array(locs)
        self.scales = np.array(scales)
        if np.any(self.scales < 0):
            raise ValueError("Scales must be non-negative")
        # Mean is loc if df > 1, undefined otherwise
        self.means = np.where(self.dfs > 1, self.locs, np.nan)
        self.interval = (None, None)

    def pull_arm(self, arm: int) -> float:
        """Pull arm and return Student-t reward."""
        # Use numpy Generator.standard_t for sampling to avoid SciPy rng compatibility issues.
        # standard_t returns samples centered at 0 with scale=1.
        reward = self.locs[arm] + self.scales[arm] * self.rng.standard_t(self.dfs[arm])
        self.history.append((arm, reward))
        self.timestep += 1
        return reward

    def get_optimal_arm(self) -> int:
        finite_means = self.means[~np.isnan(self.means)]
        if len(finite_means) == 0:
            return 0  # Arbitrary choice when all means are undefined
        return int(np.nanargmax(self.means))

    def get_arm_means(self) -> np.ndarray:
        return self.means.copy()
