"""
Multi-Armed Bandit Algorithm Implementations

This module provides various bandit algorithms, primarily UCB-based variants
with both anytime and fixed-horizon versions. It also includes a Thompson
Sampling implementation for comparison.
"""

from abc import ABC, abstractmethod
from typing import Optional
import copy

import numpy as np


def basic_conf_intervals(error_prob_bound, arms_pulls, arms_estimates_history):
    """compute the confidence intervals in the way UCB does"""
    if np.min(arms_pulls) == 0:
        pulls_nonzero = copy.copy(arms_pulls)
        pulls_nonzero[pulls_nonzero == 0] = 1e-10
        confidence_radii = ...  # PODMexercise
    else:
        confidence_radii = ...  # PODMexercise
    upper_bounds = arms_estimates_history[-1] + confidence_radii
    lower_bounds = arms_estimates_history[-1] - confidence_radii
    return lower_bounds, upper_bounds


def variance_conf_intervals(
    error_prob_bound, arms_pulls, arms_estimates_history, variances_vec
):
    """compute the confidence intervals using known variances"""
    if np.min(arms_pulls) == 0:
        pulls_nonzero = copy.copy(arms_pulls)
        pulls_nonzero[pulls_nonzero == 0] = 1e-10
        confidence_radii = np.sqrt(
            2 * np.log(1 / error_prob_bound) * variances_vec / pulls_nonzero
        )
    else:
        confidence_radii = np.sqrt(
            2 * np.log(1 / error_prob_bound) * variances_vec / arms_pulls
        )
    upper_bounds = arms_estimates_history[-1] + confidence_radii
    lower_bounds = arms_estimates_history[-1] - confidence_radii
    return lower_bounds, upper_bounds


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
        self.last_exploration_step = -1  # default value of -1 when no explo phase
        self.reset()

    def reset(self):
        """Reset algorithm to initial state."""
        self.timestep = 0
        self.arms_pulls = np.zeros(self.n_arms)  # Number of pulls for each arm
        self.arms_estimates_history = [np.zeros(self.n_arms)]
        self.rewards_history = [
            []
            # Store all rewards
            for _ in range(self.n_arms)
        ]
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
        new_estimates[arm] = (((n - 1) * new_estimates[arm]) + reward) / n
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

    def __init__(self, n_arms: int, T_explore: int, random_seed: Optional[int] = None):
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
            if self.last_exploration_step == -1:
                self.last_exploration_step = self.timestep - 1
            return int(
                np.argmax(
                    self.arms_estimates_history[self.n_arms * self.T_explore - 1])
            )


class ExploreThenWeakCommit(BanditAlgorithm):
    """
    Explore-then-Commit algorithm that first explores each arm and then
    commits to the best one.
    """

    def __init__(self, n_arms: int, T_explore: int, random_seed: Optional[int] = None):
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
            if self.last_exploration_step == -1:
                self.last_exploration_step = self.timestep - 1
            return int(np.argmax(self.arms_estimates_history[-1]))


class EpsilonGreedy(BanditAlgorithm):
    """
    Epsilon-Greedy algorithm with a constant epsilon for exploration.
    """

    def __init__(
        self, n_arms: int, epsilon: float = 0.1, random_seed: Optional[int] = None
    ):
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


class UCB(BanditAlgorithm):
    """
    UCB algorithm - standard anytime version.
    """

    def __init__(
        self,
        n_arms: int,
        error_prob_bound: float,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_arms, random_seed)
        self.error_prob_bound = error_prob_bound

    def select_arm(self) -> int:
        """Select arm using UCB formula."""
        if self.timestep < self.n_arms:
            return self.timestep

        _, ucb_values = basic_conf_intervals(
            self.error_prob_bound, self.arms_pulls, self.arms_estimates_history
        )

        return int(np.argmax(ucb_values))

    def get_confidence_intervals(self) -> np.ndarray:
        """Return confidence intervals based on UCB formula."""
        if self.timestep == 0:
            return np.zeros((self.n_arms, 2))

        lower_bounds, upper_bounds = basic_conf_intervals(
            self.error_prob_bound, self.arms_pulls, self.arms_estimates_history
        )
        return np.column_stack([lower_bounds, upper_bounds])


class SequentialElimination(BanditAlgorithm):
    """
    Sequential Elimination algorithm.
    """

    def __init__(
        self, n_arms: int, error_prob_bound: float, random_seed: Optional[int] = None
    ):
        super().__init__(n_arms, random_seed)
        self.error_prob_bound = error_prob_bound
        self.active_arms = list(range(n_arms))
        self.phase = 0  # 0 for exploration/elimination phase, 1 for exploitation phase
        self.exploration_phase_length = 0  # Initial exploration phase length

    def select_arm(self) -> int:
        """Select arm using Sequential Elimination strategy."""
        if self.timestep < self.n_arms:
            return self.timestep
        elif self.phase == 0:
            # Exploration phase
            return self.active_arms[int(np.argmin(self.arms_pulls[self.active_arms]))]
        else:
            # Exploitation phase: always choose the best arm (which should be only one left)
            return self.active_arms[
                int(np.argmax(
                    self.arms_estimates_history[-1][self.active_arms]))
            ]

    def update(self, arm: int, reward: float):
        super().update(arm, reward)
        if self.phase == 0:
            self.exploration_phase_length += 1
            lower_bounds, upper_bounds = basic_conf_intervals(
                self.error_prob_bound, self.arms_pulls, self.arms_estimates_history
            )
            self.active_arms = [
                arm for arm in self.active_arms if ...]  # PODMexercise
            if len(self.active_arms) <= 1:
                if self.last_exploration_step == -1:
                    self.last_exploration_step = self.timestep
                self.phase = 1  # Switch to exploitation phase

    def get_confidence_intervals(self) -> np.ndarray:
        """Return confidence intervals based on UCB formula."""
        if self.timestep == 0:
            return np.zeros((self.n_arms, 2))

        lower_bounds, upper_bounds = basic_conf_intervals(
            self.error_prob_bound, self.arms_pulls, self.arms_estimates_history
        )
        return np.column_stack([lower_bounds, upper_bounds])


class AETC(BanditAlgorithm):
    """
    Explore-Then-Commit, Adaptive exploration time
    """

    def __init__(
        self, n_arms: int, error_prob_bound: float, random_seed: Optional[int] = None
    ):
        super().__init__(n_arms, random_seed)
        self.error_prob_bound = error_prob_bound
        self.exploration_phase = True

    def get_confidence_intervals(self) -> np.ndarray:
        """Return confidence intervals based on UCB formula."""
        if self.timestep == 0:
            return np.zeros((self.n_arms, 2))

        lower_bounds, upper_bounds = basic_conf_intervals(
            self.error_prob_bound, self.arms_pulls, self.arms_estimates_history
        )

        return np.column_stack([lower_bounds, upper_bounds])

    def select_arm(self) -> int:
        """Select an arm based on exploration or commitment phase."""
        if self.exploration_phase:
            # Exploration phase: cycle through arms
            confidence_intervals = self.get_confidence_intervals()
            max_lower = np.max(confidence_intervals[:, 0])
            stop_exploration = np.sum(
                max_lower < confidence_intervals[:, 1]) == 1
            if stop_exploration:
                self.last_exploration_step = self.timestep
                self.exploration_phase = False
            return self.timestep % self.n_arms
        else:
            # Commitment phase: choose the best arm at the end of exploration
            return int(
                np.argmax(
                    self.arms_estimates_history[self.last_exploration_step])
            )


class UCB_Var(BanditAlgorithm):
    """
    UCB modified algorithm with different variances
    """

    def __init__(
        self,
        n_arms: int,
        error_prob_bound: float,
        variances_vec: np.ndarray,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_arms, random_seed)
        self.error_prob_bound = error_prob_bound
        self.variances_vec = variances_vec

    def select_arm(self) -> int:
        """Select arm using UCB"""
        if self.timestep < self.n_arms:
            return self.timestep

        _, upper_bounds = variance_conf_intervals(
            self.error_prob_bound,
            self.arms_pulls,
            self.arms_estimates_history,
            self.variances_vec,
        )

        return int(np.argmax(upper_bounds))

    def get_confidence_intervals(self) -> np.ndarray:
        """Return confidence intervals"""
        if self.timestep == 0:
            return np.zeros((self.n_arms, 2))

        lower_bounds, upper_bounds = variance_conf_intervals(
            self.error_prob_bound,
            self.arms_pulls,
            self.arms_estimates_history,
            self.variances_vec,
        )
        return np.column_stack([lower_bounds, upper_bounds])
