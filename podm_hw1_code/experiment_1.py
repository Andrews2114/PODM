"""
Homework 1 main python file
This file experiments with Explore-Then-Commit and Epsilon Greedy strategies for the multi-armed bandit problem, with a Bernoulli reward.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from bandit_environment import BernoulliBandit
from bandit_algorithm import ExploreThenCommit, EpsilonGreedy
from plotting import finalize_plot


# folder where to save the figures
fig_folder = "figures/"
os.makedirs(fig_folder, exist_ok=True)


# parameters for bandit
# number of arms
n_arms = 2
# suboptimality gap
delta = 0.2
prob_vec = [0.5 - delta / 2, 0.5 + delta / 2]
# time horizon
time_horizon = 2000

# parameters for explore-then-commit
explore_time_etc = 100
seed_etc = 0  # to fix randomness

# parameters for epsilon-greedy
eps = 0.1
seed_e_greedy = 0  # to fix randomness
# try seed 4195


# --------- Explore-Then-Commit -----------

# initialise environment
env1 = BernoulliBandit(probabilities=prob_vec, random_seed=seed_etc)
# initialise bandit algorithm
etc_alg = ExploreThenCommit(
    n_arms=n_arms, T_explore=explore_time_etc, random_seed=seed_etc
)


# set up loop
regret = np.zeros(time_horizon)
for timestep in range(time_horizon):
    # algorithm chooses an arm
    chosen_arm = etc_alg.select_arm()
    # environment provides a reward
    reward = env1.pull_arm(chosen_arm)
    # algorithm updates its internal state
    etc_alg.update(chosen_arm, reward)
    # compute regret
    instant_regret = env1.get_regret(chosen_arm)
    if timestep == 0:
        regret[timestep] = instant_regret
    else:
        regret[timestep] = regret[timestep - 1] + instant_regret


# plot regret
plt.plot(range(time_horizon), regret, label="ETC")
# added dotted vertical line to indicate end of exploration phase
plt.axvline(
    x=n_arms * explore_time_etc, color="r", linestyle="--", label="End of Exploration"
)
finalize_plot(xlabel="Time step", ylabel="Cumulative Regret",
              title="Cumulative Regret (ETC)", path=fig_folder + "etc_cumulative_regret.pdf")


# plot chosen arm
chosen_arms = np.array(etc_alg.arm_history)
plt.scatter(range(time_horizon), chosen_arms, label="Chosen Arm", s=1)
plt.axvline(
    x=n_arms * explore_time_etc, color="r", linestyle="--", label="End of Exploration"
)
plt.yticks(np.arange(n_arms))
finalize_plot(xlabel="Time step", ylabel="Chosen Arm",
              title="Chosen Arm over Time (ETC)", path=fig_folder + "etc_chosen_arm.pdf")


# plot arm estimates over time
for i in range(n_arms):
    plt.plot(range(time_horizon),
             etc_alg.get_arms_estimates_history()[:, i],
             label="Arm" + str(i))
plt.axvline(
    x=n_arms * explore_time_etc, color="r", linestyle="--", label="End of Exploration"
)
finalize_plot(xlabel="Time step", ylabel="Arm reward estimate",
              title="Reward estimate over time (ETC)", path=fig_folder + "etc_estimate_rewards.pdf")


# ------ Epsilon-Greedy ---------------

# initialise environment
env2 = BernoulliBandit(probabilities=prob_vec, random_seed=seed_e_greedy)
# initialise bandit algorithm
e_greedy_alg = EpsilonGreedy(
    n_arms=n_arms, epsilon=eps, random_seed=seed_e_greedy)


# set up loop
regret = np.zeros(time_horizon)
for timestep in range(time_horizon):
    # algorithm chooses an arm
    chosen_arm = e_greedy_alg.select_arm()
    # environment provides a reward
    reward = env2.pull_arm(chosen_arm)
    # algorithm updates its internal state
    e_greedy_alg.update(chosen_arm, reward)
    # compute regret
    instant_regret = env2.get_regret(chosen_arm)
    if timestep == 0:
        regret[timestep] = instant_regret
    else:
        regret[timestep] = regret[timestep - 1] + instant_regret


# plot regret
plt.plot(range(time_horizon), regret, label="eps-greedy")
finalize_plot(xlabel="Time step", ylabel="Cumulative Regret",
              title="Cumulative Regret (e_greedy)", path=fig_folder + "e_greedy_cumulative_regret.pdf")


# plot chosen arm
chosen_arms = np.array(e_greedy_alg.arm_history)
plt.scatter(
    [i for i in range(time_horizon)
     if i not in e_greedy_alg.exploration_history],
    [
        chosen_arms[i]
        for i in range(time_horizon)
        if i not in e_greedy_alg.exploration_history
    ],
    label="estimated best arm",
    s=1,
)
plt.scatter(
    e_greedy_alg.exploration_history,
    [chosen_arms[i] for i in e_greedy_alg.exploration_history],
    label="random exploration",
    s=1,
    c="red",
)
plt.yticks(np.arange(n_arms))
finalize_plot(xlabel="Time step", ylabel="Chosen Arm",
              title="Chosen Arm over Time (e_greedy)", path=fig_folder + "e_greedy_chosen_arm.pdf")


# plot arm estimates over time
for i in range(n_arms):
    plt.plot(range(time_horizon),
             e_greedy_alg.get_arms_estimates_history()[:, i],
             label="Arm" + str(i))
finalize_plot(xlabel="Time step", ylabel="Arm reward estimate",
              title="Reward estimate over time (e_greedy)", path=fig_folder + "e_greedy_reward_estimate.pdf")
