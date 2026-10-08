# Principles of Online Decision Making

This repository contains code and experiments for an EPFL course on the principles of online decision making, with a focus on multi-armed bandits and regret minimization.

The project is organized as a set of small Python-based assignments exploring how learning agents balance exploration and exploitation in uncertain environments.

## Overview

The code implements several classic online decision-making models and algorithms, including:

- Explore-Then-Commit
- Epsilon-Greedy
- UCB (Upper Confidence Bound)
- Sequential Elimination
- AETC (Adaptive Explore-Then-Commit)
- UCB-Var (variance-aware UCB)

It also includes different stochastic bandit environments such as:

- Bernoulli bandits
- Gaussian bandits
- Uniform bandits
- Pareto bandits
- Student-t bandits

The main objective is to study cumulative regret, estimate optimal arms, and compare the empirical performance of different algorithms under various reward distributions.

## Repository structure

```text
PODM/
├── podm_hw1_code/
│   ├── bandit_algorithm.py
│   ├── bandit_environment.py
│   ├── experiment_1.py
│   ├── plotting.py
│   └── .ipynb_checkpoints/
├── podm_hw2_code/
│   ├── bandit_algorithm.py
│   ├── bandit_environment.py
│   ├── experiment_1.ipynb
│   ├── plotting.py
│   └── .ipynb_checkpoints/
├── figures/
├── LICENSE
└── README.md
```

## Homework 1: basic bandit algorithms

The `podm_hw1_code` folder contains the first set of experiments, focusing on the classic bandit setting with Bernoulli rewards.

This homework compares:

- Explore-Then-Commit (ETC)
- Epsilon-Greedy

The script evaluates cumulative regret over time and tracks arm selection behavior, illustrating the standard exploration-exploitation tradeoff.

## Homework 2: advanced UCB-style bandits

The `podm_hw2_code` folder extends the analysis to more advanced algorithms and more diverse environments.

This assignment studies:

- UCB versus variance-aware UCB
- Sequential elimination methods
- Adaptive exploration strategies
- Performance across different reward distributions

The notebook runs repeated simulations and plots final regret, arm pulls, and reward statistics to compare algorithm quality under uncertainty.

## Core concepts

### Multi-armed bandits

A multi-armed bandit is a sequential decision problem in which the learner repeatedly chooses an action (arm) and receives a random reward. The goal is to maximize cumulative reward or minimize regret relative to the best possible policy.

### Regret

Regret quantifies the performance loss caused by choosing suboptimal arms. In online learning, minimizing cumulative regret is a central objective.

### Exploration vs exploitation

The algorithms in this repository illustrate the central challenge of online decision making:

- Exploration: gather information about uncertain arms
- Exploitation: choose arms with promising observed performance

Different algorithms handle this tradeoff differently, depending on the assumptions and confidence bounds they use.

## Dependencies

This project uses Python with the following standard scientific libraries:

- NumPy
- Matplotlib
- Jupyter Notebook

## How to run

### Homework 1

```bash
cd podm_hw1_code
python experiment_1.py
```

### Homework 2

Open the notebook in Jupyter:

```bash
cd podm_hw2_code
jupyter notebook experiment_1.ipynb
```

## Notes

This repository is intended as a compact educational implementation of online decision-making principles, especially in the context of stochastic bandits and regret-minimization methods. It is a practical companion to the theory taught in the EPFL course on principles of online decision making.

## License

This project is distributed under the MIT license. See [LICENSE](LICENSE) for details.
