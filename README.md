# RL-From-Scratch : Value-Based Reinforcement Learning Engine

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green.svg)](https://gymnasium.farama.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean](https://img.shields.io/badge/Code%20Style-Strict%20Typing-brightgreen.svg)](#)

A modular, production-grade Reinforcement Learning (RL) engineering engine implemented from fundamental mathematical principles. This repository bridges the theoretical foundations of **Markov Decision Processes (MDP)** and **Temporal Difference (TD) learning** with modern **Deep Q-Networks (DQN)** applied to continuous observation control benchmarks.

---

## Table of Contents

- [Architectural Overview](#architectural-overview)
- [Repository Structure](#repository-structure)
- [Mathematical Foundations](#mathematical-foundations)
  - [1. Bellman Optimality Equation & TD Error](#1-bellman-optimality-equation--td-error)
  - [2. Exploration vs. Exploitation ($\epsilon$-Greedy Schedule)](#2-exploration-vs-exploitation-epsilon-greedy-schedule)
  - [3. Experience Replay (Decorrelation of Transitions)](#3-experience-replay-decorrelation-of-transitions)
  - [4. Fixed Target Network (Mitigating Moving Target Divergence)](#4-fixed-target-network-mitigating-moving-target-divergence)
  - [5. Loss Formulation & Gradient Norm Clipping](#5-loss-formulation--gradient-norm-clipping)
  - [6. MDP Termination vs. Environment Truncation](#6-mdp-termination-vs-environment-truncation)
- [Installation](#installation)
- [Command-Line Interface (CLI) Usage](#command-line-interface-cli-usage)
- [Experimental Results & Empirical Benchmark Analysis](#experimental-results--empirical-benchmark-analysis)
  - [Part A: Tabular Q-Learning (FrozenLake-v1)](#part-a-tabular-q-learning-frozenlake-v1)
  - [Part B: Deep Q-Network (CartPole-v1)](#part-b-deep-q-network-cartpole-v1)
- [Software Engineering & Design Principles](#software-engineering--design-principles)

---

## Architectural Overview

The engine separates discrete tabular dynamic programming from parametric continuous-state approximation while sharing uniform evaluation, visualization, and logging utilities:

```
+-----------------------------------------------------------------------------------+
|                                  CLI (main.py)                                    |
|                 [--mode all | qlearning | dqn | demo]                             |
+-----------------------------------------+-----------------------------------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   v                                             v
       [ Tabular Q-Learning ]                         [ Deep Q-Network (DQN) ]
        (src/q_learning.py)                              (src/dqn_agent.py)
  +---------------------------------+           +---------------------------------+
  | - Pure NumPy Implementation     |           | - PyTorch MLP Q-Network         |
  | - Exact Bellman Updates         |           | - Experience Replay Buffer      |
  | - Unbiased Random Tie-Breaking  |           | - Frozen Target Network         |
  | - Benchmark: FrozenLake-v1      |           | - Benchmark: CartPole-v1        |
  +---------------------------------+           +---------------------------------+
                   |                                             |
                   +----------------------+----------------------+
                                          v
                                   [ src/utils.py ]
                        - Lag-Free Rolling Moving Averages
                        - Dynamic Device Dispatch (CUDA/MPS/CPU)
                        - Publication-Ready Figure Generation
```

---

## Repository Structure

```
rl-from-scratch/
├── .gitignore               # Strict exclusion of caches and binary weights while preserving figures
├── requirements.txt         # Pinned minimum dependencies (Gymnasium, NumPy, PyTorch, Matplotlib)
├── README.md                # Engineering and theoretical documentation
├── main.py                  # Unified CLI entrypoint with multi-modal dispatch
├── src/
│   ├── __init__.py          # Package initialization with semantic versioning (__version__ = "1.0.0")
│   ├── utils.py             # Device detection, lag-free moving average, and plotting routines
│   ├── q_learning.py        # NumPy tabular Q-Learning agent, training, evaluation, and gamma study
│   ├── dqn_agent.py         # PyTorch MLP Q-Network, ReplayBuffer, and DQNAgent class
│   └── train_dqn.py         # DQN training loop on CartPole-v1, early stopping, and evaluation
└── results/                 # Auto-generated experiment outputs (figures, model checkpoints)
    ├── q_learning_deterministic.png
    ├── q_learning_stochastic.png
    ├── q_learning_comparison.png
    ├── dqn_cartpole_rewards.png
    └── dqn_cartpole.pth
```

---

## Mathematical Foundations

### 1. Bellman Optimality Equation & TD Error

Reinforcement learning formulates decision-making as a Markov Decision Process defined by the 5-tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. The expected cumulative return from taking action $a$ in state $s$ under optimal policy $\pi^*$ satisfies the **Bellman Optimality Equation**:

$$Q^*(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \max_{a' \in \mathcal{A}} Q^*(s', a')$$

In model-free Temporal Difference learning, transition dynamics $\mathcal{P}$ and reward distribution $\mathcal{R}$ are unknown. The agent bootstraps estimates from sample trajectories via the **Temporal Difference (TD) error** $\delta_t$:

$$\delta_t = \underbrace{R_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q(S_{t+1}, a')(1 - d_t)}_{\text{TD Target } Y_t} - Q(S_t, A_t)$$

The tabular update rule adjusts the state-action value table according to learning rate $\alpha \in (0, 1]$:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \, \delta_t$$

Where $d_t \in \{0, 1\}$ is the absorbing terminal state indicator.

### 2. Exploration vs. Exploitation ($\epsilon$-Greedy Schedule)

To prevent premature sub-optimal convergence, the agent balances exploration and exploitation using an exponential $\epsilon$-greedy schedule:

$$\pi(a \mid s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|}, & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\epsilon}{|\mathcal{A}|}, & \text{otherwise} \end{cases}$$

The exploration parameter decays exponentially after each episode:

$$\epsilon_{k+1} = \max(\epsilon_{\min}, \epsilon_k \cdot \lambda_{\text{decay}})$$

> **Engineering Detail (Unbiased Tie-Breaking):** Standard `argmax` functions consistently select the lowest index during ties. When multiple actions possess identical maximal Q-values (e.g. during initialization where $Q(s, \cdot) = 0$), deterministic selection causes systematic directional bias. In `src/q_learning.py`, ties are resolved via uniform random sampling over all maximal candidates:
>
> $$\mathcal{A}^*_{\text{candidates}} = \{a \in \mathcal{A} \mid |Q(s, a) - \max_{a'} Q(s, a')| < 10^{-12}\}$$

### 3. Experience Replay (Decorrelation of Transitions)

Sequential rollouts produce non-stationary, highly auto-correlated data vectors $(S_t, A_t, R_{t+1}, S_{t+1})$. Directly feeding consecutive states to a neural network violates the fundamental **Independent and Identically Distributed (i.i.d.)** assumption of Stochastic Gradient Descent (SGD), causing catastrophic forgetting and weight divergence.

The **Experience Replay Buffer** $\mathcal{D}$ stores transitions in a bounded circular buffer:

$$\mathcal{D} = \{(s_i, a_i, r_i, s'_i, d_i)\}_{i=1}^{N}$$

Minibatches of size $B = 64$ are sampled uniformly at random $\mathcal{U}(\mathcal{D})$, decorrelating transitions and enabling sample reuse across gradient iterations.

### 4. Fixed Target Network (Mitigating Moving Target Divergence)

Standard Q-learning with function approximation minimizes the Mean Squared Bellman Error:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta) (1 - d) - Q(s, a; \theta) \right)^2 \right]$$

Updating parameters $\theta$ simultaneously modifies both the prediction $Q(s, a; \theta)$ and the target $Y_t(\theta)$. This "chasing its own tail" dynamic leads to policy oscillation or divergence.

DQN decouples target generation by maintaining a separate **Target Network** parameterized by frozen weights $\theta^-$:

$$Y_t = r + \gamma \max_{a'} Q(s', a'; \theta^-) (1 - d)$$

The weights $\theta^-$ are synchronized periodically every $C$ episodes ($C = 10$ in `train_dqn.py`):

$$\theta^- \leftarrow \theta$$

### 5. Loss Formulation & Gradient Norm Clipping

To provide stability against large outlier Bellman errors, we optimize the **Huber Loss** (Smooth $L_1$ Loss) with threshold $\beta = 1.0$:

$$\mathcal{L}_{\text{Huber}}(\delta) = \begin{cases} \frac{1}{2} \delta^2, & \text{if } |\delta| \le 1 \\ |\delta| - \frac{1}{2}, & \text{otherwise} \end{cases}$$

Gradient backpropagation is further safeguarded via $L_2$ gradient norm clipping:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{10.0}{\|\mathbf{g}\|_2}\right)$$

### 6. MDP Termination vs. Environment Truncation

Modern Gymnasium environments decouple episode termination into two distinct Boolean flags:
- `terminated` (**True MDP Terminal State**): The system entered an absorbing boundary state (e.g. pole fell past angle threshold, cart left rails, or agent fell into a FrozenLake hole). Future expected reward is strictly $0$.
- `truncated` (**Time Horizon Limit**): The episode was truncated due to reaching an artificial ceiling (e.g. 500 steps in CartPole-v1). The state is **not** absorbing; the agent could continue collecting reward.

```python
# Exact Bellman update logic:
done_for_bellman = terminated  # NOT (terminated or truncated)
target_q = reward + (1.0 - float(done_for_bellman)) * gamma * max_next_q
```

Zeroing the target on truncation introduces a catastrophic negative bias that punishes the agent for surviving up to the time limit.

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Recommended: Isolated virtual environment

```bash
# Clone the repository
git clone https://github.com/your-username/rl-from-scratch.git
cd rl-from-scratch

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install exact requirements
pip install -r requirements.txt
```

---

## Command-Line Interface (CLI) Usage

The engine is controlled via `main.py` with the `--mode` flag:

```bash
# 1. Run the entire end-to-end benchmark suite (Tabular Q-Learning + Deep Q-Network)
python main.py --mode all

# 2. Run Tabular Q-Learning experiments (Deterministic, Stochastic, and Gamma sensitivity)
python main.py --mode qlearning

# 3. Train and benchmark the Deep Q-Network on CartPole-v1
python main.py --mode dqn

# 4. Launch real-time visual rendering (requires display / GUI support)
python main.py --mode demo
```

---

## Experimental Results & Empirical Benchmark Analysis

### Part A: Tabular Q-Learning (FrozenLake-v1)

#### 1. Deterministic Gridworld (`is_slippery=False`)
In a deterministic 4x4 gridworld, the transition probability $\mathcal{P}(s' \mid s, a) = 1.0$ for valid adjacent cells. The agent rapidly identifies the shortest risk-free trajectory to the goal $[G]$, converging to a **100.0% evaluation success rate** within 500 episodes.

![Deterministic Q-Learning](results/q_learning_deterministic.png)

*Figure 1: Rapid convergence of tabular Q-learning on deterministic FrozenLake-v1. The moving average (100 episodes) achieves optimal reward plateau without variance.*

```
Optimal Deterministic Policy Grid:
+------+------+------+------+
|  [ → ]|  [ → ]|  [ ↓ ]|  [ ← ]|
+------+------+------+------+
|  [ ↓ ]|  [H] |  [ ↓ ]|  [H] |
+------+------+------+------+
|  [ → ]|  [ ↓ ]|  [ ↓ ]|  [H] |
+------+------+------+------+
|  [H] |  [ → ]|  [ → ]|  [G] |
+------+------+------+------+
```

#### 2. Stochastic Gridworld (`is_slippery=True`)
When the ice is slippery, taking action $a$ only executes with probability $1/3$; with probability $2/3$, the agent slips perpendicularly. Under these stochastic dynamics, the shortest path is fatal because walking along hole borders carries high risk of slipping into an absorbing hole $[H]$.

The tabular agent discovers the conservative safety policy (e.g. intentionally aiming toward exterior boundary walls to cushion random slips), reaching an empirical evaluation success rate between **70% and 82%** (matching theoretical optimality on FrozenLake 4x4).

![Stochastic Q-Learning](results/q_learning_stochastic.png)

*Figure 2: Episodic returns and rolling moving average on stochastic FrozenLake-v1 across 2,500 episodes.*

#### 3. Hyperparameter Sensitivity: Discount Factor ($\gamma$) Study
We evaluated the credit assignment horizon by comparing three discount factors: $\gamma \in \{0.80, 0.95, 0.99\}$.

![Gamma Comparison](results/q_learning_comparison.png)

*Figure 3: Comparative learning curves for $\gamma = 0.80$, $\gamma = 0.95$, and $\gamma = 0.99$ on stochastic FrozenLake-v1.*

- **$\gamma = 0.80$ (Myopic):** The agent excessively discounts rewards that are 6-10 steps away. Because FrozenLake yields a sparse reward ($+1.0$) exclusively at the terminal goal $[G]$, a low discount factor results in near-zero propagated TD targets at the starting states, leading to poor policy formation and high failure rates.
- **$\gamma = 0.95$ & $\gamma = 0.99$ (Far-Sighted):** Allows reward propagation over long Markovian trajectories, providing sufficient signal gradient for the agent to navigate around hazards and converge reliably.

---

### Part B: Deep Q-Network (CartPole-v1)

CartPole-v1 features a continuous 4-dimensional state vector $\mathbf{s} = (x, \dot{x}, \theta, \dot{\theta})$ and a discrete action space $\mathcal{A} = \{0, 1\}$. The environment reward is $+1.0$ per surviving step, with a ceiling of $500.0$. The environment is officially considered **solved** when the average score over 100 consecutive episodes exceeds **475.0**.

![DQN CartPole Convergence](results/dqn_cartpole_rewards.png)

*Figure 4: Deep Q-Network training curve on CartPole-v1. The agent breaks through the exploration plateau and triggers early stopping at $\ge 475.0$ points.*

#### Empirical Performance Metrics:
- **Benchmark Criterion:** Early stopping triggered once rolling 100-episode mean return $\ge 475.0 / 500.0$.
- **Evaluation Benchmark (Greedy $\epsilon = 0$, 100 rollouts):**
  - **Mean Return:** $500.00 \pm 0.00$
  - **Success Rate:** $100\%$ optimal episode horizon.

---

## Software Engineering & Design Principles

This codebase is structured according to senior ML engineering best practices:

1. **Strict Type Annotations:** Full compliance with PEP 484 and PEP 526, leveraging `from __future__ import annotations` and explicit union types (`float | None`).
2. **Deterministic Reproducibility:** Master seed propagation across NumPy pseudo-random generators (`np.random.default_rng(seed)`), PyTorch CPU/CUDA engines, and Gymnasium environments.
3. **Decoupled Architecture:** Strict separation between Agent algorithms (`dqn_agent.py`, `q_learning.py`), Execution Pipelines (`train_dqn.py`), and Support Utilities (`utils.py`).
4. **Defensive Resource Management:** Dynamic compute device selection (CUDA $\to$ Apple Silicon MPS $\to$ CPU), automatic creation of filesystem artifacts via `pathlib.Path`, and guaranteed figure resource release with `plt.close()`.
5. **No Synthetic Shortcuts:** Zero placeholders, fully realized vector mathematics, and native Gymnasium API integration.
