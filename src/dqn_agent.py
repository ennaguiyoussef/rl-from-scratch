"""Deep Q-Network (DQN) Agent Architecture and Experience Replay.

Implements Mnih et al. (2015) Human-level control through deep reinforcement learning.
Includes:
- Multi-Layer Perceptron (MLP) Q-Network with explicit tensor shape tracking.
- Experience Replay Buffer for breaking temporal autocorrelation in sequential MDP transitions.
- Fixed Target Network stabilization to mitigate moving-target divergence.
- Huber Loss (Smooth L1) with gradient norm clipping for robust gradient backpropagation.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import get_device


class QNetwork(nn.Module):
    """Deep Q-Network approximating the optimal action-value function Q*(s, a).

    Architecture:
        Fully Connected MLP: state_dim -> 128 -> 128 -> action_dim
        Activation: Rectified Linear Unit (ReLU) on hidden projections.

    Tensor Dimension Transformations:
        Input state:         [batch_size, state_dim]
        Linear 1 (fc1):      [batch_size, 128]
        Activation (relu1):  [batch_size, 128]
        Linear 2 (fc2):      [batch_size, 128]
        Activation (relu2):  [batch_size, 128]
        Linear 3 (fc3):      [batch_size, action_dim] (unnormalized Q-value per action)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        """Initializes the QNetwork.

        Args:
            state_dim: Dimensionality of continuous observation vector.
            action_dim: Cardinality of discrete action space.
            hidden_dim: Number of neurons per hidden layer (default: 128).
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Performs forward inference of action-value estimates.

        Args:
            state: Observation tensor of shape [batch_size, state_dim].

        Returns:
            torch.Tensor: Q-value projections of shape [batch_size, action_dim].
        """
        # Tensor Shape: [batch_size, state_dim] -> [batch_size, 128]
        x = self.relu1(self.fc1(state))
        # Tensor Shape: [batch_size, 128] -> [batch_size, 128]
        x = self.relu2(self.fc2(x))
        # Tensor Shape: [batch_size, 128] -> [batch_size, action_dim]
        q_values = self.fc3(x)
        return q_values


class ReplayBuffer:
    """Circular Experience Replay Buffer for decorrelating RL transitions.

    Stores transition tuples (s_t, a_t, r_t, s_{t+1}, d_t) and samples uniformly
    to approximate independent and identically distributed (i.i.d.) minibatches
    for stochastic gradient descent updates.
    """

    def __init__(self, capacity: int = 10_000, seed: int | None = None) -> None:
        """Initializes circular memory queue.

        Args:
            capacity: Maximum number of stored transitions before FIFO eviction.
            seed: Optional seed for reproducible uniform sampling.
        """
        self.capacity = capacity
        self.memory: collections.deque[
            tuple[np.ndarray, int, float, np.ndarray, bool]
        ] = collections.deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Appends a new environmental transition to memory.

        Args:
            state: Initial observation array of shape [state_dim].
            action: Action taken (scalar int).
            reward: Environmental scalar reward.
            next_state: Resulting observation array of shape [state_dim].
            done: Terminal absorbing flag for Bellman bootstrapping.
        """
        self.memory.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(
        self,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples a uniform random batch of transitions as typed NumPy arrays.

        Args:
            batch_size: Number of transitions to draw.

        Returns:
            tuple of NumPy arrays:
                - states: shape [batch_size, state_dim], dtype float32
                - actions: shape [batch_size], dtype int64
                - rewards: shape [batch_size], dtype float32
                - next_states: shape [batch_size, state_dim], dtype float32
                - dones: shape [batch_size], dtype float32 (1.0 for terminal, 0.0 otherwise)
        """
        indices = self.rng.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Returns the current transition count in buffer."""
        return len(self.memory)


class DQNAgent:
    """Deep Q-Network Agent with Target Network Stabilization and Huber Loss.

    Maintains two neural networks:
        1. Policy Network (parameterized by theta): Trained actively via Adam.
        2. Target Network (parameterized by theta^-): Frozen and periodically
           synchronized to compute stable Bellman bootstrapped regression targets:
           Y_t = r_t + gamma * max_a' Q(s_{t+1}, a'; theta^-) * (1 - done_t)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        """Initializes the DQNAgent.

        Args:
            state_dim: Observation vector dimensionality.
            action_dim: Discrete action count.
            gamma: Temporal discount factor.
            lr: Adam optimizer learning rate.
            epsilon_start: Initial exploratory probability.
            epsilon_min: Minimum exploration threshold.
            epsilon_decay: Multiplicative decay factor per episode.
            buffer_capacity: Maximum size of experience replay memory.
            batch_size: Minibatch size for gradient updates.
            device: Compute device (CPU/GPU/MPS). Defaults to auto-detection.
            seed: Random seed for action selection and memory sampling.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)

        self.device = device if device is not None else get_device()
        self.rng = np.random.default_rng(seed)

        # Primary Policy Network (theta)
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        # Secondary Frozen Target Network (theta^-)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.sync_target_network()
        self.target_net.eval()

        # Robust loss function: Huber Loss (Smooth L1 Loss)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay memory buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Selects an action using an epsilon-greedy policy with neural network inference.

        Args:
            state: Continuous state vector of shape [state_dim].
            explore: Whether exploratory random actions are allowed.

        Returns:
            int: Selected discrete action index.
        """
        if explore and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, self.action_dim))

        state_tensor = (
            torch.as_tensor(state, dtype=torch.float32, device=self.device)
            .unsqueeze(0)  # Shape: [1, state_dim]
        )

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)  # Shape: [1, action_dim]
            action = int(q_values.argmax(dim=1).item())
        self.policy_net.train()

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Enqueues transition tuple into experience replay memory."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """Samples a minibatch and performs one gradient descent step.

        Tensor Dimensions during Optimization:
            states_t:       [batch_size, state_dim]
            actions_t:      [batch_size, 1]
            rewards_t:      [batch_size, 1]
            next_states_t:  [batch_size, state_dim]
            dones_t:        [batch_size, 1]
            current_q:      [batch_size, 1] (gathered along action index)
            target_q:       [batch_size, 1] (bootstrapped from target network)

        Returns:
            float | None: Scalar loss value if update occurred, else None if warm-up.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Vectorized transfer to compute device with explicit tensor dimension alignment
        states_t = torch.from_numpy(states).to(self.device)                       # [B, state_dim]
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)        # [B, 1]
        rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)        # [B, 1]
        next_states_t = torch.from_numpy(next_states).to(self.device)             # [B, state_dim]
        dones_t = torch.from_numpy(dones).unsqueeze(1).to(self.device)            # [B, 1]

        # Extract predicted Q-value for taken action: Q(s, a; theta)
        # policy_net(states_t) -> [B, action_dim]
        # .gather(1, actions_t) -> [B, 1]
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Bootstrapped Bellman target via Target Network:
        # max_a' Q(s', a'; theta^-) -> [B, 1]
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            # Enforce 0.0 terminal reward absorption when done == 1.0:
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        # Compute Huber Loss (Smooth L1) between prediction and Bellman target
        loss = self.criterion(current_q, target_q)

        # Backpropagation with gradient norm clipping to prevent gradient explosion
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def sync_target_network(self) -> None:
        """Copies parameter weights from Policy Network to Target Network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> float:
        """Applies exponential decay step to exploration rate epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, filepath: str | Path) -> Path:
        """Serializes Policy Network weights to disk.

        Args:
            filepath: Destination file path (e.g. results/dqn_cartpole.pth).

        Returns:
            Path: Absolute path to saved model weights.
        """
        save_path = Path(filepath).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), save_path)
        return save_path

    def load(self, filepath: str | Path) -> None:
        """Loads serialized weights into Policy and Target networks.

        Args:
            filepath: Source model checkpoint path.
        """
        load_path = Path(filepath).resolve()
        if not load_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at: {load_path}")
        state_dict = torch.load(load_path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.sync_target_network()
