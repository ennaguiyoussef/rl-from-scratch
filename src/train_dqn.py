"""Training and Evaluation Pipelines for Deep Q-Network on CartPole-v1.

Features:
- Rigorous handling of Gymnasium termination vs truncation semantics in the
  Bellman optimality equation (bootstrapping non-absorbing state transitions).
- Periodic target network parameter synchronization (target_update_freq).
- Early stopping upon achieving the official benchmark solved criterion (MA >= 475.0 / 500.0).
- Automatic model checkpoint serialization and training trajectory visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np

from src.dqn_agent import DQNAgent
from src.utils import get_device, get_results_dir, plot_training_rewards


def train_dqn_cartpole(
    env_name: str = "CartPole-v1",
    n_episodes: int = 600,
    batch_size: int = 64,
    target_update_freq: int = 10,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    buffer_capacity: int = 10_000,
    early_stop_score: float = 475.0,
    seed: int = 42,
) -> tuple[DQNAgent, list[float]]:
    """Trains a Deep Q-Network agent on CartPole-v1 with early stopping.

    Gymnasium Transition Semantics Note:
        In CartPole-v1, an episode terminates either when the pole angle/cart position
        exceeds stability bounds (terminated=True, true absorbing failure state) or
        when the step count reaches the artificial environment horizon (truncated=True).
        In the Bellman update:
            Target = r + gamma * max_a' Q(s', a') * (1 - terminated)
        We explicitly use `terminated` (and NOT `truncated`) as the absorption flag.
        Treating truncation as terminal would falsely penalize the agent for surviving
        until the episode horizon.

    Args:
        env_name: Gymnasium environment id ("CartPole-v1").
        n_episodes: Maximum training episodes.
        batch_size: Minibatch size sampled from Experience Replay.
        target_update_freq: Frequency (in episodes) for copying policy_net to target_net.
        gamma: Discount factor.
        lr: Learning rate for Adam optimizer.
        epsilon_start: Starting exploratory probability.
        epsilon_min: Exploration floor.
        epsilon_decay: Exploration decay multiplier per episode.
        buffer_capacity: Capacity of ReplayBuffer deque.
        early_stop_score: Rolling 100-episode average threshold for early convergence.
        seed: Random seed for reproducibility.

    Returns:
        tuple[DQNAgent, list[float]]: Trained DQNAgent and episodic return history.
    """
    device = get_device()
    print(f"[*] Initializing DQN Training on {env_name} using device: {device}")

    env = gym.make(env_name)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        lr=lr,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )

    rewards_history: list[float] = []
    results_dir = get_results_dir()
    checkpoint_path = results_dir / "dqn_cartpole.pth"
    plot_filename = "dqn_cartpole_rewards.png"

    solved_episode: int | None = None

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed + episode if seed is not None else None)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store transition: 'terminated' marks true MDP terminal absorption;
            # 'truncated' is artificial timeout and must NOT zero future discounted rewards.
            agent.store_transition(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                done=terminated,
            )

            # Optimization step on sampled replay minibatch
            agent.train_step()

            state = next_state
            episode_reward += float(reward)
            done = terminated or truncated

        # Decay exploration rate after each episode
        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        # Periodic synchronization of target network
        if episode % target_update_freq == 0:
            agent.sync_target_network()

        # Compute moving average across the last 100 episodes
        window_rewards = rewards_history[-100:]
        moving_avg = float(np.mean(window_rewards))

        if episode % 20 == 0 or episode == 1:
            print(
                f"Episode {episode:03d}/{n_episodes} | "
                f"Reward: {episode_reward:5.1f} | "
                f"MA (100): {moving_avg:5.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Replay: {len(agent.memory):5d}"
            )

        # Early stopping verification
        if len(rewards_history) >= 100 and moving_avg >= early_stop_score:
            print(
                f"\n[SUCCESS] Environment SOLVED at episode {episode}! "
                f"Rolling 100-episode mean return: {moving_avg:.2f} >= {early_stop_score:.1f}"
            )
            solved_episode = episode
            break

    env.close()

    # Persist model weights
    saved_weights_path = agent.save(checkpoint_path)
    print(f"[+] Model checkpoint persisted to: {saved_weights_path}")

    # Plot and save learning trajectory
    title_suffix = (
        f" (Solved @ Ep {solved_episode})" if solved_episode else f" ({len(rewards_history)} eps)"
    )
    saved_figure_path = plot_training_rewards(
        rewards=rewards_history,
        title=f"DQN Convergence on CartPole-v1{title_suffix}",
        save_filename=plot_filename,
        window=100,
        target_score=early_stop_score,
    )
    print(f"[+] Training rewards plot saved to: {saved_figure_path}")

    return agent, rewards_history


def evaluate_dqn_cartpole(
    agent: DQNAgent,
    env_name: str = "CartPole-v1",
    n_eval_episodes: int = 100,
    seed: int = 123,
) -> tuple[float, float]:
    """Evaluates DQNAgent policy deterministically (epsilon = 0) without exploration.

    Args:
        agent: Trained DQNAgent instance.
        env_name: Gymnasium environment identifier.
        n_eval_episodes: Number of benchmark evaluation rollouts.
        seed: Random seed for evaluation episodes.

    Returns:
        tuple[float, float]: Empirical mean and standard deviation of episodic returns.
    """
    env = gym.make(env_name)
    eval_returns: list[float] = []

    for ep in range(n_eval_episodes):
        state, _ = env.reset(seed=seed + ep if seed is not None else None)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated

        eval_returns.append(ep_reward)

    env.close()

    mean_score = float(np.mean(eval_returns))
    std_score = float(np.std(eval_returns))

    print("\n" + "=" * 65)
    print(f"DQN EVALUATION BENCHMARK ({env_name})")
    print(f"Episodes:           {n_eval_episodes}")
    print(f"Mean Return:        {mean_score:.2f} / 500.00")
    print(f"Std Deviation:      {std_score:.2f}")
    print(f"Min / Max Return:   {np.min(eval_returns):.1f} / {np.max(eval_returns):.1f}")
    print("=" * 65)

    return mean_score, std_score
