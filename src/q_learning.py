"""Tabular Q-Learning Implementation in Pure NumPy.

Implements model-free temporal difference value-based reinforcement learning
for discrete state and action spaces on Gymnasium Toy-Text environments (FrozenLake-v1).
Features strict epsilon-greedy action selection with unbiased random tie-breaking,
exact Bellman temporal difference updates, and comprehensive evaluation benchmarks.
"""

from __future__ import annotations

from typing import Sequence
import gymnasium as gym
import numpy as np

from src.utils import plot_gamma_comparison, plot_training_rewards


class QLearningAgent:
    """Model-Free Off-Policy Temporal Difference Q-Learning Agent.

    Maintains a tabular state-action value function Q: S x A -> R.
    Updates are governed by the Bellman Optimality Equation:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') * (1 - terminated) - Q(s, a)]
    where (1 - terminated) enforces terminal state absorption with zero expected future value.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999,
        seed: int | None = None,
    ) -> None:
        """Initializes the QLearningAgent.

        Args:
            n_states: Cardinality of the discrete observation space |S|.
            n_actions: Cardinality of the discrete action space |A|.
            alpha: Learning rate parameter in (0, 1].
            gamma: Temporal discount factor in [0, 1].
            epsilon_start: Initial exploration probability.
            epsilon_min: Lower bound for epsilon exploration floor.
            epsilon_decay: Multiplicative decay coefficient per episode.
            seed: Optional integer seed for pseudo-random number generator.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.rng = np.random.default_rng(seed)
        # Initialize tabular Q-values uniformly at zero: shape [|S|, |A|]
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def select_action(self, state: int, explore: bool = True) -> int:
        """Selects an action according to an epsilon-greedy policy with random tie-breaking.

        When exploring, selects uniformly at random from A.
        When exploiting, identifies all actions achieving max_a Q(s, a) within floating point
        tolerance and samples uniformly among candidate ties to prevent directional exploration bias.

        Args:
            state: Current discrete environment state s in {0, ..., |S| - 1}.
            explore: Flag determining whether stochastic exploration is active.

        Returns:
            int: Discrete action index a in {0, ..., |A| - 1}.
        """
        if explore and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, self.n_actions))

        state_q = self.q_table[state]
        max_q = np.max(state_q)
        # Unbiased random tie-breaking among all maximal action values
        best_actions = np.flatnonzero(np.isclose(state_q, max_q, atol=1e-12))
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> float:
        """Executes the standard Bellman Temporal Difference (TD) update rule.

        Calculates:
            Target = r + gamma * max_a' Q(s', a') * (1 - terminated)
            TD_error = Target - Q(s, a)
            Q(s, a) <- Q(s, a) + alpha * TD_error

        Args:
            state: Current state s.
            action: Selected action a.
            reward: Immediate scalar reward r.
            next_state: Resulting state s'.
            terminated: True if transition reached a true MDP absorbing terminal state.

        Returns:
            float: Scalar temporal difference error (delta).
        """
        # If state is absorbing terminal (e.g. Hole or Goal in FrozenLake), future value is 0.0
        future_val = 0.0 if terminated else float(np.max(self.q_table[next_state]))
        target = float(reward) + self.gamma * future_val
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> float:
        """Applies exponential decay schedule to exploration rate epsilon.

        Returns:
            float: Decayed exploration rate bounded below by epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon


def train_q_learning(
    env_name: str = "FrozenLake-v1",
    is_slippery: bool = False,
    n_episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.9995,
    map_name: str = "4x4",
    seed: int = 42,
) -> tuple[QLearningAgent, list[float]]:
    """Trains a QLearningAgent on FrozenLake-v1.

    Args:
        env_name: Gymnasium environment identifier.
        is_slippery: If True, transition transitions are stochastic (slippery ice).
        n_episodes: Total training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon_start: Initial exploration probability.
        epsilon_min: Minimum exploration floor.
        epsilon_decay: Exploration decay multiplier per episode.
        map_name: FrozenLake grid configuration ("4x4" or "8x8").
        seed: Random seed for reproducibility.

    Returns:
        tuple[QLearningAgent, list[float]]: The trained agent and history of episodic returns.
    """
    env = gym.make(env_name, is_slippery=is_slippery, map_name=map_name)
    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    rewards_history: list[float] = []

    for episode in range(n_episodes):
        # Deterministic episodic initialization with offset seed
        state, _ = env.reset(seed=seed + episode if seed is not None else None)
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Bellman update uses terminated (absorbing terminal state), not truncation
            agent.update(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                terminated=terminated,
            )

            state = next_state
            episode_reward += float(reward)
            done = terminated or truncated

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

    env.close()
    return agent, rewards_history


def evaluate_q_learning(
    agent: QLearningAgent,
    env_name: str = "FrozenLake-v1",
    is_slippery: bool = False,
    n_episodes: int = 100,
    map_name: str = "4x4",
    seed: int = 123,
) -> float:
    """Evaluates agent policy performance greedily (epsilon = 0) without exploration.

    Args:
        agent: Trained QLearningAgent instance.
        env_name: Gymnasium environment identifier.
        is_slippery: Environment transition stochasticity.
        n_episodes: Number of test evaluation episodes.
        map_name: FrozenLake grid size.
        seed: Evaluation random seed.

    Returns:
        float: Success rate expressed as a percentage in [0.0, 100.0].
    """
    env = gym.make(env_name, is_slippery=is_slippery, map_name=map_name)
    successes = 0

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep if seed is not None else None)
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated

        if ep_reward > 0.0:
            successes += 1

    env.close()
    success_rate = (successes / n_episodes) * 100.0
    return success_rate


def display_optimal_policy(agent: QLearningAgent, map_name: str = "4x4") -> str:
    """Visualizes the derived greedy policy on the 4x4 FrozenLake grid in the terminal.

    Gymnasium FrozenLake Action Mapping:
        0: Left  (←)
        1: Down  (↓)
        2: Right (→)
        3: Up    (↑)

    Args:
        agent: Trained QLearningAgent containing populated Q-table.
        map_name: Map identifier (defaults to "4x4").

    Returns:
        str: Multi-line string representation of the grid policy.
    """
    # Unicode directional arrows matching action indices
    action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    # Canonical 4x4 FrozenLake map layout definition
    # 0: S (Start), 5,7,11,12: H (Hole), 15: G (Goal), others: F (Frozen)
    holes = {5, 7, 11, 12}
    goal = 15

    lines: list[str] = []
    header = f"=== Optimal Greedy Policy (FrozenLake {map_name}) ==="
    lines.append(header)
    lines.append("+" + "------+" * 4)

    for r in range(4):
        row_str = "|"
        for c in range(4):
            state = r * 4 + c
            if state in holes:
                cell = "  [H] "
            elif state == goal:
                cell = "  [G] "
            else:
                best_action = agent.select_action(state, explore=False)
                symbol = action_symbols.get(best_action, "?")
                cell = f"  [ {symbol} ]"
            row_str += cell + "|"
        lines.append(row_str)
        lines.append("+" + "------+" * 4)

    policy_display = "\n".join(lines)
    print(policy_display)
    return policy_display


def run_gamma_experiment(
    gammas: Sequence[float] = (0.8, 0.95, 0.99),
    n_episodes: int = 2500,
    seed: int = 42,
) -> dict[float, list[float]]:
    """Evaluates the sensitivity of Q-Learning to discount factor gamma in stochastic settings.

    Trains multiple agents on stochastic FrozenLake-v1 (is_slippery=True) across
    specified discount factors and plots the comparative learning curves.

    Args:
        gammas: Sequence of discount factors to benchmark.
        n_episodes: Training iterations per gamma configuration.
        seed: Random seed for repeatability.

    Returns:
        dict[float, list[float]]: Mapping of gamma value to reward history.
    """
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SENSITIVITY STUDY: DISCOUNT FACTOR (GAMMA)")
    print("=" * 70)

    results_by_gamma: dict[float, list[float]] = {}

    for gamma in gammas:
        print(f"[*] Training Q-Learning agent with gamma = {gamma:.2f} ({n_episodes} episodes)...")
        agent, rewards = train_q_learning(
            is_slippery=True,
            n_episodes=n_episodes,
            alpha=0.1,
            gamma=gamma,
            epsilon_start=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.9995,
            seed=seed,
        )
        results_by_gamma[gamma] = rewards
        eval_acc = evaluate_q_learning(agent, is_slippery=True, n_episodes=100, seed=seed)
        print(f"    -> Evaluation Success Rate (gamma={gamma:.2f}): {eval_acc:.1f}%")

    saved_plot = plot_gamma_comparison(
        results_by_gamma=results_by_gamma,
        save_filename="q_learning_comparison.png",
        window=100,
    )
    print(f"[+] Comparative gamma figure saved to: {saved_plot}")
    return results_by_gamma


def run_q_learning_pipeline() -> None:
    """Executes the complete Tabular Q-Learning training, evaluation, and benchmark pipeline.

    Stages:
        1. Deterministic FrozenLake-v1 (is_slippery=False) policy extraction & verification.
        2. Stochastic FrozenLake-v1 (is_slippery=True) policy extraction & evaluation.
        3. Systematic Gamma discount factor comparative study (0.80 vs 0.95 vs 0.99).
    """
    print("\n" + "#" * 70)
    print("# STAGE 1: DETERMINISTIC Q-LEARNING (FrozenLake-v1 is_slippery=False)")
    print("#" * 70)

    det_agent, det_rewards = train_q_learning(
        is_slippery=False,
        n_episodes=1000,
        alpha=0.2,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.998,
        seed=42,
    )
    det_eval = evaluate_q_learning(det_agent, is_slippery=False, n_episodes=100, seed=123)
    print(f"\n[+] Deterministic Evaluation Success Rate: {det_eval:.2f}%")
    print("\n[+] Learned Optimal Policy (Deterministic):")
    display_optimal_policy(det_agent)

    det_plot = plot_training_rewards(
        rewards=det_rewards,
        title="Q-Learning Convergence on Deterministic FrozenLake-v1",
        save_filename="q_learning_deterministic.png",
        window=50,
        target_score=1.0,
    )
    print(f"[+] Saved deterministic reward plot: {det_plot}")

    print("\n" + "#" * 70)
    print("# STAGE 2: STOCHASTIC Q-LEARNING (FrozenLake-v1 is_slippery=True)")
    print("#" * 70)

    stoch_agent, stoch_rewards = train_q_learning(
        is_slippery=True,
        n_episodes=2500,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=42,
    )
    stoch_eval = evaluate_q_learning(stoch_agent, is_slippery=True, n_episodes=100, seed=123)
    print(f"\n[+] Stochastic Evaluation Success Rate: {stoch_eval:.2f}%")
    print("\n[+] Learned Optimal Policy (Stochastic Slippery):")
    display_optimal_policy(stoch_agent)

    stoch_plot = plot_training_rewards(
        rewards=stoch_rewards,
        title="Q-Learning Convergence on Stochastic FrozenLake-v1 (is_slippery=True)",
        save_filename="q_learning_stochastic.png",
        window=100,
    )
    print(f"[+] Saved stochastic reward plot: {stoch_plot}")

    print("\n" + "#" * 70)
    print("# STAGE 3: HYPERPARAMETER STUDY (Gamma Comparison)")
    print("#" * 70)
    run_gamma_experiment(gammas=[0.8, 0.95, 0.99], n_episodes=2500, seed=42)

    print("\n[OK] Q-Learning pipeline completed successfully!")
