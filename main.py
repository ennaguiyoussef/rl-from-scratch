"""CLI Entrypoint for the RL-From-Scratch Reinforcement Learning Engine.

Provides unified command-line control to execute tabular Q-Learning pipelines,
Deep Q-Network (DQN) training/evaluation routines, and real-time visual demos.

Usage:
    python main.py --mode all          # Executes complete benchmark suite (default)
    python main.py --mode qlearning    # Executes Tabular Q-Learning experiments
    python main.py --mode dqn          # Trains and evaluates Deep Q-Network
    python main.py --mode demo         # Runs real-time graphical policy rollouts
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym

from src.dqn_agent import DQNAgent
from src.q_learning import (
    evaluate_q_learning,
    run_q_learning_pipeline,
    train_q_learning,
)
from src.train_dqn import evaluate_dqn_cartpole, train_dqn_cartpole
from src.utils import get_results_dir


def run_cartpole_visual_demo(agent: DQNAgent, n_episodes: int = 3) -> None:
    """Renders real-time visual rollouts of trained DQN agent on CartPole-v1.

    Args:
        agent: Trained DQNAgent.
        n_episodes: Number of demonstration episodes to render.
    """
    print(f"\n[*] Launching Real-Time DQN Visual Demonstration on CartPole-v1 ({n_episodes} episodes)...")
    try:
        env = gym.make("CartPole-v1", render_mode="human")
    except Exception as exc:
        print(f"[!] Warning: Unable to initialize human render mode ({exc}).")
        print("    Running demonstration in headless console mode instead.")
        env = gym.make("CartPole-v1")

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            step_count += 1
            done = terminated or truncated
            time.sleep(0.02)  # Maintain ~50 FPS for human viewing

        print(f"    [CartPole Demo] Episode {ep}/{n_episodes} - Steps / Score: {ep_reward:.1f}")

    env.close()


def run_frozenlake_visual_demo(n_episodes: int = 3) -> None:
    """Renders visual demonstration of optimal tabular Q-learning on FrozenLake-v1.

    Args:
        n_episodes: Number of demonstration episodes.
    """
    print(f"\n[*] Training & Rendering Tabular Q-Learning Demo on FrozenLake-v1 ({n_episodes} episodes)...")
    # Train deterministic agent to illustrate perfect navigation
    agent, _ = train_q_learning(is_slippery=False, n_episodes=1000, seed=42)

    try:
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
    except Exception:
        # Fallback to ANSI text rendering if display window is unsupported
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="ansi")

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        if env.render_mode == "ansi":
            print(f"\n--- FrozenLake Episode {ep} ---")
            print(env.render())

        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated

            if env.render_mode == "ansi":
                print(env.render())
            time.sleep(0.15)

        status = "GOAL REACHED! (+1.0)" if ep_reward > 0.0 else "FELL IN HOLE (0.0)"
        print(f"    [FrozenLake Demo] Episode {ep}/{n_episodes} - Result: {status}")

    env.close()


def run_demo_mode() -> None:
    """Handles demo mode: verifies or trains DQN weights, then launches visual rendering."""
    results_dir = get_results_dir()
    weights_path = results_dir / "dqn_cartpole.pth"

    if not weights_path.is_file():
        print(f"[*] No pre-trained weights found at {weights_path}.")
        print("[*] Triggering fast DQN training to generate checkpoint...")
        agent, _ = train_dqn_cartpole(n_episodes=500, early_stop_score=475.0)
    else:
        print(f"[*] Found pre-trained weights at {weights_path}.")
        dummy_env = gym.make("CartPole-v1")
        state_dim = int(dummy_env.observation_space.shape[0])
        action_dim = int(dummy_env.action_space.n)
        dummy_env.close()

        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load(weights_path)

    # 1. CartPole Continuous Control DQN Demo
    run_cartpole_visual_demo(agent=agent, n_episodes=3)

    # 2. FrozenLake Discrete Tabular Q-Learning Demo
    run_frozenlake_visual_demo(n_episodes=3)


def main() -> None:
    """Parses CLI arguments and orchestrates RL engine execution."""
    parser = argparse.ArgumentParser(
        prog="RL-From-Scratch",
        description=(
            "RL-From-Scratch : Value-Based Reinforcement Learning Engine "
            "(Tabular Q-Learning & Deep Q-Network)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["qlearning", "dqn", "demo", "all"],
        help="Pipeline execution mode: qlearning, dqn, demo, or all.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  RL-FROM-SCRATCH: VALUE-BASED REINFORCEMENT LEARNING ENGINE")
    print(f"  Target Mode: {args.mode.upper()}")
    print("=" * 70)

    start_time = time.perf_counter()

    if args.mode == "qlearning":
        run_q_learning_pipeline()

    elif args.mode == "dqn":
        agent, rewards = train_dqn_cartpole()
        evaluate_dqn_cartpole(agent=agent, n_eval_episodes=100)

    elif args.mode == "demo":
        run_demo_mode()

    elif args.mode == "all":
        print("\n>>> STARTING PART 1: TABULAR Q-LEARNING EXPERIMENTS <<<")
        run_q_learning_pipeline()

        print("\n>>> STARTING PART 2: DEEP Q-NETWORK (DQN) ON CARTPOLE-V1 <<<")
        agent, rewards = train_dqn_cartpole()
        evaluate_dqn_cartpole(agent=agent, n_eval_episodes=100)

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 70)
    print(f"[COMPLETE] Execution finished in {elapsed:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
