"""Utility functions for RL-From-Scratch engine.

Provides reusable tools for directory management, compute device detection,
data smoothing (lag-free rolling moving averages), and scientific plotting
for Reinforcement Learning experiment benchmarking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


def get_results_dir(dir_name: str = "results") -> Path:
    """Retrieves and automatically provisions the experiment results directory.

    Resolves the directory path relative to the repository root to guarantee
    consistent output storage regardless of the working directory from which
    execution is invoked.

    Args:
        dir_name: Name of the output directory (default: "results").

    Returns:
        Path: Absolute pathlib.Path to the created/verified directory.
    """
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_device() -> torch.device:
    """Detects and returns the optimal available PyTorch computing device.

    Evaluates hardware acceleration support in hierarchical order:
    1. NVIDIA CUDA GPU
    2. Apple Silicon MPS (Metal Performance Shaders)
    3. CPU fallback

    Returns:
        torch.device: The selected compute device.
    """
    if torch is None:
        raise ImportError(
            "PyTorch is not installed in the current environment. "
            "Please run: pip install -r requirements.txt"
        )
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_moving_average(
    values: Sequence[float] | np.ndarray,
    window: int = 100,
) -> np.ndarray:
    """Computes a rolling moving average without temporal phase delay at the start.

    Conventional moving averages with fixed windows introduce artificial lag
    or NaN artifacts during initial warm-up steps. This implementation employs
    an expanding cumulative mean for steps t < window:
        MA[t] = sum(values[0 : t + 1]) / (t + 1)
    and transitions smoothly to the sliding window for t >= window:
        MA[t] = sum(values[t - window + 1 : t + 1]) / window

    Args:
        values: 1D sequence or NumPy array of numeric observations.
        window: Sliding window duration in episodes (default: 100).

    Returns:
        np.ndarray: Vector of smoothed values having identical length to values.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float64)
    if window <= 1:
        return arr.copy()

    cumsum = np.cumsum(arr)
    moving_avg = np.zeros(n, dtype=np.float64)

    # Initial warm-up segment: expanding mean to eliminate phase shift
    warmup_len = min(window, n)
    indices = np.arange(1, warmup_len + 1, dtype=np.float64)
    moving_avg[:warmup_len] = cumsum[:warmup_len] / indices

    # Steady-state rolling window segment
    if n > window:
        moving_avg[window:] = (cumsum[window:] - cumsum[:-window]) / window

    return moving_avg


def plot_training_rewards(
    rewards: Sequence[float] | np.ndarray,
    title: str,
    save_filename: str,
    window: int = 100,
    target_score: float | None = None,
) -> Path:
    """Plots and persists episodic reward curves and their rolling moving average.

    Visualizes raw rewards in high transparency (alpha=0.28) alongside a bold
    trendline representing the moving average over `window` episodes. Optionally
    overlays an environmental benchmark target score.

    Args:
        rewards: Sequence of episodic cumulative returns.
        title: Descriptive plot title.
        save_filename: Target image filename (e.g. 'dqn_cartpole_rewards.png').
        window: Smoothing window width (default: 100).
        target_score: Optional threshold indicating task resolution.

    Returns:
        Path: Absolute path to the saved figure.
    """
    results_dir = get_results_dir()
    save_path = results_dir / save_filename

    raw_rewards = np.asarray(rewards, dtype=np.float64)
    episodes = np.arange(1, len(raw_rewards) + 1)
    moving_avg = compute_moving_average(raw_rewards, window=window)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # Raw episodic trajectory (transparent)
    ax.plot(
        episodes,
        raw_rewards,
        color="#1f77b4",
        alpha=0.28,
        label="Raw Episode Reward",
        linewidth=1.0,
    )

    # Bold rolling moving average trend
    ax.plot(
        episodes,
        moving_avg,
        color="#1f77b4",
        linewidth=2.2,
        label=f"Rolling Mean ({window} episodes)",
    )

    # Environmental benchmark target line
    if target_score is not None:
        ax.axhline(
            y=target_score,
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            label=f"Solved Target Threshold ({target_score:.1f})",
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=11, fontweight="medium")
    ax.set_ylabel("Episodic Return", fontsize=11, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return save_path


def plot_gamma_comparison(
    results_by_gamma: dict[float, Sequence[float] | np.ndarray],
    save_filename: str = "q_learning_comparison.png",
    window: int = 100,
) -> Path:
    """Plots comparative convergence dynamics across multiple discount factors (gamma).

    Renders moving average reward trajectories for diverse discount factors on
    a single comparative canvas to evaluate credit assignment and temporal horizon
    effects on agent policy formation.

    Args:
        results_by_gamma: Mapping from gamma value to sequence of episodic returns.
        save_filename: Target output filename.
        window: Moving average smoothing window.

    Returns:
        Path: Absolute path to the persisted comparative figure.
    """
    results_dir = get_results_dir()
    save_path = results_dir / save_filename

    palette = ["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    for idx, (gamma, rewards) in enumerate(sorted(results_by_gamma.items())):
        raw = np.asarray(rewards, dtype=np.float64)
        episodes = np.arange(1, len(raw) + 1)
        ma = compute_moving_average(raw, window=window)
        color = palette[idx % len(palette)]
        final_val = ma[-1] if len(ma) > 0 else 0.0

        ax.plot(
            episodes,
            ma,
            color=color,
            linewidth=2.0,
            label=rf"$\gamma = {gamma:.2f}$ (Final MA: {final_val:.3f})",
        )

    ax.set_title(
        r"Discount Factor Sensitivity Analysis on Stochastic FrozenLake-v1 ($\gamma \in \{0.80, 0.95, 0.99\}$)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Episode", fontsize=11, fontweight="medium")
    ax.set_ylabel(f"Rolling Success Rate ({window} eps)", fontsize=11, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return save_path
