"""Visualization tools for BRACE evaluation.

Produces:
- Trajectory plots with obstacle/goal positions
- Belief entropy over time
- Gamma over time
- Comparison bar charts across conditions
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_trajectory(
    ee_positions: np.ndarray,
    object_positions: np.ndarray,
    obstacle_positions: np.ndarray,
    true_goal_idx: int,
    gammas: Optional[np.ndarray] = None,
    title: str = "Trajectory",
    save_path: Optional[str] = None,
    workspace_bounds: Optional[dict] = None,
) -> None:
    """Plot a single episode trajectory with goals and obstacles.

    Args:
        ee_positions: (T, 2) end-effector XY positions over time.
        object_positions: (n_objects, 2) goal object positions.
        obstacle_positions: (n_obstacles, 2) obstacle positions.
        true_goal_idx: index of the true target object.
        gammas: (T,) optional gamma values for color-coding the trajectory.
        title: plot title.
        save_path: if provided, save figure to this path.
        workspace_bounds: dict with x_min, x_max, y_min, y_max.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    wb = workspace_bounds or {"x_min": 0.20, "x_max": 0.60, "y_min": -0.30, "y_max": 0.45}
    ax.set_xlim(wb["x_min"] - 0.05, wb["x_max"] + 0.05)
    ax.set_ylim(wb["y_min"] - 0.05, wb["y_max"] + 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)

    for i, obs in enumerate(obstacle_positions):
        circle = Circle(obs, 0.03, color="red", alpha=0.5, label="Obstacle" if i == 0 else None)
        ax.add_patch(circle)

    for i, obj in enumerate(object_positions):
        color = "green" if i == true_goal_idx else "blue"
        label = "True Goal" if i == true_goal_idx else ("Distractor" if i == 0 else None)
        circle = Circle(obj, 0.025, color=color, alpha=0.7, label=label)
        ax.add_patch(circle)

    if gammas is not None and len(gammas) == len(ee_positions):
        for t in range(len(ee_positions) - 1):
            color = plt.cm.coolwarm(gammas[t])
            ax.plot(
                ee_positions[t : t + 2, 0],
                ee_positions[t : t + 2, 1],
                color=color,
                linewidth=2,
            )
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="γ (assistance level)")
    else:
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], "k-", linewidth=1.5, label="Trajectory")

    ax.plot(ee_positions[0, 0], ee_positions[0, 1], "ko", markersize=8, label="Start")
    ax.plot(ee_positions[-1, 0], ee_positions[-1, 1], "k*", markersize=12, label="End")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory plot to {save_path}")
    plt.close(fig)


def plot_belief_entropy(
    beliefs: np.ndarray,
    true_goal_idx: int,
    title: str = "Belief Entropy",
    save_path: Optional[str] = None,
) -> None:
    """Plot belief entropy and true-goal probability over time.

    Args:
        beliefs: (T, n_goals) belief distributions over time.
        true_goal_idx: index of the true target.
        title: plot title.
        save_path: optional save path.
    """
    T = beliefs.shape[0]
    n_goals = beliefs.shape[1]

    entropy = -np.sum(beliefs * np.log(np.clip(beliefs, 1e-10, 1.0)), axis=1)
    p_true = beliefs[:, true_goal_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(range(T), entropy, "b-", linewidth=1.5)
    ax1.set_ylabel("Belief Entropy (nats)")
    ax1.set_title(title)
    ax1.axhline(y=np.log(n_goals), color="gray", linestyle="--", alpha=0.5, label="Max entropy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(T), p_true, "g-", linewidth=1.5, label="P(true goal)")
    ax2.axhline(y=1.0 / n_goals, color="gray", linestyle="--", alpha=0.5, label="Uniform")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_over_time(
    gammas: np.ndarray,
    title: str = "Gamma (Assistance Level) Over Time",
    save_path: Optional[str] = None,
) -> None:
    """Plot gamma values over an episode."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(len(gammas)), gammas, "purple", linewidth=1.5)
    ax.fill_between(range(len(gammas)), 0, gammas, alpha=0.2, color="purple")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("γ")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(
    conditions: list[str],
    metrics: dict[str, list[float]],
    title: str = "Condition Comparison",
    save_path: Optional[str] = None,
) -> None:
    """Plot grouped bar chart comparing conditions across metrics."""
    n_conditions = len(conditions)
    n_metrics = len(metrics)
    x = np.arange(n_conditions)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric_name)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
