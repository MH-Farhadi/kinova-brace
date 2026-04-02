"""Reward functions for expert (SAC) and arbitration (PPO) training.

Expert reward: dense, goal-aware, 2D distances.
Arbitration reward: belief-aware BRACE reward with all paper components.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ExpertRewardConfig:
    progress_weight: float = 3.0
    heading_penalty_weight: float = 0.8
    obstacle_penalty_weight: float = 2.5
    d_safe: float = 0.06
    goal_bonus: float = 5.0
    collision_penalty: float = 10.0


class ExpertReward:
    """Dense reward for expert (SAC) training. Goal-aware, 2D Euclidean distances.

    R = progress_w * (d_{t-1} - d_t) / d_max
        - heading_w * ||delta_theta||^2
        - obstacle_w * exp(-d_obs_min / d_safe)
    """

    def __init__(self, config: Optional[ExpertRewardConfig] = None):
        self.cfg = config or ExpertRewardConfig()

    def __call__(
        self,
        prev_dist_to_goal: float,
        curr_dist_to_goal: float,
        initial_dist: float,
        delta_heading: float,
        min_obstacle_dist: float,
        collision: bool = False,
        grasped: bool = False,
    ) -> float:
        if collision:
            return -self.cfg.collision_penalty

        if grasped:
            return self.cfg.goal_bonus

        d_max = max(initial_dist, 1e-6)
        progress = self.cfg.progress_weight * (prev_dist_to_goal - curr_dist_to_goal) / d_max
        heading_penalty = -self.cfg.heading_penalty_weight * delta_heading ** 2
        obstacle_penalty = -self.cfg.obstacle_penalty_weight * math.exp(
            -min_obstacle_dist / self.cfg.d_safe
        )

        return progress + heading_penalty + obstacle_penalty


@dataclass
class ArbitrationRewardConfig:
    w_collision: float = 10.0
    w_proximity: float = 2.5
    w_far: float = 1.5
    w_progress: float = 3.0
    w_autonomy: float = 1.5
    w_goal: float = 2.0
    near_threshold: float = 0.06
    far_threshold: float = 0.20


class ArbitrationReward:
    """BRACE reward for arbitration (PPO) training. Belief-aware.

    R = -w_coll * 1_collision
        + w_prox * gamma * p_max * 1_near
        - w_far * gamma * 1_far
        + w_prog * p_max * (d_{t-1} - d_t)
        - w_auto * gamma^2
        + w_goal * log(p_true)
    """

    def __init__(self, config: Optional[ArbitrationRewardConfig] = None):
        self.cfg = config or ArbitrationRewardConfig()

    def __call__(
        self,
        gamma: float,
        belief: np.ndarray,
        true_goal_idx: int,
        prev_dist_to_goal: float,
        curr_dist_to_goal: float,
        min_obstacle_dist: float,
        collision: bool = False,
    ) -> float:
        p_max = float(np.max(belief))
        p_true = float(belief[true_goal_idx])

        collision_term = -self.cfg.w_collision if collision else 0.0

        is_near = min_obstacle_dist < self.cfg.near_threshold
        proximity_term = self.cfg.w_proximity * gamma * p_max if is_near else 0.0

        is_far = min_obstacle_dist > self.cfg.far_threshold
        far_term = -self.cfg.w_far * gamma if is_far else 0.0

        progress_term = self.cfg.w_progress * p_max * (prev_dist_to_goal - curr_dist_to_goal)

        autonomy_penalty = -self.cfg.w_autonomy * gamma ** 2

        goal_term = self.cfg.w_goal * math.log(max(p_true, 1e-10))

        return (
            collision_term
            + proximity_term
            + far_term
            + progress_term
            + autonomy_penalty
            + goal_term
        )
