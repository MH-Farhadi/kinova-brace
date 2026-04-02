"""Expert policy wrapper for frozen SAC model.

Loads a trained SAC model and provides optimal actions toward the true goal.
The expert is frozen (no gradient updates) during arbitration training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class ExpertPolicy:
    """Wraps a trained SAC model for expert action prediction.

    Given full state (including true target identity), returns the optimal
    Cartesian velocity action. The model is frozen during arbitration training.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.model = None
        self.device = device
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Load a trained SAC model from a .zip file."""
        from stable_baselines3 import SAC

        self.model = SAC.load(model_path, device=self.device)
        self.model.policy.set_training_mode(False)

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Predict action given an observation with true goal info.

        Args:
            obs: observation array (includes true goal one-hot for expert env).
            deterministic: use deterministic policy (default True for expert).

        Returns:
            action: (3,) array [vx, vy, gripper] in [-1, 1].
        """
        if self.model is None:
            return self._fallback_action(obs)

        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def predict_batch(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Predict actions for a batch of observations."""
        if self.model is None:
            return np.array([self._fallback_action(o) for o in obs])

        actions = []
        for o in obs:
            a, _ = self.model.predict(o, deterministic=deterministic)
            actions.append(a)
        return np.array(actions)

    @staticmethod
    def _fallback_action(obs: np.ndarray) -> np.ndarray:
        """Simple potential-field fallback when no model is loaded."""
        return np.zeros(3, dtype=np.float32)


class PotentialFieldExpert:
    """Analytical expert using potential field (goal attraction + obstacle repulsion).

    Does not require training. Useful as a baseline or when SAC is unavailable.
    """

    def __init__(
        self,
        repulsion_radius: float = 0.10,
        repulsion_gain: float = 0.5,
    ):
        self.repulsion_radius = repulsion_radius
        self.repulsion_gain = repulsion_gain

    def predict(
        self,
        ee_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacle_positions: np.ndarray,
        close_gripper: bool = False,
    ) -> np.ndarray:
        """Compute expert action via potential field.

        Args:
            ee_pos: (2,) current EE position.
            goal_pos: (2,) target goal position.
            obstacle_positions: (n_obstacles, 2) obstacle positions.
            close_gripper: whether to close the gripper.

        Returns:
            action: (3,) [vx, vy, gripper] in [-1, 1].
        """
        to_goal = goal_pos - ee_pos
        goal_dist = np.linalg.norm(to_goal)
        goal_dir = to_goal / max(goal_dist, 1e-8)

        repulse = np.zeros(2, dtype=np.float32)
        if obstacle_positions.shape[0] > 0:
            for obs_pos in obstacle_positions:
                diff = ee_pos - obs_pos
                dist = np.linalg.norm(diff)
                if dist < self.repulsion_radius and dist > 1e-8:
                    push_dir = diff / dist
                    strength = self.repulsion_gain / (dist ** 2)
                    repulse += push_dir * strength

        combined = goal_dir + repulse
        mag = np.linalg.norm(combined)
        if mag > 1e-8:
            combined = combined / mag

        gripper = 1.0 if close_gripper else -1.0
        action = np.array([combined[0], combined[1], gripper], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)
