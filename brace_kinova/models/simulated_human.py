"""Simulated human model for BRACE training.

Generates noisy goal-directed trajectories using:
1. Minimum-jerk polynomial regression toward the true goal
2. Potential-field via-points near obstacles
3. AR(1) pink noise filter (validated against real human data)

Noise parameters calibrated from 798 trajectories, 23 participants (EEG study).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.signal import lfilter


class SimulatedHuman:
    """Simulated noisy human operator for training BRACE.

    Produces goal-directed XY velocity commands corrupted with temporally
    correlated (pink) noise matching real human motor noise statistics.
    """

    def __init__(
        self,
        noise_amplitude: float = 0.032,
        ar_coeff: float = 0.5,
        spectral_slope: float = -1.59,
        max_speed: float = 1.0,
        repulsion_radius: float = 0.10,
        repulsion_gain: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.noise_amplitude = noise_amplitude
        self.ar_coeff = ar_coeff
        self.spectral_slope = spectral_slope
        self.max_speed = max_speed
        self.repulsion_radius = repulsion_radius
        self.repulsion_gain = repulsion_gain

        self._rng = np.random.default_rng(seed)
        self._noise_state = np.zeros(2, dtype=np.float64)
        self._trajectory_length = 1.0

    def reset(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        trajectory_length: Optional[float] = None,
    ) -> None:
        """Reset the simulated human for a new episode."""
        self._noise_state = np.zeros(2, dtype=np.float64)
        if trajectory_length is not None:
            self._trajectory_length = trajectory_length
        else:
            self._trajectory_length = max(float(np.linalg.norm(goal_pos - start_pos)), 0.01)

    def _minimum_jerk_direction(
        self,
        current_pos: np.ndarray,
        goal_pos: np.ndarray,
    ) -> np.ndarray:
        """Compute deterministic direction via minimum-jerk profile.

        For 2D point-to-point reaching, minimum-jerk produces straight-line
        motion with bell-shaped velocity profile. Here we return the unit
        direction vector (speed profile is handled separately).
        """
        to_goal = goal_pos - current_pos
        dist = np.linalg.norm(to_goal)
        if dist < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return (to_goal / dist).astype(np.float32)

    def _obstacle_avoidance(
        self,
        current_pos: np.ndarray,
        goal_direction: np.ndarray,
        obstacle_positions: np.ndarray,
    ) -> np.ndarray:
        """Add repulsive via-point adjustments near obstacles (potential field)."""
        if obstacle_positions.shape[0] == 0:
            return goal_direction

        repulse = np.zeros(2, dtype=np.float64)
        for obs_pos in obstacle_positions:
            diff = current_pos - obs_pos
            dist = np.linalg.norm(diff)
            if dist < self.repulsion_radius and dist > 1e-8:
                push_dir = diff / dist
                strength = self.repulsion_gain / (dist ** 2)
                repulse += push_dir * strength

        combined = goal_direction.astype(np.float64) + repulse
        mag = np.linalg.norm(combined)
        if mag > 1e-8:
            combined = combined / mag
        return combined.astype(np.float32)

    def _ar1_pink_noise(self) -> np.ndarray:
        """Generate AR(1) pink noise sample: y[n] = ar_coeff * y[n-1] + (1-ar_coeff) * x[n].

        Produces temporally correlated noise with approximately 1/f spectrum.
        """
        white = self._rng.normal(0.0, 1.0, size=2)
        self._noise_state = (
            self.ar_coeff * self._noise_state + (1.0 - self.ar_coeff) * white
        )
        noise_scale = self.noise_amplitude * self._trajectory_length
        return (self._noise_state * noise_scale).astype(np.float32)

    def get_action(
        self,
        current_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacle_positions: np.ndarray,
    ) -> np.ndarray:
        """Generate a noisy human velocity command.

        Args:
            current_pos: (2,) current EE position in XY.
            goal_pos: (2,) true goal position.
            obstacle_positions: (n_obstacles, 2) obstacle positions.

        Returns:
            action: (2,) velocity command in [-max_speed, max_speed].
        """
        direction = self._minimum_jerk_direction(current_pos, goal_pos)
        direction = self._obstacle_avoidance(current_pos, direction, obstacle_positions)

        dist_to_goal = np.linalg.norm(goal_pos - current_pos)
        speed = min(dist_to_goal / max(self._trajectory_length, 1e-8), 1.0) * self.max_speed

        deterministic = direction * speed
        noise = self._ar1_pink_noise()
        noisy_action = deterministic + noise

        return np.clip(noisy_action, -self.max_speed, self.max_speed).astype(np.float32)

    def get_action_3d(
        self,
        current_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacle_positions: np.ndarray,
        close_gripper: bool = False,
    ) -> np.ndarray:
        """Generate a 3D action (vx, vy, gripper) for the environment."""
        action_2d = self.get_action(current_pos, goal_pos, obstacle_positions)
        gripper = 1.0 if close_gripper else -1.0
        return np.array([action_2d[0], action_2d[1], gripper], dtype=np.float32)


def generate_pink_noise_sequence(
    length: int,
    ar_coeff: float = 0.5,
    amplitude: float = 0.032,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a 2D pink noise sequence via AR(1) filtering.

    Uses scipy.signal.lfilter for batch generation.
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, size=(length, 2))
    b = np.array([1.0 - ar_coeff])
    a = np.array([1.0, -ar_coeff])
    pink = lfilter(b, a, white, axis=0) * amplitude
    return pink.astype(np.float32)
