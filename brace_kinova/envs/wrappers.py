"""Observation wrappers for the reach-and-grasp environment."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizeObservation(gym.ObservationWrapper):
    """Running mean/std normalization of observations."""

    def __init__(self, env: gym.Env, clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.clip = clip
        self.epsilon = epsilon
        self._mean = np.zeros(obs_shape, dtype=np.float64)
        self._var = np.ones(obs_shape, dtype=np.float64)
        self._count = 0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._update_stats(obs)
        normed = (obs - self._mean) / np.sqrt(self._var + self.epsilon)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def _update_stats(self, obs: np.ndarray) -> None:
        self._count += 1
        if self._count == 1:
            self._mean = obs.astype(np.float64)
            self._var = np.zeros_like(self._mean)
        else:
            delta = obs.astype(np.float64) - self._mean
            self._mean += delta / self._count
            delta2 = obs.astype(np.float64) - self._mean
            self._var += (delta * delta2 - self._var) / self._count

    def get_state(self) -> dict:
        return {"mean": self._mean.copy(), "var": self._var.copy(), "count": self._count}

    def set_state(self, state: dict) -> None:
        self._mean = state["mean"]
        self._var = state["var"]
        self._count = state["count"]


class GoalMaskedObservation(gym.ObservationWrapper):
    """Appends the true goal one-hot to the observation (for expert training).

    Expects the wrapped env to have a `_true_goal_idx` and `n_objects` attribute.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base_env = env.unwrapped
        n_objects = getattr(base_env, "n_objects", 3)
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_dim + n_objects,),
            dtype=np.float32,
        )
        self._n_objects = n_objects

    def observation(self, obs: np.ndarray) -> np.ndarray:
        base_env = self.env.unwrapped
        goal_idx = getattr(base_env, "_true_goal_idx", 0)
        one_hot = np.zeros(self._n_objects, dtype=np.float32)
        one_hot[goal_idx] = 1.0
        return np.concatenate([obs, one_hot])


class ClipAction(gym.ActionWrapper):
    """Clips actions to the action space bounds."""

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.action_space.low, self.action_space.high)
