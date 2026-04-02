"""2D planar reach-and-grasp Gymnasium environment for BRACE training.

Lightweight standalone environment (no physics engine) for rapid iteration.
All motion is constrained to the XY plane at a fixed Z height.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from brace_kinova.envs.scenarios import ScenarioConfig, SCENARIOS


class ReachGraspEnv(gym.Env):
    """2D planar reach-and-grasp with obstacles and goal ambiguity.

    Observation (continuous Box):
        - EE position (2): XY
        - EE velocity (2): XY
        - Gripper state (1): 0=open, 1=closed
        - Relative XY to each object (2 * n_objects)
        - Relative XY to each obstacle (2 * n_obstacles)
        - Distance to nearest obstacle (1)
        - Normalized progress toward each object (n_objects)
        Total: 5 + 2*n_objects + 2*n_obstacles + 1 + n_objects

    Action (continuous Box, shape=(3,)):
        - vx, vy: Cartesian EE velocity in [-1, 1], scaled by max_velocity
        - gripper: <0 = open, >=0 = close
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scenario: str | ScenarioConfig = "full_complexity",
        max_steps: int = 200,
        max_velocity: float = 0.15,
        workspace_bounds: Optional[dict] = None,
        grasp_threshold: float = 0.04,
        obstacle_radius: float = 0.03,
        ee_radius: float = 0.015,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        reward_type: str = "expert",
    ):
        super().__init__()

        if isinstance(scenario, str):
            self.scenario = SCENARIOS[scenario]
        else:
            self.scenario = scenario

        self.max_steps = max_steps
        self.max_velocity = max_velocity
        self.grasp_threshold = grasp_threshold
        self.obstacle_radius = obstacle_radius
        self.ee_radius = ee_radius
        self.render_mode = render_mode
        self.reward_type = reward_type

        wb = workspace_bounds or {}
        self.x_min = wb.get("x_min", 0.20)
        self.x_max = wb.get("x_max", 0.60)
        self.y_min = wb.get("y_min", -0.30)
        self.y_max = wb.get("y_max", 0.45)
        self.ws_diag = math.hypot(self.x_max - self.x_min, self.y_max - self.y_min)

        self.ee_init_x = wb.get("initial_x", 0.30)
        self.ee_init_y = wb.get("initial_y", 0.0)
        self.dt = wb.get("dt", 0.05)

        self.n_objects = self.scenario.n_objects
        self.n_obstacles = self.scenario.n_obstacles

        obs_dim = 5 + 2 * self.n_objects + 2 * self.n_obstacles + 1 + self.n_objects
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self._step_count = 0
        self._ee_pos = np.zeros(2, dtype=np.float32)
        self._ee_vel = np.zeros(2, dtype=np.float32)
        self._gripper_state = 0.0
        self._object_positions = np.zeros((self.n_objects, 2), dtype=np.float32)
        self._obstacle_positions = np.zeros((self.n_obstacles, 2), dtype=np.float32)
        self._true_goal_idx = 0
        self._prev_dist_to_goal = 0.0
        self._initial_dist_to_goal = 0.0
        self._prev_heading = 0.0
        self._collision_count = 0

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

    @property
    def true_goal_position(self) -> np.ndarray:
        return self._object_positions[self._true_goal_idx]

    @property
    def n_goals(self) -> int:
        return self.n_objects

    def _sample_positions(self, n: int, existing: list[np.ndarray], min_sep: float) -> np.ndarray:
        """Sample n non-overlapping positions within scenario bounds."""
        positions = []
        x_lo, x_hi = self.scenario.object_x_range
        y_lo, y_hi = self.scenario.object_y_range
        max_attempts = 500

        for _ in range(n):
            for attempt in range(max_attempts):
                x = self.np_random.uniform(x_lo, x_hi)
                y = self.np_random.uniform(y_lo, y_hi)
                pos = np.array([x, y], dtype=np.float32)
                too_close = False
                for other in existing + positions:
                    if np.linalg.norm(pos - other) < min_sep:
                        too_close = True
                        break
                if not too_close:
                    positions.append(pos)
                    break
            else:
                x = self.np_random.uniform(x_lo, x_hi)
                y = self.np_random.uniform(y_lo, y_hi)
                positions.append(np.array([x, y], dtype=np.float32))

        return np.array(positions, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        rel_objects = self._object_positions - self._ee_pos[np.newaxis, :]
        parts = [
            self._ee_pos,
            self._ee_vel,
            np.array([self._gripper_state], dtype=np.float32),
            rel_objects.flatten(),
        ]

        if self.n_obstacles > 0:
            rel_obstacles = self._obstacle_positions - self._ee_pos[np.newaxis, :]
            obs_dists = np.linalg.norm(rel_obstacles, axis=1)
            min_obs_dist = float(obs_dists.min()) if len(obs_dists) > 0 else 1.0
            parts.append(rel_obstacles.flatten())
        else:
            min_obs_dist = 1.0

        parts.append(np.array([min_obs_dist], dtype=np.float32))

        dists_to_objects = np.linalg.norm(
            self._object_positions - self._ee_pos[np.newaxis, :], axis=1
        )
        progress = 1.0 - np.clip(dists_to_objects / max(self.ws_diag, 1e-6), 0, 1)
        parts.append(progress.astype(np.float32))

        return np.concatenate(parts)

    def _check_collision(self) -> bool:
        if self.n_obstacles == 0:
            return False
        dists = np.linalg.norm(
            self._obstacle_positions - self._ee_pos[np.newaxis, :], axis=1
        )
        return bool(np.any(dists < (self.obstacle_radius + self.ee_radius)))

    def _check_grasp(self) -> bool:
        dist = np.linalg.norm(self._ee_pos - self.true_goal_position)
        return bool(dist < self.grasp_threshold and self._gripper_state > 0.5)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self._collision_count = 0

        self._ee_pos = np.array(
            [self.ee_init_x, self.ee_init_y], dtype=np.float32
        )
        jitter = self.np_random.uniform(-0.03, 0.03, size=2).astype(np.float32)
        self._ee_pos += jitter
        self._ee_vel = np.zeros(2, dtype=np.float32)
        self._gripper_state = 0.0

        ee_list = [self._ee_pos.copy()]
        self._object_positions = self._sample_positions(
            self.n_objects, ee_list, self.scenario.min_object_separation
        )

        if self.n_obstacles > 0:
            existing = ee_list + [p for p in self._object_positions]
            self._obstacle_positions = self._sample_positions(
                self.n_obstacles,
                existing,
                self.scenario.min_obstacle_object_separation,
            )
        else:
            self._obstacle_positions = np.zeros((0, 2), dtype=np.float32)

        self._true_goal_idx = int(self.np_random.integers(0, self.n_objects))
        self._prev_dist_to_goal = float(
            np.linalg.norm(self._ee_pos - self.true_goal_position)
        )
        self._initial_dist_to_goal = self._prev_dist_to_goal
        self._prev_heading = 0.0

        obs = self._get_obs()
        info = {
            "true_goal_idx": self._true_goal_idx,
            "object_positions": self._object_positions.copy(),
            "obstacle_positions": self._obstacle_positions.copy(),
            "n_objects": self.n_objects,
            "n_obstacles": self.n_obstacles,
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0)
        vx, vy = action[0] * self.max_velocity, action[1] * self.max_velocity
        gripper_cmd = action[2]

        self._ee_vel = np.array([vx, vy], dtype=np.float32)
        self._ee_pos = self._ee_pos + self._ee_vel * self.dt
        self._ee_pos[0] = np.clip(self._ee_pos[0], self.x_min, self.x_max)
        self._ee_pos[1] = np.clip(self._ee_pos[1], self.y_min, self.y_max)

        self._gripper_state = 1.0 if gripper_cmd >= 0.0 else 0.0

        self._step_count += 1
        terminated = False
        truncated = False
        reward = 0.0

        collision = self._check_collision()
        if collision:
            self._collision_count += 1
            terminated = True
            reward = -10.0

        grasped = self._check_grasp()
        if grasped and not terminated:
            terminated = True
            reward = 5.0

        if not terminated:
            d_goal = float(np.linalg.norm(self._ee_pos - self.true_goal_position))
            d_max = max(self._initial_dist_to_goal, 1e-6)

            heading = math.atan2(vy, vx) if (abs(vx) + abs(vy)) > 1e-6 else self._prev_heading
            delta_heading = heading - self._prev_heading

            if self.n_obstacles > 0:
                obs_dists = np.linalg.norm(
                    self._obstacle_positions - self._ee_pos[np.newaxis, :], axis=1
                )
                min_obs_dist = float(obs_dists.min())
            else:
                min_obs_dist = 1.0

            reward = (
                3.0 * (self._prev_dist_to_goal - d_goal) / d_max
                - 0.8 * delta_heading ** 2
                - 2.5 * math.exp(-min_obs_dist / 0.06)
            )

            self._prev_dist_to_goal = d_goal
            self._prev_heading = heading

        if self._step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "collision": collision,
            "grasped": grasped,
            "true_goal_idx": self._true_goal_idx,
            "dist_to_goal": float(np.linalg.norm(self._ee_pos - self.true_goal_position)),
            "collision_count": self._collision_count,
            "step_count": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def get_expert_obs(self) -> np.ndarray:
        """Observation for the expert: includes true goal identity."""
        base_obs = self._get_obs()
        goal_one_hot = np.zeros(self.n_objects, dtype=np.float32)
        goal_one_hot[self._true_goal_idx] = 1.0
        return np.concatenate([base_obs, goal_one_hot])

    def set_scenario(self, scenario: str | ScenarioConfig) -> None:
        """Dynamically switch scenario (used by curriculum manager)."""
        if isinstance(scenario, str):
            self.scenario = SCENARIOS[scenario]
        else:
            self.scenario = scenario

        self.n_objects = self.scenario.n_objects
        self.n_obstacles = self.scenario.n_obstacles

        obs_dim = 5 + 2 * self.n_objects + 2 * self.n_obstacles + 1 + self.n_objects
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def render(self):
        if self.render_mode is None:
            return None
        # Minimal rendering stub; full visualization in visualize.py
        return None


class ExpertReachGraspEnv(ReachGraspEnv):
    """Variant where the expert sees the true goal (appended one-hot).

    The observation space is extended by n_objects dimensions (one-hot goal).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        expert_obs_dim = self.observation_space.shape[0] + self.n_objects
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(expert_obs_dim,), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()
        goal_one_hot = np.zeros(self.n_objects, dtype=np.float32)
        goal_one_hot[self._true_goal_idx] = 1.0
        return np.concatenate([base, goal_one_hot])


class ArbitrationEnv(gym.Env):
    """BRACE arbitration training environment.

    Wraps ReachGraspEnv with belief vector concatenated into obs.
    The action is scalar gamma in [-1, 1], mapped to [0, 1] via 0.5*(a+1).
    Blending: a_exec = (1-gamma)*h_t + gamma*w_t.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        base_env: ReachGraspEnv,
        n_goals: int = 3,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.base_env = base_env
        self.n_goals = n_goals
        self.render_mode = render_mode

        base_obs_dim = base_env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_obs_dim + n_goals,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._belief = np.ones(n_goals, dtype=np.float32) / n_goals
        self._human_action = np.zeros(3, dtype=np.float32)
        self._expert_action = np.zeros(3, dtype=np.float32)
        self._gamma = 0.0

    def set_actions(
        self, human_action: np.ndarray, expert_action: np.ndarray, belief: np.ndarray
    ) -> None:
        """Set current human/expert actions and belief (called externally each step)."""
        self._human_action = human_action
        self._expert_action = expert_action
        self._belief = belief

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.base_env.reset(**kwargs)
        self._belief = np.ones(self.n_goals, dtype=np.float32) / self.n_goals
        fused = np.concatenate([obs, self._belief])
        return fused, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        gamma_raw = float(np.clip(action[0], -1.0, 1.0))
        self._gamma = 0.5 * (gamma_raw + 1.0)

        blended = (1.0 - self._gamma) * self._human_action + self._gamma * self._expert_action
        obs, _, terminated, truncated, info = self.base_env.step(blended)

        info["gamma"] = self._gamma
        info["belief"] = self._belief.copy()

        fused = np.concatenate([obs, self._belief])
        reward = 0.0  # Computed externally by ArbitrationReward
        return fused, reward, terminated, truncated, info


gym.register(
    id="BraceReachGrasp-v0",
    entry_point="brace_kinova.envs.reach_grasp_env:ReachGraspEnv",
)

gym.register(
    id="BraceExpertReachGrasp-v0",
    entry_point="brace_kinova.envs.reach_grasp_env:ExpertReachGraspEnv",
)
