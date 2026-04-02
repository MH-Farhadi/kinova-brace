"""Curriculum manager for 5-stage progressive training.

Tracks per-stage metrics and automatically advances when criteria are met.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from brace_kinova.envs.scenarios import ScenarioConfig, SCENARIOS


@dataclass
class StageMetrics:
    """Running metrics for a single curriculum stage."""

    episodes: int = 0
    successes: int = 0
    collisions: int = 0
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=200))

    @property
    def success_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.successes / self.episodes

    @property
    def collision_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.collisions / self.episodes

    def is_plateau(self, window: int = 200) -> bool:
        """Check if reward has plateaued over the last `window` episodes."""
        if len(self.recent_rewards) < window:
            return False
        rewards = list(self.recent_rewards)
        first_half = np.mean(rewards[: window // 2])
        second_half = np.mean(rewards[window // 2 :])
        return abs(second_half - first_half) < 0.01 * abs(first_half + 1e-8)


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""

    name: str
    n_objects: int = 1
    n_obstacles: int = 0
    min_episodes: int = 100
    success_threshold: float = 0.80
    max_collision_rate: Optional[float] = None
    plateau_window: Optional[int] = None


class CurriculumManager:
    """Manages 5-stage curriculum with automatic advancement."""

    DEFAULT_STAGES = [
        StageConfig("basic_reaching", 1, 0, 100, 0.80),
        StageConfig("collision_avoidance", 1, 3, 200, 0.75, max_collision_rate=0.15),
        StageConfig("challenging_obstacles", 1, 4, 300, 0.70),
        StageConfig("goal_ambiguity", 3, 3, 400, 0.65),
        StageConfig("full_complexity", 3, 4, plateau_window=200),
    ]

    def __init__(self, stages: Optional[list[StageConfig]] = None):
        self.stages = stages or self.DEFAULT_STAGES
        self.current_stage_idx = 0
        self._metrics = [StageMetrics() for _ in self.stages]

    @classmethod
    def from_yaml(cls, config: dict) -> CurriculumManager:
        """Build from a YAML curriculum config section."""
        stages = []
        for s in config.get("stages", []):
            stages.append(
                StageConfig(
                    name=s["name"],
                    n_objects=s.get("n_objects", 1),
                    n_obstacles=s.get("n_obstacles", 0),
                    min_episodes=s.get("min_episodes", 100),
                    success_threshold=s.get("success_threshold", 0.0),
                    max_collision_rate=s.get("max_collision_rate"),
                    plateau_window=s.get("plateau_window"),
                )
            )
        return cls(stages)

    @property
    def current_stage(self) -> StageConfig:
        return self.stages[self.current_stage_idx]

    @property
    def current_scenario(self) -> ScenarioConfig:
        """Get the ScenarioConfig for the current curriculum stage."""
        stage = self.current_stage
        if stage.name in SCENARIOS:
            return SCENARIOS[stage.name]
        return ScenarioConfig(
            name=stage.name,
            n_objects=stage.n_objects,
            n_obstacles=stage.n_obstacles,
        )

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    @property
    def metrics(self) -> StageMetrics:
        return self._metrics[self.current_stage_idx]

    def record_episode(
        self, success: bool, collision: bool, total_reward: float
    ) -> None:
        """Record results from a completed episode."""
        m = self.metrics
        m.episodes += 1
        if success:
            m.successes += 1
        if collision:
            m.collisions += 1
        m.recent_rewards.append(total_reward)

    def should_advance(self) -> bool:
        """Check if criteria for advancing to the next stage are met."""
        if self.is_final_stage:
            return False

        stage = self.current_stage
        m = self.metrics

        if m.episodes < stage.min_episodes:
            return False

        if stage.plateau_window is not None:
            return m.is_plateau(stage.plateau_window)

        if m.success_rate < stage.success_threshold:
            return False

        if stage.max_collision_rate is not None:
            if m.collision_rate > stage.max_collision_rate:
                return False

        return True

    def advance(self) -> bool:
        """Advance to the next stage if possible. Returns True if advanced."""
        if self.should_advance():
            self.current_stage_idx += 1
            return True
        return False

    def get_state(self) -> dict:
        return {
            "current_stage_idx": self.current_stage_idx,
            "metrics": [
                {
                    "episodes": m.episodes,
                    "successes": m.successes,
                    "collisions": m.collisions,
                }
                for m in self._metrics
            ],
        }

    def load_state(self, state: dict) -> None:
        self.current_stage_idx = state["current_stage_idx"]
        for i, ms in enumerate(state.get("metrics", [])):
            if i < len(self._metrics):
                self._metrics[i].episodes = ms["episodes"]
                self._metrics[i].successes = ms["successes"]
                self._metrics[i].collisions = ms["collisions"]
