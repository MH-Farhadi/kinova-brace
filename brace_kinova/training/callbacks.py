"""SB3 callbacks for logging, checkpointing, and curriculum advancement."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from brace_kinova.training.curriculum import CurriculumManager


class CurriculumCallback(BaseCallback):
    """Monitors episode metrics and advances the curriculum stage when criteria are met."""

    def __init__(
        self,
        curriculum: CurriculumManager,
        env_update_fn=None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.env_update_fn = env_update_fn

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]
                success = info.get("grasped", False)
                collision = info.get("collision", False)
                reward = ep_info.get("r", 0.0)
                self.curriculum.record_episode(success, collision, reward)

        if self.curriculum.advance():
            stage = self.curriculum.current_stage
            if self.verbose:
                print(
                    f"[Curriculum] Advanced to stage {self.curriculum.current_stage_idx}: "
                    f"{stage.name} (n_obj={stage.n_objects}, n_obs={stage.n_obstacles})"
                )
            if self.env_update_fn is not None:
                self.env_update_fn(self.curriculum.current_scenario)

        return True

    def _on_training_end(self) -> None:
        if self.verbose:
            m = self.curriculum.metrics
            print(
                f"[Curriculum] Final stage: {self.curriculum.current_stage.name} | "
                f"Episodes: {m.episodes} | Success: {m.success_rate:.2%} | "
                f"Collisions: {m.collision_rate:.2%}"
            )


class MetricsCallback(BaseCallback):
    """Logs episode-level metrics to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []
        self._successes = []
        self._collisions = []
        self._gammas = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])
                self._successes.append(float(info.get("grasped", False)))
                self._collisions.append(float(info.get("collision", False)))

            if "gamma" in info:
                self._gammas.append(info["gamma"])

        if self.num_timesteps % 1024 == 0 and len(self._episode_rewards) > 0:
            self.logger.record("custom/mean_reward", np.mean(self._episode_rewards[-100:]))
            self.logger.record("custom/mean_length", np.mean(self._episode_lengths[-100:]))
            self.logger.record("custom/success_rate", np.mean(self._successes[-100:]))
            self.logger.record("custom/collision_rate", np.mean(self._collisions[-100:]))
            if self._gammas:
                self.logger.record("custom/mean_gamma", np.mean(self._gammas[-100:]))

        return True


class CheckpointCallback(BaseCallback):
    """Periodically saves model checkpoints and belief module."""

    def __init__(
        self,
        save_freq: int,
        save_dir: str,
        belief_module=None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = Path(save_dir)
        self.belief_module = belief_module

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = self.save_dir / f"arbitration_policy_{self.num_timesteps}.zip"
            self.model.save(str(path))
            if self.verbose:
                print(f"[Checkpoint] Saved policy to {path}")

            if self.belief_module is not None:
                import torch
                bp = self.save_dir / f"bayesian_inference_{self.num_timesteps}.pt"
                torch.save(self.belief_module.state_dict(), str(bp))
                if self.verbose:
                    print(f"[Checkpoint] Saved belief module to {bp}")

        return True
