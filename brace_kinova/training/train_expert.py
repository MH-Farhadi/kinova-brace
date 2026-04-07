"""Train SAC expert for the reach-and-grasp environment.

Usage:
    python -m brace_kinova.training.train_expert --config configs/expert.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from brace_kinova.envs.reach_grasp_env import ExpertReachGraspEnv
from brace_kinova.envs.scenarios import SCENARIOS


class WallClockCheckpointCallback(BaseCallback):
    """Save model/replay-buffer/VecNormalize state every N seconds."""

    def __init__(self, save_dir: Path, interval_seconds: int = 3600, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.save_dir = save_dir
        self.interval_seconds = int(interval_seconds)
        self._last_save_time = time.time()

    def _on_step(self) -> bool:
        now = time.time()
        if now - self._last_save_time < self.interval_seconds:
            return True

        self.save_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        step = int(self.num_timesteps)
        model_path = self.save_dir / f"expert_sac_time_{stamp}_{step}.zip"
        replay_path = self.save_dir / f"expert_sac_time_{stamp}_{step}.replay_buffer.pkl"
        norm_path = self.save_dir / f"expert_sac_time_{stamp}_{step}.vecnormalize.pkl"

        self.model.save(str(model_path))
        self.model.save_replay_buffer(str(replay_path))
        env = self.model.get_env()
        if env is not None:
            env.save(str(norm_path))

        self._last_save_time = now
        if self.verbose:
            print(f"[Expert] Wall-clock checkpoint saved: {model_path.name}")
        return True


def make_env(env_cfg: dict, scenario: str = "basic_reaching", seed: int = 0):
    """Factory for creating expert training environments."""

    def _init():
        env = ExpertReachGraspEnv(
            scenario=scenario,
            max_steps=env_cfg.get("max_steps", 200),
            max_velocity=env_cfg.get("max_velocity", 0.15),
            workspace_bounds=env_cfg,
            grasp_threshold=env_cfg.get("grasp_threshold", 0.04),
            seed=seed,
        )
        return Monitor(env)

    return _init


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def main():
    parser = argparse.ArgumentParser(description="Train SAC expert for BRACE")
    parser.add_argument("--config", type=str, default="brace_kinova/configs/expert.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume_model", type=str, default=None)
    parser.add_argument("--resume_replay_buffer", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_config_path = config.get("env_config", "brace_kinova/configs/env.yaml")
    with open(env_config_path) as f:
        env_cfg = yaml.safe_load(f)

    seed = config.get("seed", 42)
    device = resolve_device(args.device or config.get("device", "auto"))
    print(f"[Expert] Using device: {device}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_envs = config.get("training", {}).get("n_envs", 4)
    ws = env_cfg.get("workspace", {})
    ee = env_cfg.get("end_effector", {})
    ep = env_cfg.get("episode", {})

    env_params = {
        "x_min": ws.get("x_min", 0.20),
        "x_max": ws.get("x_max", 0.60),
        "y_min": ws.get("y_min", -0.30),
        "y_max": ws.get("y_max", 0.45),
        "max_steps": ep.get("max_steps", 200),
        "max_velocity": ee.get("max_velocity", 0.15),
        "initial_x": ee.get("initial_x", 0.30),
        "initial_y": ee.get("initial_y", 0.0),
        "dt": ep.get("dt", 0.05),
        "grasp_threshold": env_cfg.get("objects", {}).get("grasp_threshold", 0.04),
    }

    env_fns = [make_env(env_params, scenario="basic_reaching", seed=seed + i) for i in range(n_envs)]
    if n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([make_env(env_params, scenario="basic_reaching", seed=seed + 100)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    sac_cfg = config.get("sac", {})
    training_cfg = config.get("training", {})

    save_dir = Path(training_cfg.get("save_dir", "./checkpoints"))
    log_dir = Path(training_cfg.get("log_dir", "./logs/expert"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_model:
        print(f"[Expert] Resuming model from: {args.resume_model}")
        model = SAC.load(args.resume_model, env=vec_env, device=device)
        if args.resume_replay_buffer:
            print(f"[Expert] Loading replay buffer: {args.resume_replay_buffer}")
            model.load_replay_buffer(args.resume_replay_buffer)
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=sac_cfg.get("learning_rate", 3e-4),
            buffer_size=sac_cfg.get("buffer_size", 400_000),
            batch_size=sac_cfg.get("batch_size", 1024),
            tau=sac_cfg.get("tau", 0.005),
            gamma=sac_cfg.get("gamma", 0.99),
            train_freq=sac_cfg.get("train_freq", 64),
            gradient_steps=sac_cfg.get("gradient_steps", 64),
            learning_starts=sac_cfg.get("learning_starts", 10_000),
            policy_kwargs={"net_arch": sac_cfg.get("net_arch", [256, 256, 256])},
            tensorboard_log=str(log_dir),
            device=device,
            seed=seed,
            verbose=1,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(log_dir),
        eval_freq=training_cfg.get("eval_freq", 20_000),
        n_eval_episodes=training_cfg.get("eval_episodes", 20),
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=training_cfg.get("save_freq", 50_000),
        save_path=str(save_dir),
        name_prefix="expert_sac",
    )
    wallclock_callback = WallClockCheckpointCallback(
        save_dir=save_dir,
        interval_seconds=training_cfg.get("checkpoint_every_seconds", 3600),
    )

    total_timesteps = training_cfg.get("total_timesteps", 1_000_000)
    print(f"[Expert] Training SAC for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, wallclock_callback],
        progress_bar=True,
        reset_num_timesteps=not bool(args.resume_model),
    )

    final_path = save_dir / "expert_sac"
    model.save(str(final_path))
    vec_env.save(str(save_dir / "expert_vecnormalize.pkl"))
    print(f"[Expert] Saved model to {final_path}.zip")
    print(f"[Expert] Saved VecNormalize stats to {save_dir / 'expert_vecnormalize.pkl'}")


if __name__ == "__main__":
    main()
