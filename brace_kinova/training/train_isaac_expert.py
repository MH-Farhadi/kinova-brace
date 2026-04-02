"""Train SAC expert for the BRACE reach-and-grasp task in Isaac Sim.

Launches Isaac Sim via AppLauncher, builds the Kinova Jaco2 tabletop
scene, then trains a SAC policy on ``IsaacExpertReachGraspEnv``.

Usage (from repo root, inside the Isaac Lab Python environment)::

    isaaclab -p -m brace_kinova.training.train_isaac_expert \\
        --config brace_kinova/configs/isaac_expert.yaml --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- AppLauncher MUST run before any Isaac Lab imports --------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SAC expert (Isaac Sim)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config", type=str, default="brace_kinova/configs/isaac_expert.yaml")
parser.add_argument("--device", type=str, default=None)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Safe to import Isaac Lab / BRACE modules now -------------------------
import yaml
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import isaaclab.sim as sim_utils

from brace_kinova.envs.isaac_env import (
    IsaacExpertReachGraspEnv,
    setup_brace_scene,
)
from brace_kinova.envs.isaac_config import IsaacBraceEnvConfig, IsaacBraceSceneConfig


def build_env_config(env_cfg: dict) -> IsaacBraceEnvConfig:
    """Construct an ``IsaacBraceEnvConfig`` from a parsed YAML dict."""
    ws = env_cfg.get("workspace", {})
    obj = env_cfg.get("objects", {})
    obs = env_cfg.get("obstacles", {})
    ee = env_cfg.get("end_effector", {})
    ep = env_cfg.get("episode", {})
    sc = env_cfg.get("scene", {})
    ik = env_cfg.get("ik", {})
    gr = env_cfg.get("gripper", {})

    scene_cfg = IsaacBraceSceneConfig(
        table_height=sc.get("table_height", 0.80),
        robot_base_height=sc.get("robot_base_height", 0.80),
        object_spawn_z=sc.get("object_spawn_z", 0.825),
    )

    return IsaacBraceEnvConfig(
        ee_link_name=ee.get("ee_link_name", "j2n6s300_end_effector"),
        x_min=ws.get("x_min", 0.20),
        x_max=ws.get("x_max", 0.60),
        y_min=ws.get("y_min", -0.30),
        y_max=ws.get("y_max", 0.45),
        z_fixed=ws.get("z_fixed", 0.10),
        max_velocity=ee.get("max_velocity", 0.15),
        ee_initial_x=ee.get("initial_x", 0.30),
        ee_initial_y=ee.get("initial_y", 0.0),
        max_n_objects=obj.get("n_objects", 3),
        max_n_obstacles=obs.get("n_obstacles", 4),
        grasp_threshold=obj.get("grasp_threshold", 0.04),
        collision_threshold=obs.get("collision_threshold", 0.045),
        object_x_range=(obj.get("spawn_x_min", 0.25), obj.get("spawn_x_max", 0.55)),
        object_y_range=(obj.get("spawn_y_min", -0.25), obj.get("spawn_y_max", 0.40)),
        min_object_separation=obj.get("min_separation", 0.10),
        min_obstacle_object_separation=obs.get("min_separation_from_objects", 0.08),
        max_steps=ep.get("max_steps", 200),
        control_dt=ep.get("control_dt", 0.05),
        physics_dt=ep.get("physics_dt", 1.0 / 240.0),
        decimation=ep.get("decimation", 12),
        stabilize_steps=ep.get("stabilize_steps", 60),
        ik_method=ik.get("method", "dls"),
        gripper_open_pos=gr.get("open_pos", 0.0),
        gripper_close_pos=gr.get("close_pos", 1.2),
        scene=scene_cfg,
        seed=env_cfg.get("seed", 42),
    )


def main() -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_config_path = config.get("env_config", "brace_kinova/configs/isaac_env.yaml")
    with open(env_config_path) as f:
        env_cfg_yaml = yaml.safe_load(f)

    seed = config.get("seed", 42)
    device = config.get("device", "cuda:0")
    print(f"[IsaacExpert] device={device}  seed={seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build env config
    env_config = build_env_config(env_cfg_yaml)

    # Reward overrides from training config
    reward_cfg = config.get("reward", {})
    env_config.reward_progress_weight = reward_cfg.get("progress_weight", 3.0)
    env_config.reward_heading_penalty = reward_cfg.get("heading_penalty", 0.8)
    env_config.reward_obstacle_penalty = reward_cfg.get("obstacle_penalty", 2.5)
    env_config.reward_d_safe = reward_cfg.get("d_safe", 0.06)
    env_config.reward_goal_bonus = reward_cfg.get("goal_bonus", 5.0)
    env_config.reward_collision_penalty = reward_cfg.get("collision_penalty", 10.0)

    # ---- Build Isaac Sim scene -------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(
        dt=env_config.physics_dt,
        device=device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 0.0, 3.2], target=[0.0, 0.0, 0.5])

    scene_entities, _origins = setup_brace_scene(env_config.scene)
    robot = scene_entities["kinova_j2n6s300"]

    sim.reset()
    robot.update(env_config.physics_dt)
    print("[IsaacExpert] Scene initialised.")

    # ---- Create environment ----------------------------------------------
    scenario = config.get("scenario", "basic_reaching")

    def make_env():
        env = IsaacExpertReachGraspEnv(
            sim=sim, robot=robot, config=env_config, scenario=scenario,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # ---- SAC -------------------------------------------------------------
    sac_cfg = config.get("sac", {})
    training_cfg = config.get("training", {})

    save_dir = Path(training_cfg.get("save_dir", "./checkpoints"))
    log_dir = Path(training_cfg.get("log_dir", "./logs/isaac_expert"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=sac_cfg.get("learning_rate", 3e-4),
        buffer_size=sac_cfg.get("buffer_size", 200_000),
        batch_size=sac_cfg.get("batch_size", 512),
        tau=sac_cfg.get("tau", 0.005),
        gamma=sac_cfg.get("gamma", 0.99),
        train_freq=sac_cfg.get("train_freq", 32),
        gradient_steps=sac_cfg.get("gradient_steps", 32),
        learning_starts=sac_cfg.get("learning_starts", 5_000),
        policy_kwargs={"net_arch": sac_cfg.get("net_arch", [256, 256, 256])},
        tensorboard_log=str(log_dir),
        device=device,
        seed=seed,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(log_dir),
        eval_freq=training_cfg.get("eval_freq", 10_000),
        n_eval_episodes=training_cfg.get("eval_episodes", 10),
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=training_cfg.get("save_freq", 25_000),
        save_path=str(save_dir),
        name_prefix="isaac_expert_sac",
    )

    total = training_cfg.get("total_timesteps", 500_000)
    print(f"[IsaacExpert] Training SAC for {total} timesteps …")

    model.learn(total_timesteps=total, callback=[eval_cb, ckpt_cb], progress_bar=True)

    model.save(str(save_dir / "expert_sac"))
    vec_env.save(str(save_dir / "expert_vecnormalize.pkl"))
    print(f"[IsaacExpert] Saved model  → {save_dir / 'expert_sac.zip'}")
    print(f"[IsaacExpert] Saved stats  → {save_dir / 'expert_vecnormalize.pkl'}")

    simulation_app.close()


if __name__ == "__main__":
    main()
