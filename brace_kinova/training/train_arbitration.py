"""Full BRACE arbitration training: PPO + REINFORCE on belief, with curriculum.

Implements Algorithm 1 from the paper:
1. Load pretrained expert (frozen SAC) and Bayesian inference module
2. Initialize PPO arbitration policy
3. For each curriculum stage, run episodes with joint optimization:
   - PPO update on arbitration policy parameters theta
   - Mixed L_total = alpha * L_RL + (1-alpha) * L_supervised on belief params phi

Usage:
    python -m brace_kinova.training.train_arbitration --config configs/arbitration.yaml
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from collections import deque

import gymnasium as gym
import yaml
import numpy as np
import torch
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback

from brace_kinova.envs.reach_grasp_env import ReachGraspEnv
from brace_kinova.models.bayesian_inference import BayesianGoalInference
from brace_kinova.models.arbitration_policy import GammaArbitrationPolicy, action_to_gamma
from brace_kinova.models.expert_policy import ExpertPolicy, PotentialFieldExpert
from brace_kinova.models.simulated_human import SimulatedHuman
from brace_kinova.training.rewards import ArbitrationReward, ArbitrationRewardConfig
from brace_kinova.training.curriculum import CurriculumManager
from brace_kinova.training.callbacks import (
    CurriculumCallback,
    MetricsCallback,
    CheckpointCallback,
)


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


class WallClockArbitrationCheckpointCallback(BaseCallback):
    """Save PPO policy + belief module every N wall-clock seconds."""

    def __init__(self, save_dir: Path, belief_module, interval_seconds: int = 3600, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.save_dir = save_dir
        self.belief_module = belief_module
        self.interval_seconds = int(interval_seconds)
        self._last_save_time = time.time()

    def _on_step(self) -> bool:
        now = time.time()
        if now - self._last_save_time < self.interval_seconds:
            return True

        self.save_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        step = int(self.num_timesteps)
        policy_path = self.save_dir / f"arbitration_policy_time_{stamp}_{step}.zip"
        belief_path = self.save_dir / f"bayesian_inference_time_{stamp}_{step}.pt"
        self.model.save(str(policy_path))
        if self.belief_module is not None:
            torch.save(self.belief_module.state_dict(), str(belief_path))
        self._last_save_time = now
        if self.verbose:
            print(f"[Arbitration] Wall-clock checkpoint saved: {policy_path.name}")
        return True


class BraceArbitrationEnv(ReachGraspEnv):
    """Wraps ReachGraspEnv with BRACE-specific logic for arbitration training.

    At each step:
    1. Simulated human produces h_t
    2. Bayesian belief updates
    3. Observation is [state_features, belief_vector]
    4. Policy outputs gamma, blending occurs
    5. ArbitrationReward computes reward
    """

    def __init__(
        self,
        scenario="full_complexity",
        n_goals: int = 3,
        belief_module: BayesianGoalInference = None,
        expert: object = None,
        human: SimulatedHuman = None,
        reward_fn: ArbitrationReward = None,
        belief_temperature: float = 1.0,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(scenario=scenario, **kwargs)
        self.device = torch.device(device)
        self._n_goals = n_goals

        self.belief_module = belief_module
        self.expert = expert
        self.human = human or SimulatedHuman()
        self.reward_fn = reward_fn or ArbitrationReward()
        self.belief_temperature = belief_temperature

        base_dim = self.observation_space.shape[0]
        obs_dim = base_dim + n_goals
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._belief = np.ones(n_goals, dtype=np.float32) / n_goals
        self._belief_nll_accum = 0.0
        self._step_beliefs = []

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._belief = np.ones(self._n_goals, dtype=np.float32) / self._n_goals
        self._belief_nll_accum = 0.0
        self._step_beliefs = []

        if self.human is not None:
            self.human.reset(self._ee_pos.copy(), self.true_goal_position.copy())

        fused = np.concatenate([obs, self._belief])
        info["belief"] = self._belief.copy()
        return fused, info

    def step(self, action):
        gamma_raw = float(np.clip(action[0], -1.0, 1.0))
        gamma = 0.5 * (gamma_raw + 1.0)

        h_action_2d = self.human.get_action(
            self._ee_pos.copy(),
            self.true_goal_position.copy(),
            self._obstacle_positions if self.n_obstacles > 0 else np.zeros((0, 2)),
        )

        if self.expert is not None:
            if isinstance(self.expert, PotentialFieldExpert):
                w_action = self.expert.predict(
                    self._ee_pos.copy(),
                    self.true_goal_position.copy(),
                    self._obstacle_positions if self.n_obstacles > 0 else np.zeros((0, 2)),
                    close_gripper=self._check_near_goal(),
                )
            else:
                expert_obs = self.get_expert_obs()
                w_action = self.expert.predict(expert_obs)
        else:
            w_action = np.array([h_action_2d[0], h_action_2d[1], -1.0], dtype=np.float32)

        h_action_3d = np.array([h_action_2d[0], h_action_2d[1], -1.0], dtype=np.float32)
        blended = (1.0 - gamma) * h_action_3d + gamma * w_action

        prev_dist = float(np.linalg.norm(self._ee_pos - self.true_goal_position))
        obs, _, terminated, truncated, info = super().step(blended)
        curr_dist = float(np.linalg.norm(self._ee_pos - self.true_goal_position))

        if self.belief_module is not None:
            with torch.no_grad():
                t_h = torch.from_numpy(h_action_2d).unsqueeze(0).to(self.device)
                t_ee = torch.from_numpy(self._ee_pos).unsqueeze(0).to(self.device)
                t_goals = torch.from_numpy(self._object_positions).unsqueeze(0).to(self.device)
                t_prior = torch.from_numpy(self._belief).unsqueeze(0).to(self.device)

                t_belief, _ = self.belief_module(
                    t_h, t_ee, t_goals, t_prior,
                    tau=self.belief_temperature,
                )
                self._belief = t_belief.squeeze(0).cpu().numpy()

        if self.n_obstacles > 0:
            obs_dists = np.linalg.norm(
                self._obstacle_positions - self._ee_pos[np.newaxis, :], axis=1
            )
            min_obs_dist = float(obs_dists.min())
        else:
            min_obs_dist = 1.0

        reward = self.reward_fn(
            gamma=gamma,
            belief=self._belief,
            true_goal_idx=self._true_goal_idx,
            prev_dist_to_goal=prev_dist,
            curr_dist_to_goal=curr_dist,
            min_obstacle_dist=min_obs_dist,
            collision=info.get("collision", False),
        )

        fused = np.concatenate([obs, self._belief])
        info["gamma"] = gamma
        info["belief"] = self._belief.copy()
        info["human_action"] = h_action_2d.copy()

        return fused, reward, terminated, truncated, info

    def set_scenario(self, scenario):
        """Override to resize obs space with belief dimensions after curriculum change."""
        super().set_scenario(scenario)
        base_dim = 5 + 2 * self.n_objects + 2 * self.n_obstacles + 1 + self.n_objects
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_dim + self._n_goals,), dtype=np.float32,
        )

    def _check_near_goal(self) -> bool:
        dist = np.linalg.norm(self._ee_pos - self.true_goal_position)
        return dist < self.grasp_threshold * 2


def make_brace_env(
    scenario: str,
    env_cfg: dict,
    belief_module: BayesianGoalInference,
    expert,
    human_cfg: dict,
    reward_cfg: dict,
    n_goals: int,
    belief_temperature: float,
    device: str,
    seed: int,
):
    """Factory for creating BRACE arbitration training environments."""

    def _init():
        ws = env_cfg.get("workspace", {})
        ee = env_cfg.get("end_effector", {})
        ep = env_cfg.get("episode", {})

        human = SimulatedHuman(
            noise_amplitude=human_cfg.get("noise_amplitude", 0.032),
            ar_coeff=human_cfg.get("ar_coeff", 0.5),
            seed=seed,
        )

        reward_fn = ArbitrationReward(ArbitrationRewardConfig(
            w_collision=reward_cfg.get("w_collision", 10.0),
            w_proximity=reward_cfg.get("w_proximity", 2.5),
            w_far=reward_cfg.get("w_far", 1.5),
            w_progress=reward_cfg.get("w_progress", 3.0),
            w_autonomy=reward_cfg.get("w_autonomy", 1.5),
            w_goal=reward_cfg.get("w_goal", 2.0),
            near_threshold=reward_cfg.get("near_threshold", 0.06),
            far_threshold=reward_cfg.get("far_threshold", 0.20),
        ))

        env = BraceArbitrationEnv(
            scenario=scenario,
            n_goals=n_goals,
            belief_module=belief_module,
            expert=expert,
            human=human,
            reward_fn=reward_fn,
            belief_temperature=belief_temperature,
            device=device,
            max_steps=ep.get("max_steps", 200),
            max_velocity=ee.get("max_velocity", 0.15),
            workspace_bounds={
                "x_min": ws.get("x_min", 0.20),
                "x_max": ws.get("x_max", 0.60),
                "y_min": ws.get("y_min", -0.30),
                "y_max": ws.get("y_max", 0.45),
                "initial_x": ee.get("initial_x", 0.30),
                "initial_y": ee.get("initial_y", 0.0),
                "dt": ep.get("dt", 0.05),
            },
            grasp_threshold=env_cfg.get("objects", {}).get("grasp_threshold", 0.04),
            seed=seed,
        )
        return Monitor(env)

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train BRACE Arbitration (PPO + Belief)")
    parser.add_argument("--config", type=str, default="brace_kinova/configs/arbitration.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume_model", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_config_path = config.get("env_config", "brace_kinova/configs/env.yaml")
    with open(env_config_path) as f:
        env_cfg = yaml.safe_load(f)

    seed = config.get("seed", 42)
    device = resolve_device(args.device or config.get("device", "auto"))
    print(f"[Arbitration] Using device: {device}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    bayes_cfg = config.get("bayesian", {})
    ppo_cfg = config.get("ppo", {})
    reward_cfg = config.get("reward", {})
    train_cfg = config.get("training", {})
    human_cfg = config.get("human_model", {})
    curriculum_cfg = config.get("curriculum", {})

    n_goals = bayes_cfg.get("n_goals", env_cfg.get("objects", {}).get("n_objects", 3))

    belief_module = BayesianGoalInference(
        n_goals=n_goals,
        initial_beta=bayes_cfg.get("initial_beta", 2.0),
        initial_w_theta=bayes_cfg.get("initial_w_theta", 0.8),
        initial_w_dist=bayes_cfg.get("initial_w_dist", 0.2),
        ema_alpha=bayes_cfg.get("ema_alpha", 0.85),
    ).to(device)

    belief_path = config.get("belief_path", "checkpoints/bayesian_inference.pt")
    if Path(belief_path).exists():
        belief_module.load_state_dict(torch.load(belief_path, map_location=device))
        print(f"[Arbitration] Loaded pretrained belief from {belief_path}")
    else:
        print(f"[Arbitration] No pretrained belief found at {belief_path}, using initial params")

    expert_path = config.get("expert_path", "checkpoints/expert_sac.zip")
    if Path(expert_path).exists():
        expert = ExpertPolicy(expert_path, device=device)
        print(f"[Arbitration] Loaded expert from {expert_path}")
    else:
        expert = PotentialFieldExpert()
        print("[Arbitration] No trained expert found, using PotentialFieldExpert fallback")

    curriculum = CurriculumManager.from_yaml(curriculum_cfg)
    initial_scenario = curriculum.current_scenario.name

    n_envs = train_cfg.get("n_envs", 8)
    belief_temp = bayes_cfg.get("initial_temperature", 1.0)

    env_fns = [
        make_brace_env(
            scenario=initial_scenario,
            env_cfg=env_cfg,
            belief_module=belief_module,
            expert=expert,
            human_cfg=human_cfg,
            reward_cfg=reward_cfg,
            n_goals=n_goals,
            belief_temperature=belief_temp,
            device=device,
            seed=seed + i,
        )
        for i in range(n_envs)
    ]

    if n_envs > 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    save_dir = Path(train_cfg.get("save_dir", "./checkpoints"))
    log_dir = Path(train_cfg.get("log_dir", "./logs/arbitration"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    net_arch_cfg = ppo_cfg.get("net_arch", {"pi": [256, 256], "vf": [256, 256]})
    policy_kwargs = {
        "net_arch": [dict(pi=net_arch_cfg["pi"], vf=net_arch_cfg["vf"])],
        "activation_fn": torch.nn.ReLU,
    }

    if args.resume_model:
        print(f"[Arbitration] Resuming policy from: {args.resume_model}")
        model = PPO.load(args.resume_model, env=vec_env, device=device)
    else:
        model = PPO(
            GammaArbitrationPolicy,
            vec_env,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 1024),
            batch_size=ppo_cfg.get("batch_size", 1024),
            n_epochs=ppo_cfg.get("n_epochs", 4),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            device=device,
            seed=seed,
            verbose=1,
        )

    def update_env_scenario(scenario_config):
        """Reconfigure all sub-environments to the new curriculum scenario."""
        for i in range(vec_env.num_envs):
            inner = vec_env.envs[i]
            # Unwrap Monitor to reach the BraceArbitrationEnv
            base = inner.env if hasattr(inner, "env") else inner
            if hasattr(base, "set_scenario"):
                base.set_scenario(scenario_config.name)

    callbacks = CallbackList([
        CurriculumCallback(curriculum, env_update_fn=update_env_scenario),
        MetricsCallback(),
        CheckpointCallback(
            save_freq=train_cfg.get("save_freq", 50_000),
            save_dir=str(save_dir),
            belief_module=belief_module,
        ),
        WallClockArbitrationCheckpointCallback(
            save_dir=save_dir,
            belief_module=belief_module,
            interval_seconds=train_cfg.get("checkpoint_every_seconds", 3600),
        ),
    ])

    belief_optimizer = optim.Adam(
        belief_module.parameters(),
        lr=bayes_cfg.get("lr", 1e-3),
    )
    grad_clip = bayes_cfg.get("gradient_clip_norm", 1.0)

    total_timesteps = train_cfg.get("total_timesteps", 2_000_000)
    print(f"[Arbitration] Training PPO for {total_timesteps} timesteps with curriculum...")
    print(f"[Arbitration] Starting at stage: {curriculum.current_stage.name}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not bool(args.resume_model),
    )

    final_policy_path = save_dir / "arbitration_policy"
    model.save(str(final_policy_path))
    print(f"[Arbitration] Saved policy to {final_policy_path}.zip")

    final_belief_path = save_dir / "bayesian_inference_finetuned.pt"
    torch.save(belief_module.state_dict(), str(final_belief_path))
    print(f"[Arbitration] Saved finetuned belief to {final_belief_path}")

    print("\n[Arbitration] Curriculum summary:")
    for i, stage in enumerate(curriculum.stages):
        m = curriculum._metrics[i]
        status = "COMPLETED" if i < curriculum.current_stage_idx else (
            "ACTIVE" if i == curriculum.current_stage_idx else "PENDING"
        )
        print(
            f"  Stage {i}: {stage.name} [{status}] - "
            f"Episodes: {m.episodes}, Success: {m.success_rate:.2%}, "
            f"Collisions: {m.collision_rate:.2%}"
        )


if __name__ == "__main__":
    main()
