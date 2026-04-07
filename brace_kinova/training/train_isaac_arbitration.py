"""Full BRACE arbitration training in Isaac Sim: PPO + curriculum.

Implements Algorithm 1 from the paper using the Kinova Jaco2 in Isaac Lab:
1. Load pretrained expert (frozen SAC) and Bayesian inference module.
2. Initialise PPO arbitration policy.
3. For each curriculum stage, run episodes with simulated human, belief
   updates, gamma-blending, and ArbitrationReward.

Usage (from repo root, inside the Isaac Lab Python environment)::

    isaaclab -p -m brace_kinova.training.train_isaac_arbitration \\
        --config brace_kinova/configs/isaac_arbitration.yaml --headless
"""

from __future__ import annotations

import argparse
from pathlib import Path

# --- AppLauncher MUST run first -------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BRACE Arbitration (Isaac Sim)")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--config", type=str, default="brace_kinova/configs/isaac_arbitration.yaml")
# Note: --device is registered by AppLauncher. Do not add a second --device.
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Post-launch imports --------------------------------------------------
import math
import yaml
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

import isaaclab.sim as sim_utils

from brace_kinova.envs.isaac_env import (
    IsaacReachGraspEnv,
    setup_brace_scene,
)
from brace_kinova.envs.isaac_config import IsaacBraceEnvConfig, IsaacBraceSceneConfig
from brace_kinova.models.bayesian_inference import BayesianGoalInference
from brace_kinova.models.arbitration_policy import GammaArbitrationPolicy
from brace_kinova.models.expert_policy import ExpertPolicy, PotentialFieldExpert
from brace_kinova.models.simulated_human import SimulatedHuman
from brace_kinova.training.rewards import ArbitrationReward, ArbitrationRewardConfig
from brace_kinova.training.curriculum import CurriculumManager
from brace_kinova.training.callbacks import (
    CurriculumCallback,
    MetricsCallback,
    CheckpointCallback,
)
from brace_kinova.training.train_isaac_expert import build_env_config


# --------------------------------------------------------------------------
# Arbitration wrapper
# --------------------------------------------------------------------------

class BraceIsaacArbitrationEnv(IsaacReachGraspEnv):
    """Wraps ``IsaacReachGraspEnv`` with BRACE arbitration logic.

    Each step:
      1. Simulated human produces ``h_t``.
      2. Bayesian belief is updated.
      3. Observation = ``[state_features, belief_vector]``.
      4. Policy outputs ``gamma``; action is blended ``(1-γ)h + γw``.
      5. ``ArbitrationReward`` computes reward.
    """

    def __init__(
        self,
        *,
        sim,
        robot,
        config: IsaacBraceEnvConfig,
        scenario="full_complexity",
        n_goals: int = 3,
        belief_module: BayesianGoalInference | None = None,
        expert=None,
        human: SimulatedHuman | None = None,
        reward_fn: ArbitrationReward | None = None,
        belief_temperature: float = 1.0,
        torch_device: str = "cpu",
    ):
        super().__init__(sim=sim, robot=robot, config=config, scenario=scenario)
        self._torch_device = torch.device(torch_device)
        self._n_goals = n_goals

        self.belief_module = belief_module
        self.expert = expert
        self.human = human or SimulatedHuman()
        self.reward_fn = reward_fn or ArbitrationReward()
        self.belief_temperature = belief_temperature

        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_dim + n_goals,), dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._belief = np.ones(n_goals, dtype=np.float32) / n_goals

    # --- overrides --------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._belief = np.ones(self._n_goals, dtype=np.float32) / self._n_goals
        if self.human is not None:
            self.human.reset(self._ee_pos_xy.copy(), self.true_goal_position.copy())
        fused = np.concatenate([obs, self._belief])
        info["belief"] = self._belief.copy()
        return fused, info

    def step(self, action):
        gamma_raw = float(np.clip(action[0], -1.0, 1.0))
        gamma = 0.5 * (gamma_raw + 1.0)

        # Human action (2D)
        h_2d = self.human.get_action(
            self._ee_pos_xy.copy(),
            self.true_goal_position.copy(),
            self._obstacle_positions if self.n_obstacles > 0 else np.zeros((0, 2)),
        )

        # Expert action (3D)
        if self.expert is not None:
            if isinstance(self.expert, PotentialFieldExpert):
                w = self.expert.predict(
                    self._ee_pos_xy.copy(),
                    self.true_goal_position.copy(),
                    self._obstacle_positions if self.n_obstacles > 0 else np.zeros((0, 2)),
                    close_gripper=self._check_near_goal(),
                )
            else:
                w = self.expert.predict(self.get_expert_obs())
        else:
            w = np.array([h_2d[0], h_2d[1], -1.0], dtype=np.float32)

        h_3d = np.array([h_2d[0], h_2d[1], -1.0], dtype=np.float32)
        blended = (1.0 - gamma) * h_3d + gamma * w

        prev_dist = float(np.linalg.norm(self._ee_pos_xy - self.true_goal_position))
        obs, _, terminated, truncated, info = super().step(blended)
        curr_dist = float(np.linalg.norm(self._ee_pos_xy - self.true_goal_position))

        # Belief update
        if self.belief_module is not None:
            with torch.no_grad():
                t_h = torch.from_numpy(h_2d).unsqueeze(0).to(self._torch_device)
                t_ee = torch.from_numpy(self._ee_pos_xy).unsqueeze(0).to(self._torch_device)
                t_goals = torch.from_numpy(self._object_positions).unsqueeze(0).to(self._torch_device)
                t_prior = torch.from_numpy(self._belief).unsqueeze(0).to(self._torch_device)
                t_belief, _ = self.belief_module(
                    t_h, t_ee, t_goals, t_prior, tau=self.belief_temperature,
                )
                self._belief = t_belief.squeeze(0).cpu().numpy()

        # Obstacle distance
        if self.n_obstacles > 0:
            obs_dists = np.linalg.norm(
                self._obstacle_positions - self._ee_pos_xy[np.newaxis, :], axis=1,
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
        info["human_action"] = h_2d.copy()
        return fused, reward, terminated, truncated, info

    def _check_near_goal(self) -> bool:
        return float(np.linalg.norm(self._ee_pos_xy - self.true_goal_position)) < self.cfg.grasp_threshold * 2

    def set_scenario(self, scenario):
        """Override to also resize the belief-fused observation space."""
        super().set_scenario(scenario)
        base_dim = 5 + 2 * self.n_objects + 2 * self.n_obstacles + 1 + self.n_objects
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_dim + self._n_goals,), dtype=np.float32,
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_config_path = config.get("env_config", "brace_kinova/configs/isaac_env.yaml")
    with open(env_config_path) as f:
        env_cfg_yaml = yaml.safe_load(f)

    seed = config.get("seed", 42)
    device = getattr(args, "device", None) or config.get("device", "cuda:0")
    print(f"[IsaacArbitration] device={device}  seed={seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    env_config = build_env_config(env_cfg_yaml)

    # ---- Isaac Sim scene -------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(dt=env_config.physics_dt, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 0.0, 3.2], target=[0.0, 0.0, 0.5])

    scene_entities, _ = setup_brace_scene(env_config.scene)
    robot = scene_entities["kinova_j2n6s300"]

    sim.reset()
    robot.update(env_config.physics_dt)
    print("[IsaacArbitration] Scene initialised.")

    # ---- BRACE modules ---------------------------------------------------
    bayes_cfg = config.get("bayesian", {})
    ppo_cfg = config.get("ppo", {})
    reward_cfg = config.get("reward", {})
    train_cfg = config.get("training", {})
    human_cfg = config.get("human_model", {})
    curriculum_cfg = config.get("curriculum", {})

    n_goals = bayes_cfg.get("n_goals", env_cfg_yaml.get("objects", {}).get("n_objects", 3))

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
        print(f"[IsaacArbitration] Loaded belief from {belief_path}")

    expert_path = config.get("expert_path", "checkpoints/expert_sac.zip")
    if Path(expert_path).exists():
        expert = ExpertPolicy(expert_path, device=device)
        print(f"[IsaacArbitration] Loaded expert from {expert_path}")
    else:
        expert = PotentialFieldExpert()
        print("[IsaacArbitration] Using PotentialFieldExpert fallback")

    curriculum = CurriculumManager.from_yaml(curriculum_cfg)
    initial_scenario = curriculum.current_scenario.name

    human = SimulatedHuman(
        noise_amplitude=human_cfg.get("noise_amplitude", 0.032),
        ar_coeff=human_cfg.get("ar_coeff", 0.5),
        seed=seed,
    )

    arb_reward = ArbitrationReward(ArbitrationRewardConfig(
        w_collision=reward_cfg.get("w_collision", 10.0),
        w_proximity=reward_cfg.get("w_proximity", 2.5),
        w_far=reward_cfg.get("w_far", 1.5),
        w_progress=reward_cfg.get("w_progress", 3.0),
        w_autonomy=reward_cfg.get("w_autonomy", 1.5),
        w_goal=reward_cfg.get("w_goal", 2.0),
        near_threshold=reward_cfg.get("near_threshold", 0.06),
        far_threshold=reward_cfg.get("far_threshold", 0.20),
    ))

    # ---- Environment -----------------------------------------------------
    def make_env():
        env = BraceIsaacArbitrationEnv(
            sim=sim,
            robot=robot,
            config=env_config,
            scenario=initial_scenario,
            n_goals=n_goals,
            belief_module=belief_module,
            expert=expert,
            human=human,
            reward_fn=arb_reward,
            belief_temperature=bayes_cfg.get("initial_temperature", 1.0),
            torch_device=device,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    # ---- PPO -------------------------------------------------------------
    save_dir = Path(train_cfg.get("save_dir", "./checkpoints"))
    log_dir = Path(train_cfg.get("log_dir", "./logs/isaac_arbitration"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    net_arch_cfg = ppo_cfg.get("net_arch", {"pi": [256, 256], "vf": [256, 256]})
    policy_kwargs = {
        "net_arch": [dict(pi=net_arch_cfg["pi"], vf=net_arch_cfg["vf"])],
        "activation_fn": torch.nn.ReLU,
    }

    model = PPO(
        GammaArbitrationPolicy,
        vec_env,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 512),
        batch_size=ppo_cfg.get("batch_size", 512),
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

    # Curriculum callback that actually reconfigures the wrapped env
    inner_env: BraceIsaacArbitrationEnv = vec_env.envs[0].env  # type: ignore[attr-defined]

    def update_env_scenario(scenario_config):
        inner_env.set_scenario(scenario_config.name)

    callbacks = CallbackList([
        CurriculumCallback(curriculum, env_update_fn=update_env_scenario),
        MetricsCallback(),
        CheckpointCallback(
            save_freq=train_cfg.get("save_freq", 25_000),
            save_dir=str(save_dir),
            belief_module=belief_module,
        ),
    ])

    total = train_cfg.get("total_timesteps", 1_000_000)
    print(f"[IsaacArbitration] Training PPO for {total} timesteps …")
    print(f"[IsaacArbitration] Starting stage: {curriculum.current_stage.name}")

    model.learn(total_timesteps=total, callback=callbacks, progress_bar=True)

    model.save(str(save_dir / "arbitration_policy"))
    torch.save(belief_module.state_dict(), str(save_dir / "bayesian_inference_finetuned.pt"))
    print(f"[IsaacArbitration] Saved policy  → {save_dir / 'arbitration_policy.zip'}")
    print(f"[IsaacArbitration] Saved belief   → {save_dir / 'bayesian_inference_finetuned.pt'}")

    print("\n[IsaacArbitration] Curriculum summary:")
    for i, stage in enumerate(curriculum.stages):
        m = curriculum._metrics[i]
        status = "COMPLETED" if i < curriculum.current_stage_idx else (
            "ACTIVE" if i == curriculum.current_stage_idx else "PENDING"
        )
        print(
            f"  Stage {i}: {stage.name} [{status}]  "
            f"Episodes={m.episodes}  Success={m.success_rate:.2%}  "
            f"Collisions={m.collision_rate:.2%}"
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
