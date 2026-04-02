"""Evaluation suite for BRACE and baselines.

Metrics (matching paper Experiment 3 table):
- Success rate (%)
- Time to grasp (steps)
- Collision count per episode
- Belief accuracy at 25%, 50%, 75% path completion
- Gamma statistics (mean, constrained vs unconstrained regions)
- Belief entropy over time

Usage:
    python -m brace_kinova.evaluation.evaluate --config configs/arbitration.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import torch

from brace_kinova.envs.reach_grasp_env import ReachGraspEnv
from brace_kinova.models.bayesian_inference import BayesianGoalInference
from brace_kinova.models.expert_policy import ExpertPolicy, PotentialFieldExpert
from brace_kinova.models.simulated_human import SimulatedHuman
from brace_kinova.models.arbitration_policy import action_to_gamma


@dataclass
class EpisodeMetrics:
    success: bool = False
    collision: bool = False
    steps: int = 0
    total_reward: float = 0.0
    gammas: list = field(default_factory=list)
    beliefs: list = field(default_factory=list)
    true_goal_idx: int = 0
    dist_to_goal_over_time: list = field(default_factory=list)


@dataclass
class EvaluationResults:
    condition: str
    n_episodes: int = 0
    success_rate: float = 0.0
    mean_steps: float = 0.0
    std_steps: float = 0.0
    mean_collisions: float = 0.0
    mean_reward: float = 0.0
    mean_gamma: float = 0.0
    belief_accuracy_25: float = 0.0
    belief_accuracy_50: float = 0.0
    belief_accuracy_75: float = 0.0
    mean_belief_entropy: float = 0.0


def compute_belief_entropy(belief: np.ndarray) -> float:
    b = np.clip(belief, 1e-10, 1.0)
    return float(-np.sum(b * np.log(b)))


def evaluate_brace(
    env: ReachGraspEnv,
    policy_path: str,
    belief_module: BayesianGoalInference,
    expert,
    human: SimulatedHuman,
    n_episodes: int = 50,
    device: str = "cpu",
) -> EvaluationResults:
    """Evaluate BRACE with trained arbitration policy + belief."""
    from stable_baselines3 import PPO

    model = PPO.load(policy_path, device=device)
    episodes = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        n_goals = info["n_objects"]
        belief = np.ones(n_goals, dtype=np.float32) / n_goals
        human.reset(env._ee_pos.copy(), env.true_goal_position.copy())

        ep_metrics = EpisodeMetrics(true_goal_idx=info["true_goal_idx"])
        done = False

        while not done:
            h_action = human.get_action(
                env._ee_pos.copy(),
                env.true_goal_position.copy(),
                env._obstacle_positions if env.n_obstacles > 0 else np.zeros((0, 2)),
            )

            if belief_module is not None:
                with torch.no_grad():
                    t_h = torch.from_numpy(h_action).unsqueeze(0).to(device)
                    t_ee = torch.from_numpy(env._ee_pos).unsqueeze(0).to(device)
                    t_goals = torch.from_numpy(env._object_positions).unsqueeze(0).to(device)
                    t_prior = torch.from_numpy(belief).unsqueeze(0).to(device)
                    t_belief, _ = belief_module(t_h, t_ee, t_goals, t_prior)
                    belief = t_belief.squeeze(0).cpu().numpy()

            fused_obs = np.concatenate([obs, belief])
            action, _ = model.predict(fused_obs, deterministic=True)
            gamma = action_to_gamma(float(action[0]))

            if isinstance(expert, PotentialFieldExpert):
                w_action = expert.predict(
                    env._ee_pos.copy(),
                    env.true_goal_position.copy(),
                    env._obstacle_positions if env.n_obstacles > 0 else np.zeros((0, 2)),
                )
            else:
                w_action = expert.predict(env.get_expert_obs())

            h3d = np.array([h_action[0], h_action[1], -1.0], dtype=np.float32)
            blended = (1.0 - gamma) * h3d + gamma * w_action

            obs, reward, terminated, truncated, info = env.step(blended)
            done = terminated or truncated

            ep_metrics.steps += 1
            ep_metrics.total_reward += reward
            ep_metrics.gammas.append(gamma)
            ep_metrics.beliefs.append(belief.copy())
            ep_metrics.dist_to_goal_over_time.append(info.get("dist_to_goal", 0.0))

            if info.get("grasped", False):
                ep_metrics.success = True
            if info.get("collision", False):
                ep_metrics.collision = True

        episodes.append(ep_metrics)

    return _aggregate_results("BRACE", episodes)


def evaluate_baselines(
    env: ReachGraspEnv,
    expert,
    human: SimulatedHuman,
    n_episodes: int = 50,
    conditions: Optional[list[str]] = None,
) -> list[EvaluationResults]:
    """Evaluate baseline conditions."""
    if conditions is None:
        conditions = ["human_only", "expert_only", "fixed_gamma_0.5"]

    results = []
    for condition in conditions:
        episodes = []
        for ep in range(n_episodes):
            obs, info = env.reset()
            human.reset(env._ee_pos.copy(), env.true_goal_position.copy())
            ep_metrics = EpisodeMetrics(true_goal_idx=info["true_goal_idx"])
            done = False

            while not done:
                h_action = human.get_action(
                    env._ee_pos.copy(),
                    env.true_goal_position.copy(),
                    env._obstacle_positions if env.n_obstacles > 0 else np.zeros((0, 2)),
                )

                if condition == "human_only":
                    action = np.array([h_action[0], h_action[1], -1.0], dtype=np.float32)
                    gamma = 0.0
                elif condition == "expert_only":
                    if isinstance(expert, PotentialFieldExpert):
                        action = expert.predict(
                            env._ee_pos.copy(),
                            env.true_goal_position.copy(),
                            env._obstacle_positions if env.n_obstacles > 0 else np.zeros((0, 2)),
                        )
                    else:
                        action = expert.predict(env.get_expert_obs())
                    gamma = 1.0
                elif condition.startswith("fixed_gamma_"):
                    gamma = float(condition.split("_")[-1])
                    h3d = np.array([h_action[0], h_action[1], -1.0], dtype=np.float32)
                    if isinstance(expert, PotentialFieldExpert):
                        w_action = expert.predict(
                            env._ee_pos.copy(),
                            env.true_goal_position.copy(),
                            env._obstacle_positions if env.n_obstacles > 0 else np.zeros((0, 2)),
                        )
                    else:
                        w_action = expert.predict(env.get_expert_obs())
                    action = (1.0 - gamma) * h3d + gamma * w_action
                else:
                    action = np.zeros(3, dtype=np.float32)
                    gamma = 0.0

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_metrics.steps += 1
                ep_metrics.total_reward += reward
                ep_metrics.gammas.append(gamma)
                ep_metrics.dist_to_goal_over_time.append(info.get("dist_to_goal", 0.0))

                if info.get("grasped", False):
                    ep_metrics.success = True
                if info.get("collision", False):
                    ep_metrics.collision = True

            episodes.append(ep_metrics)

        results.append(_aggregate_results(condition, episodes))

    return results


def _aggregate_results(condition: str, episodes: list[EpisodeMetrics]) -> EvaluationResults:
    n = len(episodes)
    successes = [e.success for e in episodes]
    steps = [e.steps for e in episodes if e.success]
    collisions = [e.collision for e in episodes]
    rewards = [e.total_reward for e in episodes]
    all_gammas = [g for e in episodes for g in e.gammas]

    ba_25, ba_50, ba_75 = 0.0, 0.0, 0.0
    entropies = []

    for e in episodes:
        if len(e.beliefs) == 0:
            continue
        T = len(e.beliefs)
        for b in e.beliefs:
            entropies.append(compute_belief_entropy(b))

        for frac, store in [(0.25, []), (0.50, []), (0.75, [])]:
            idx = min(int(frac * T), T - 1)
            b = e.beliefs[idx]
            correct = int(np.argmax(b)) == e.true_goal_idx
            store.append(float(correct))

        t25 = min(int(0.25 * T), T - 1)
        t50 = min(int(0.50 * T), T - 1)
        t75 = min(int(0.75 * T), T - 1)
        ba_25 += float(np.argmax(e.beliefs[t25]) == e.true_goal_idx)
        ba_50 += float(np.argmax(e.beliefs[t50]) == e.true_goal_idx)
        ba_75 += float(np.argmax(e.beliefs[t75]) == e.true_goal_idx)

    return EvaluationResults(
        condition=condition,
        n_episodes=n,
        success_rate=np.mean(successes) if successes else 0.0,
        mean_steps=np.mean(steps) if steps else 0.0,
        std_steps=np.std(steps) if steps else 0.0,
        mean_collisions=np.mean(collisions) if collisions else 0.0,
        mean_reward=np.mean(rewards) if rewards else 0.0,
        mean_gamma=np.mean(all_gammas) if all_gammas else 0.0,
        belief_accuracy_25=ba_25 / max(n, 1),
        belief_accuracy_50=ba_50 / max(n, 1),
        belief_accuracy_75=ba_75 / max(n, 1),
        mean_belief_entropy=np.mean(entropies) if entropies else 0.0,
    )


def print_results(results: list[EvaluationResults]) -> None:
    header = (
        f"{'Condition':<20} {'Success%':>10} {'Steps':>10} {'Collisions':>12} "
        f"{'Reward':>10} {'Gamma':>8} {'BA@25%':>8} {'BA@50%':>8} {'BA@75%':>8}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.condition:<20} {r.success_rate:>9.1%} {r.mean_steps:>10.1f} "
            f"{r.mean_collisions:>12.2f} {r.mean_reward:>10.2f} {r.mean_gamma:>8.3f} "
            f"{r.belief_accuracy_25:>8.1%} {r.belief_accuracy_50:>8.1%} "
            f"{r.belief_accuracy_75:>8.1%}"
        )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Evaluate BRACE and baselines")
    parser.add_argument("--config", type=str, default="brace_kinova/configs/arbitration.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_config_path = config.get("env_config", "brace_kinova/configs/env.yaml")
    with open(env_config_path) as f:
        env_cfg = yaml.safe_load(f)

    ws = env_cfg.get("workspace", {})
    ee = env_cfg.get("end_effector", {})
    ep = env_cfg.get("episode", {})

    env = ReachGraspEnv(
        scenario="full_complexity",
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
        seed=config.get("seed", 42),
    )

    human_cfg = config.get("human_model", {})
    human = SimulatedHuman(
        noise_amplitude=human_cfg.get("noise_amplitude", 0.032),
        ar_coeff=human_cfg.get("ar_coeff", 0.5),
        seed=config.get("seed", 42),
    )

    expert_path = config.get("expert_path", "checkpoints/expert_sac.zip")
    if Path(expert_path).exists():
        expert = ExpertPolicy(expert_path, device=args.device)
    else:
        expert = PotentialFieldExpert()
        print("[Eval] Using PotentialFieldExpert fallback")

    print("[Eval] Evaluating baselines...")
    baseline_results = evaluate_baselines(
        env, expert, human, n_episodes=args.n_episodes,
        conditions=["human_only", "expert_only", "fixed_gamma_0.5"],
    )

    policy_path = config.get("training", {}).get("save_dir", "./checkpoints") + "/arbitration_policy.zip"
    belief_path = config.get("training", {}).get("save_dir", "./checkpoints") + "/bayesian_inference_finetuned.pt"
    all_results = baseline_results

    bayes_cfg = config.get("bayesian", {})
    n_goals = bayes_cfg.get("n_goals", 3)

    if Path(policy_path).exists() and Path(belief_path).exists():
        belief = BayesianGoalInference(n_goals=n_goals).to(args.device)
        belief.load_state_dict(torch.load(belief_path, map_location=args.device))
        belief.eval()

        print("[Eval] Evaluating BRACE...")
        brace_result = evaluate_brace(
            env, policy_path, belief, expert, human,
            n_episodes=args.n_episodes, device=args.device,
        )
        all_results.append(brace_result)
    else:
        print(f"[Eval] BRACE model not found at {policy_path}, skipping")

    print("\n[Eval] Results:")
    print_results(all_results)

    if args.output:
        output_data = []
        for r in all_results:
            output_data.append({
                "condition": r.condition,
                "success_rate": r.success_rate,
                "mean_steps": r.mean_steps,
                "mean_collisions": r.mean_collisions,
                "mean_reward": r.mean_reward,
                "mean_gamma": r.mean_gamma,
                "belief_accuracy_25": r.belief_accuracy_25,
                "belief_accuracy_50": r.belief_accuracy_50,
                "belief_accuracy_75": r.belief_accuracy_75,
            })
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"[Eval] Saved results to {args.output}")


if __name__ == "__main__":
    main()
