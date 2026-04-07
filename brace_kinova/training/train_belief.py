"""Pretrain the Bayesian Goal Inference module via supervised NLL.

Generates synthetic trajectories using the simulated human model, then
optimizes L = -log P(g* | X, H) over those trajectories.

Usage:
    python -m brace_kinova.training.train_belief --config configs/belief.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from brace_kinova.models.bayesian_inference import BayesianGoalInference
from brace_kinova.models.simulated_human import SimulatedHuman


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def generate_trajectories(
    n_trajectories: int,
    trajectory_length: int,
    n_goals: int,
    env_cfg: dict,
    human_cfg: dict,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic trajectories for belief pretraining.

    Returns:
        ee_positions: (n_traj, traj_len, 2)
        human_actions: (n_traj, traj_len, 2)
        goal_positions: (n_traj, n_goals, 2)
        true_goal_indices: (n_traj,)
    """
    rng = np.random.default_rng(seed)

    ws = env_cfg.get("workspace", {})
    x_min, x_max = ws.get("x_min", 0.20), ws.get("x_max", 0.60)
    y_min, y_max = ws.get("y_min", -0.30), ws.get("y_max", 0.45)

    ee = env_cfg.get("end_effector", {})
    max_vel = ee.get("max_velocity", 0.15)
    dt = env_cfg.get("episode", {}).get("dt", 0.05)

    human = SimulatedHuman(
        noise_amplitude=human_cfg.get("noise_amplitude", 0.032),
        ar_coeff=human_cfg.get("ar_coeff", 0.5),
        seed=seed,
    )

    all_ee = np.zeros((n_trajectories, trajectory_length, 2), dtype=np.float32)
    all_actions = np.zeros((n_trajectories, trajectory_length, 2), dtype=np.float32)
    all_goals = np.zeros((n_trajectories, n_goals, 2), dtype=np.float32)
    all_true_idx = np.zeros(n_trajectories, dtype=np.int64)

    for i in range(n_trajectories):
        goals = np.column_stack([
            rng.uniform(x_min + 0.05, x_max - 0.05, n_goals),
            rng.uniform(y_min + 0.05, y_max - 0.05, n_goals),
        ]).astype(np.float32)
        true_idx = rng.integers(0, n_goals)
        goal = goals[true_idx]

        ee_pos = np.array([
            rng.uniform(x_min, x_min + 0.15),
            rng.uniform(y_min + 0.1, y_max - 0.1),
        ], dtype=np.float32)

        human.reset(ee_pos, goal)
        empty_obstacles = np.zeros((0, 2), dtype=np.float32)

        for t in range(trajectory_length):
            all_ee[i, t] = ee_pos
            action = human.get_action(ee_pos, goal, empty_obstacles)
            all_actions[i, t] = action

            ee_pos = ee_pos + action * max_vel * dt
            ee_pos[0] = np.clip(ee_pos[0], x_min, x_max)
            ee_pos[1] = np.clip(ee_pos[1], y_min, y_max)

        all_goals[i] = goals
        all_true_idx[i] = true_idx

    return all_ee, all_actions, all_goals, all_true_idx


def main():
    parser = argparse.ArgumentParser(description="Pretrain Bayesian Inference")
    parser.add_argument("--config", type=str, default="brace_kinova/configs/belief.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_config_path = config.get("env_config", "brace_kinova/configs/env.yaml")
    with open(env_config_path) as f:
        env_cfg = yaml.safe_load(f)

    seed = config.get("seed", 42)
    device = torch.device(resolve_device(args.device or config.get("device", "auto")))
    print(f"[Belief] Using device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    bayes_cfg = config.get("bayesian", {})
    train_cfg = config.get("training", {})
    human_cfg = config.get("human_model", {})

    n_goals = bayes_cfg.get("n_goals", 3)
    n_traj = train_cfg.get("n_trajectories", 5000)
    traj_len = train_cfg.get("trajectory_length", 100)

    print(f"[Belief] Generating {n_traj} trajectories of length {traj_len}...")
    ee_pos, h_actions, goals, true_idx = generate_trajectories(
        n_traj, traj_len, n_goals, env_cfg, human_cfg, seed
    )

    model = BayesianGoalInference(
        n_goals=n_goals,
        initial_beta=bayes_cfg.get("initial_beta", 2.0),
        initial_w_theta=bayes_cfg.get("initial_w_theta", 0.8),
        initial_w_dist=bayes_cfg.get("initial_w_dist", 0.2),
        ema_alpha=bayes_cfg.get("ema_alpha", 0.85),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg.get("learning_rate", 5e-3))
    epochs = train_cfg.get("epochs", 200)
    batch_size = train_cfg.get("batch_size", 512)
    checkpoint_every_seconds = int(train_cfg.get("checkpoint_every_seconds", 3600))

    t_ee = torch.from_numpy(ee_pos).to(device)
    t_actions = torch.from_numpy(h_actions).to(device)
    t_goals = torch.from_numpy(goals).to(device)
    t_true_idx = torch.from_numpy(true_idx).to(device)

    n_batches = (n_traj + batch_size - 1) // batch_size

    start_epoch = 0
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[Belief] Resumed from {args.resume_checkpoint} at epoch {start_epoch}")

    print(f"[Belief] Training for {epochs} epochs, batch_size={batch_size}, start_epoch={start_epoch}")
    save_dir = Path(train_cfg.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    next_wallclock_save = time.time() + checkpoint_every_seconds

    for epoch in range(start_epoch, epochs):
        perm = torch.randperm(n_traj, device=device)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            b_ee = t_ee[idx]           # (B, T, 2)
            b_act = t_actions[idx]     # (B, T, 2)
            b_goals = t_goals[idx]     # (B, G, 2)
            b_true = t_true_idx[idx]   # (B,)

            bs = b_ee.shape[0]
            belief = model.get_uniform_prior(bs, device)
            total_nll = torch.tensor(0.0, device=device)

            for t in range(traj_len):
                belief, nll = model(
                    human_action=b_act[:, t],
                    ee_position=b_ee[:, t],
                    goal_positions=b_goals,
                    prior_belief=belief,
                    true_goal_idx=b_true,
                    tau=1.0,
                )
                total_nll = total_nll + nll
                belief = belief.detach()

            loss = total_nll / traj_len

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | NLL: {avg_loss:.4f} | "
                f"beta={model.beta.item():.3f} w_th={model.w_theta.item():.3f} "
                f"w_d={model.w_dist.item():.3f}"
            )
        if time.time() >= next_wallclock_save:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            ckpt_path = save_dir / f"bayesian_inference_time_{stamp}_epoch{epoch+1}.ckpt.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config_path": args.config,
                },
                str(ckpt_path),
            )
            print(f"[Belief] Wall-clock checkpoint saved: {ckpt_path}")
            next_wallclock_save = time.time() + checkpoint_every_seconds

    save_path = Path(train_cfg.get("save_path", "checkpoints/bayesian_inference.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), str(save_path))
    print(f"[Belief] Saved model to {save_path}")
    print(
        f"[Belief] Final params: beta={model.beta.item():.4f}, "
        f"w_theta={model.w_theta.item():.4f}, w_dist={model.w_dist.item():.4f}"
    )


if __name__ == "__main__":
    main()
