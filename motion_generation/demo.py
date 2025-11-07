from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from isaaclab.app import AppLauncher

# Ensure project root on sys.path for modular imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from controllers import (
    CartesianVelocityJogConfig,
    CartesianVelocityJogController,
)
from assist.logger import TickLoggingConfig
from assist.objects import ObjectsTracker

from motion_generation.config import RunConfig, EpisodeConfig, TaskConfig, PlannerConfig, ObjectsConfig, LoggingConfig, GraspConfig


def run_motion_planner_demo(args: argparse.Namespace) -> int:
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import Isaac/scene modules that require an active Omniverse app
    import isaaclab.sim as sim_utils
    from environments.reach_to_grasp.utils import design_scene
    from environments.reach_to_grasp.config import DEFAULT_SCENE, DEFAULT_CAMERA
    from environments.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix
    from motion_generation.engine.episode_runner import EpisodeRunner

    # Setup simulation
    phys = PhysicsConfig(device=args.device)
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    if not args.headless:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    print("[MG] App launched; building scene...")
    # Build scene
    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    # Spawn objects (default to Nucleus YCB if not provided)
    id_to_label: Dict[str, str] = {}
    prim_paths = []
    dataset_dirs = [str(d) for d in getattr(args, "objects_dataset", [])]
    if len(dataset_dirs) == 0:
        try:
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # type: ignore
            ycb_dir = f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
        except Exception:
            ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"
        dataset_dirs = [ycb_dir]
        print(f"[MG] Using default YCB dataset: {dataset_dirs[0]}")
    else:
        print(f"[MG] Using custom object datasets: {dataset_dirs}")
    if (not getattr(args, "no_objects", False)) and int(args.num_objects) > 0:
        scale_range = None
        if getattr(args, "scale_min", None) is not None and getattr(args, "scale_max", None) is not None:
            scale_range = (float(args.scale_min), float(args.scale_max))
            print(f"[MG] Uniform scale range: {scale_range}")
        phys_loader_kwargs = object_loader_kwargs_from_physix(phys)
        loader_cfg = ObjectLoaderConfig(
            dataset_dirs=dataset_dirs,
            bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
            min_distance=float(getattr(args, "min_distance", 0.1)),
            uniform_scale_range=scale_range,
            **phys_loader_kwargs,
        )
        print(f"[MG] Spawning {int(args.num_objects)} objects in AABB min={tuple(args.spawn_min)} max={tuple(args.spawn_max)}")
        loader = ObjectLoader(loader_cfg)
        try:
            prim_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(args.num_objects))
        except Exception as e:
            print(f"[MG][WARN] Object spawn failed: {e}")
            prim_paths = []
        try:
            prim_to_label = loader.get_last_spawn_labels()
            id_to_label = {str(p).split("/")[-1]: str(lbl) for p, lbl in prim_to_label.items()}
            print(f"[MG] Spawned objects: {id_to_label}")
        except Exception:
            id_to_label = {}
            print("[MG][WARN] Could not build id->label map; labels may default to 'object'.")

    # Reset sim and robot
    sim.reset()
    origin0 = torch.tensor(scene_origins[0], device=sim.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    # Controller
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(getattr(args, "speed", 0.7)),
        workspace_min=(0.20, -0.45, 0.01),
        workspace_max=(0.6, 0.45, 0.35),
        log_ee_pos=False,
        log_ee_frame="world",
        log_every_n_steps=9999,
    )
    print(f"[MG] Controller speed={ctrl_cfg.linear_speed_mps} m/s, ee_link={ctrl_cfg.ee_link_name}")
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    controller.set_mode("translate")

    # Tick config (no logging writer for planner demo)
    tick_cfg = TickLoggingConfig(
        log_rate_hz=10,
        workspace_min=getattr(controller.config.safety_cfg, 'workspace_min', None),
        workspace_max=getattr(controller.config.safety_cfg, 'workspace_max', None),
        ee_link_name=controller.config.ee_link_name,
        arm_joint_regex=controller.config.arm_joint_regex,
    )
    session_logger = None

    # Tracker
    tracker = ObjectsTracker(prim_paths=prim_paths)

    # Run config
    # Default Replicator YAML if not provided and grasp provider is enabled
    default_rep_yaml = str((Path(__file__).resolve().parent / "motion_generation_config" / "gripper_configs" / "j2n6s300_topdown.yaml").resolve())

    run_cfg = RunConfig(
        episode=EpisodeConfig(num_episodes=int(args.num_episodes)),
        task=TaskConfig(target_label=getattr(args, "target_label", None), pregrasp_offset_m=float(getattr(args, "pregrasp", 0.10)), lift_height_m=float(getattr(args, "lift", 0.15))),
        planner=PlannerConfig(type=str(getattr(args, "planner", "scripted")), linear_speed_mps=float(getattr(args, "speed", 0.7)), tolerance_m=float(getattr(args, "tolerance", 0.005))),
        grasp=GraspConfig(
            type=str(getattr(args, "grasp", "aabb")),
            rep_gripper_prim_path=getattr(args, "rep_gripper_prim_path", None),
            rep_config_yaml_path=getattr(args, "rep_config_yaml", default_rep_yaml if str(getattr(args, "grasp", "aabb")) == "replicator" else None),
        ),
        objects=ObjectsConfig(dataset_dirs=dataset_dirs, num_objects=int(args.num_objects), spawn_min_xyz=tuple(args.spawn_min), spawn_max_xyz=tuple(args.spawn_max)),
        logging=LoggingConfig(logs_root=str(getattr(args, "logs_root", "logs/assist"))),
    )

    print("[MG] Planner demo initialized. Starting episodes...")
    runner = EpisodeRunner(sim=sim, robot=robot, controller=controller, session_logger=session_logger, tick_cfg=tick_cfg, tracker=tracker, id_to_label=id_to_label, run_cfg=run_cfg)

    # Episodes
    successes = 0
    for ep in range(int(args.num_episodes)):
        outcome = runner.run_episode(target_label=args.target_label)
        successes += int(outcome.success)

    print(f"[MG] Completed: episodes={args.num_episodes} success={successes}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Motion generation planner demo")
    # Reuse the stable demo CLI for spawn/object/controller settings
    from scripts.cli import add_demo_cli_args
    add_demo_cli_args(ap)
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--target-label", type=str, default=None)
    ap.add_argument("--objects-dataset", type=str, nargs="*", default=[])
    ap.add_argument("--pregrasp", type=float, default=0.10)
    ap.add_argument("--lift", type=float, default=0.15)
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--planner", type=str, default="scripted", choices=["scripted", "rmpflow", "curobo"])
    AppLauncher.add_app_launcher_args(ap)
    args_cli = ap.parse_args()
    raise SystemExit(run_motion_planner_demo(args_cli))


