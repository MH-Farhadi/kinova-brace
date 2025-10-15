from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from isaaclab.app import AppLauncher
from controllers import ModeManager

# Ensure project root on sys.path for modular imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import from our modular controller library
from controllers import (
    CartesianVelocityJogConfig,
    CartesianVelocityJogController,
    Se3KeyboardInput,
    ModeManager,
)


def run(sim, robot, controller: CartesianVelocityJogController, simulation_app):
    dt = sim.get_physics_dt()
    controller.reset(robot)
    print("[INFO] Cartesian velocity jog running. In headless mode, commands default to zero.")
    while simulation_app.is_running():
        controller.step(robot, dt)
        sim.step()
        robot.update(dt)


def main():
    parser = argparse.ArgumentParser(description="Cartesian velocity jog demo using modular controller.")
    parser.add_argument("--ee_link", type=str, default="j2n6s300_end_effector", help="End-effector link name")
    parser.add_argument("--speed", type=float, default=0.20, help="Linear speed (m/s)")
    parser.add_argument("--rot-speed", type=float, default=0.5, help="Angular speed (rad/s)")
    # Dataset object spawning
    parser.add_argument(
        "--dataset-dirs",
        type=str,
        nargs="*",
        default=[],
        help="One or more dataset directories containing .usd files (defaults to kinova-isaac/datasets/*)",
    )
    parser.add_argument("--num-objects", type=int, default=5, help="Number of random objects to spawn")
    parser.add_argument(
        "--spawn-min",
        type=float,
        nargs=3,
        default=[-0.30, -0.45, 0.90],
        help="Spawn AABB min xyz (meters) relative to /World/Origin1",
    )
    parser.add_argument(
        "--spawn-max",
        type=float,
        nargs=3,
        default=[0.30, 0.45, 1.10],
        help="Spawn AABB max xyz (meters) relative to /World/Origin1",
    )
    parser.add_argument("--min-distance", type=float, default=0.08, help="Min distance between spawned objects (m)")
    parser.add_argument("--scale-min", type=float, default=None, help="Optional uniform scale min for objects")
    parser.add_argument("--scale-max", type=float, default=None, help="Optional uniform scale max for objects")
    # Physics/placement controls for spawned objects
    parser.add_argument("--snap-z-to", type=float, default=None, help="Snap spawn Z to this height (m)")
    parser.add_argument("--z-clearance", type=float, default=0.02, help="Extra Z clearance above snap plane (m)")
    parser.add_argument("--density", type=float, default=None, help="Set object density (kg/m^3). If unset, default 500.")
    parser.add_argument("--mass-kg", type=float, default=None, help="Set object mass (kg). Overrides density if set.")
    parser.add_argument("--use-nucleus-ycb", action="store_true", help="Use Nucleus YCB assets instead of local datasets")
    parser.add_argument("--apply-preview-surface", action="store_true", help="Apply simple preview surface to spawned objects")
    parser.add_argument(
        "--preview-surface-diffuse",
        type=float,
        nargs=3,
        default=[0.7, 0.7, 0.7],
        help="RGB color for preview surface when applied",
    )
    parser.add_argument("--no-objects", action="store_true", help="Skip spawning random objects from datasets")
    # EE logging controls
    parser.add_argument("--print-ee", action="store_true", help="Print EE XYZ each step")
    parser.add_argument("--ee-frame", type=str, default="world", choices=["world", "base"], help="Frame for EE logging")
    parser.add_argument("--print-interval", type=int, default=1, help="Print every N steps")
    # Mode handling
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import modules that require an active Omniverse app after AppLauncher
    import isaaclab.sim as sim_utils
    try:
        from environments.reach_to_grasp.config import DEFAULT_SCENE, DEFAULT_CAMERA
        from environments.reach_to_grasp.utils import design_scene
        from environments.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    except Exception:
        from ..environments.reach_to_grasp.config import DEFAULT_SCENE, DEFAULT_CAMERA  # type: ignore
        from ..environments.reach_to_grasp.utils import design_scene  # type: ignore
        from ..environments.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds  # type: ignore

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    if not args_cli.headless:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    # Build scene and fetch robot
    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    # Optionally spawn random USD objects from dataset(s) under /World/Origin1
    if not args_cli.no_objects:
        if args_cli.use_nucleus_ycb:
            # Prefer Nucleus YCB Props folder (works if Nucleus is connected)
            # Use resolved assets root from IsaacLab utilities (Omniverse path)
            try:
                from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # type: ignore
            except Exception:
                # If import fails for any reason, fallback to common default (may not work)
                ISAAC_NUCLEUS_DIR = "/Isaac"
            dataset_dirs = [f"{ISAAC_NUCLEUS_DIR}/Props/YCB"]
        else:
            default_dataset_dirs = [
                str(ROOT / "kinova-isaac" / "datasets" / "MUGS"),
                str(ROOT / "kinova-isaac" / "datasets" / "AXIS_ALIGNED-PHYSICS"),
                str(ROOT / "kinova-isaac" / "datasets" / "AXIS_AGLINED"),
            ]
            dataset_dirs = args_cli.dataset_dirs if len(args_cli.dataset_dirs) > 0 else default_dataset_dirs
        scale_range = None
        if args_cli.scale_min is not None and args_cli.scale_max is not None:
            scale_range = (float(args_cli.scale_min), float(args_cli.scale_max))

        # Determine density value while preserving default if not specified
        density_val = float(args_cli.density) if args_cli.density is not None else 500.0

        loader_cfg = ObjectLoaderConfig(
            dataset_dirs=list(dataset_dirs),
            bounds=SpawnBounds(min_xyz=tuple(args_cli.spawn_min), max_xyz=tuple(args_cli.spawn_max)),
            min_distance=float(args_cli.min_distance),
            uniform_scale_range=scale_range,
            apply_preview_surface=bool(args_cli.apply_preview_surface),
            preview_surface_diffuse=tuple(args_cli.preview_surface_diffuse),
            snap_z_to=float(args_cli.snap_z_to) if args_cli.snap_z_to is not None else None,
            z_clearance=float(args_cli.z_clearance),
            mass_kg=float(args_cli.mass_kg) if args_cli.mass_kg is not None else None,
            density=density_val,
        )
        loader = ObjectLoader(loader_cfg)
        spawned = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(args_cli.num_objects))
        if len(spawned) > 0:
            print(f"[INFO] Spawned {len(spawned)} object(s) under /World/Origin1 from datasets: {len(dataset_dirs)} dirs")

    # Reset sim and robot state at origin 0
    sim.reset()
    origin0 = torch.tensor(scene_origins[0], device=sim.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()

    # Controller setup
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(args_cli.ee_link),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(args_cli.speed),
        # Per-axis workspace bounds in base frame (meters)
        workspace_min=(0.20, -0.45, 0.01),
        workspace_max=(0.6, 0.45, 0.35),
        log_ee_pos=bool(args_cli.print_ee),
        log_ee_frame=str(args_cli.ee_frame),
        log_every_n_steps=int(args_cli.print_interval),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    
    # Mode management setup
    mode_manager = ModeManager(initial_mode="translate")
    mode_manager.set_mode_change_callback(lambda mode: controller.set_mode(mode.value))
    controller.set_mode("translate")  # Set initial mode

    if not args_cli.headless:
        keyboard = Se3KeyboardInput(
            pos_sensitivity_per_step=ctrl_cfg.linear_speed_mps * sim.get_physics_dt(),
            rot_sensitivity_rad_per_step=float(args_cli.rot_speed) * sim.get_physics_dt(),
        )
        controller.set_input_provider(keyboard)

        # Setup mode switching callbacks
        translate_fn, rotate_fn, gripper_fn = mode_manager.get_mode_callbacks()
        keyboard.add_mode_callbacks(translate_fn, rotate_fn, gripper_fn)

    print("[INFO]: Setup complete... (Mode keys: F/f/1=translate, R/r/2=rotate, G/g/3=gripper). Start mode= translate")
    run(sim, robot, controller, simulation_app)
    simulation_app.close()


if __name__ == "__main__":
    main()


