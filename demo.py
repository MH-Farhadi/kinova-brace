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
    from scripts.cli import add_demo_cli_args
    add_demo_cli_args(parser)
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
        from environments.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix
    except Exception:
        from ..environments.reach_to_grasp.config import DEFAULT_SCENE, DEFAULT_CAMERA  # type: ignore
        from ..environments.reach_to_grasp.utils import design_scene  # type: ignore
        from ..environments.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds  # type: ignore
        from ..environments.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix  # type: ignore

    # Load physics config (for now use defaults; you can import a custom one here)
    phys = PhysicsConfig(device=args_cli.device)
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    if not args_cli.headless:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    # Build scene and fetch robot
    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    # Optionally spawn random USD objects from Nucleus YCB under /World/Origin1
    if not args_cli.no_objects:
        scale_range = None
        if args_cli.scale_min is not None and args_cli.scale_max is not None:
            scale_range = (float(args_cli.scale_min), float(args_cli.scale_max))

        # Resolve default dataset dir to YCB on Nucleus (fallback to public S3 if needed)
        try:
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # type: ignore
            ycb_dir = f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
        except Exception:
            ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"

        # Translate physics defaults to loader kwargs
        phys_loader_kwargs = object_loader_kwargs_from_physix(phys)

        loader_cfg = ObjectLoaderConfig(
            dataset_dirs=[ycb_dir],
            bounds=SpawnBounds(min_xyz=tuple(args_cli.spawn_min), max_xyz=tuple(args_cli.spawn_max)),
            min_distance=float(args_cli.min_distance),
            uniform_scale_range=scale_range,
            **phys_loader_kwargs,
        )
        loader = ObjectLoader(loader_cfg)
        spawned_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(args_cli.num_objects))
        if len(spawned_paths) > 0:
            print(f"[INFO] Spawned {len(spawned_paths)} object(s) under /World/Origin1 from Nucleus")

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


