from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

# Ensure project root on sys.path for modular imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from controllers.cartesian_velocity import CartesianVelocityJogController, CartesianVelocityJogConfig

from controllers.base import InputProvider


class Se3KeyboardInput(InputProvider):
    def __init__(self, pos_sensitivity_per_step: float, rot_sensitivity_rad_per_step: float) -> None:
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

        self._kb = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=pos_sensitivity_per_step,
                rot_sensitivity=rot_sensitivity_rad_per_step,
                gripper_term=False,
            )
        )

    def reset(self) -> None:
        pass

    def advance(self):
        return self._kb.advance()


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
    # EE logging controls
    parser.add_argument("--print-ee", action="store_true", help="Print EE XYZ each step")
    parser.add_argument("--ee-frame", type=str, default="world", choices=["world", "base"], help="Frame for EE logging")
    parser.add_argument("--print-interval", type=int, default=1, help="Print every N steps")
    # Mode handling
    parser.add_argument("--start-mode", type=str, default="translate", choices=["translate", "rotate"], help="Initial mode")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import modules that require an active Omniverse app after AppLauncher
    import isaaclab.sim as sim_utils
    try:
        from environments.reach_to_grasp.config import DEFAULT_SCENE, DEFAULT_CAMERA
        from environments.reach_to_grasp.utils import design_scene
    except Exception:
        from ..environments.reach_to_grasp.config import DEFAULT_SCENE, DEFAULT_CAMERA  # type: ignore
        from ..environments.reach_to_grasp.utils import design_scene  # type: ignore

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    if not args_cli.headless:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    # Build scene and fetch robot
    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

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
    controller.set_mode(str(args_cli.start_mode))

    if not args_cli.headless:
        keyboard = Se3KeyboardInput(
            pos_sensitivity_per_step=ctrl_cfg.linear_speed_mps * sim.get_physics_dt(),
            rot_sensitivity_rad_per_step=float(args_cli.rot_speed) * sim.get_physics_dt(),
        )
        controller.set_input_provider(keyboard)

        # Mode switching via Se3Keyboard callbacks
        def _to_translate():
            print("[KB] translate mode request")
            controller.set_mode("translate")

        def _to_rotate():
            print("[KB] rotate mode request")
            controller.set_mode("rotate")

        try:
            for k in ["f", "F", "1"]:
                keyboard._kb.add_callback(k, _to_translate)  # type: ignore[attr-defined]
            for k in ["r", "R", "2"]:
                keyboard._kb.add_callback(k, _to_rotate)     # type: ignore[attr-defined]
        except Exception:
            print("[INFO]: Failed to add keyboard callbacks")

    print("[INFO]: Setup complete... (Mode keys: F/f=translate, R/r=rotate; also 1/2). Start mode=", args_cli.start_mode)
    run(sim, robot, controller, simulation_app)
    simulation_app.close()


if __name__ == "__main__":
    main()


