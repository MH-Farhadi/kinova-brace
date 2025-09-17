import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Minimal Cartesian velocity jog with Differential IK (JACO2)")
parser.add_argument("--ee_link", type=str, default="j2n6s300_end_effector", help="End-effector link name")
parser.add_argument("--speed", type=float, default=0.05, help="Cartesian speed in m/s")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from dataclasses import replace as dc_replace

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


EE_LINK = args_cli.ee_link
LINEAR_SPEED = float(args_cli.speed)


def _spawn_scene() -> Articulation:
    # Ground-plane and light
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/defaultGroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)
    # Table
    table = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
        scale=(1.5, 2.0, 1.0),
    )
    table.func("/World/Table", table, translation=(0.0, 0.0, 0.8))
    # Robot: Kinova JACO2 6-DoF
    from isaaclab_assets import KINOVA_JACO2_N6S300_CFG

    robot_cfg = dc_replace(KINOVA_JACO2_N6S300_CFG, prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.8)

    robot = Articulation(cfg=robot_cfg)
    return robot


def _make_keyboard_interface(speed: float):
    # Use existing SE3 keyboard but ignore rotations; we only care about pos deltas via W/S A/D Q/E
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

    # map keyboard to a fixed velocity command per step
    kb = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=speed, rot_sensitivity=0.0, gripper_term=False))
    return kb


def main():
    # Create simulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.5, 0.0, 3.2), (0.0, 0.0, 0.5))

    # Spawn scene and robot
    robot = _spawn_scene()

    # Controller: differential IK in relative pose mode (we'll feed 6D twist-like delta)
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=1, device=sim.device)

    # Keyboard interface for W/S A/D Q/E
    dt = sim.get_physics_dt()
    keyboard = _make_keyboard_interface(speed=LINEAR_SPEED * dt)

    # Reset sim and robot (initializes PhysX views)
    sim.reset()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    # Snapshot a fixed hold position for all joints
    q_hold_all = joint_pos.clone()

    # Hold current joint positions as position targets so implicit actuators don't pull to zero
    robot.set_joint_position_target(q_hold_all)

    diff_ik.reset()

    # Resolve EE link index and joint ids directly from articulation (after reset)
    joint_ids, _ = robot.find_joints(".*")
    body_ids, _ = robot.find_bodies([EE_LINK])
    ee_body_id = body_ids[0]
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    # Separate arm joints from gripper joints for control
    arm_joint_ids, _ = robot.find_joints("j2n6s300_joint_[1-6]")

    print("\n[INFO] Cartesian velocity jog ready. Hold W/S A/D Q/E. Press L to zero command.")

    # Loop
    while simulation_app.is_running():
        # Read keyboard → delta pose per step; interpret first 3 as translation deltas
        cmd7 = keyboard.advance()  # [dx, dy, dz, rx, ry, rz, (ignored)]
        # Build pure-translation twist-like delta
        dx = torch.zeros(1, 6, device=sim.device)
        dx[0, 0:3] = cmd7[0:3]

        # Get current state (only for arm joints)
        jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        q_arm = robot.data.joint_pos[:, arm_joint_ids]

        # Compute ee in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Set desired pose increment: relative command around current pose
        # diff_ik expects set_command() with 6D delta when pose_rel is used
        diff_ik.set_command(dx, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        # Compute next joint positions, then derive velocity as q_des - q
        q_des = diff_ik.compute(ee_pos_b, ee_quat_b, jac, q_arm)
        qdot_arm = (q_des - q_arm) / dt

        # CRITICAL: Set position targets for ALL joints to current positions to hold against gravity
        # This neutralizes the P-term for all joints
        robot.set_joint_position_target(robot.data.joint_pos)
        
        # Set velocity targets: commanded velocities for arm, zero for others
        # First set all joint velocities to zero
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        # Then override with computed velocities for arm joints only
        robot.set_joint_velocity_target(qdot_arm, joint_ids=arm_joint_ids)
        
        # Apply gravity compensation
        gravity = robot.root_physx_view.get_gravity_compensation_forces()
        robot.set_joint_effort_target(gravity)

        # Write all targets to sim
        robot.write_data_to_sim()
        # Print current commanded linear velocity (m/s)
        vx, vy, vz = float(cmd7[0] / dt), float(cmd7[1] / dt), float(cmd7[2] / dt)
        # print(f"cmd [vx vy vz]=[{vx:+.3f} {vy:+.3f} {vz:+.3f}]")

        # Step
        sim.step()
        robot.update(dt)


if __name__ == "__main__":
    main()
    simulation_app.close()


