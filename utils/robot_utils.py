"""Robot control and state utility functions."""

from __future__ import annotations

from typing import Tuple

import torch


def reset_robot_to_origin(sim, robot, origin_xyz: Tuple[float, float, float]) -> None:
    """Reset robot to specified origin position.
    
    Args:
        sim: Simulation context
        robot: Robot articulation
        origin_xyz: Target origin position (x, y, z)
    """
    origin0 = torch.tensor(origin_xyz, device=sim.device)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()


def get_ee_pos_base_frame(robot, ee_link_name: str) -> torch.Tensor:
    """Get end-effector position in robot base frame.
    
    Args:
        robot: Robot articulation
        ee_link_name: Name of the end-effector link
        
    Returns:
        End-effector position in base frame (3,)
    """
    from isaaclab.utils.math import subtract_frame_transforms

    body_ids, _ = robot.find_bodies([ee_link_name])
    ee_id = int(body_ids[0])
    ee_pose_w = robot.data.body_pose_w[:, ee_id]       # (1,7)
    root_pose_w = robot.data.root_pose_w               # (1,7)
    pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    return pos_b[0]  # (3,)


def stabilize_with_hold(sim, robot, steps: int, dt: float) -> None:
    """Step simulation while actively holding robot at current position.
    
    This ensures the robot maintains its position during object stabilization
    by applying gravity compensation and position holding commands.
    
    Args:
        sim: Simulation context
        robot: Robot articulation
        steps: Number of simulation steps to hold
        dt: Physics timestep in seconds
    """
    if steps <= 0:
        return
    
    print(f"[MG] Stabilizing for {steps} steps while holding robot position...")
    
    # Store the current joint positions as targets
    target_joint_pos = robot.data.joint_pos.clone()
    
    for _ in range(steps):
        # Hold all joints at their current target positions
        robot.set_joint_position_target(target_joint_pos)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        
        # Apply gravity compensation to prevent sagging
        gravity = robot.root_physx_view.get_gravity_compensation_forces()
        robot.set_joint_effort_target(gravity)
        
        # Write commands and step simulation
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

