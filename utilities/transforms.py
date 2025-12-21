"""Coordinate transformation utility functions."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def world_to_base_pos(sim, robot, pos_w: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert world position to robot base frame.
    
    Args:
        sim: Simulation context
        robot: Robot articulation
        pos_w: Position in world frame (x, y, z)
        
    Returns:
        Position in base frame (x, y, z)
    """
    from isaaclab.utils.math import quat_conjugate, quat_apply

    root_pose_w = robot.data.root_pose_w[0]
    base_pos_w = root_pose_w[0:3]
    base_quat_w = root_pose_w[3:7]
    base_quat_inv = quat_conjugate(base_quat_w)
    rel_w = torch.tensor(pos_w, dtype=torch.float32, device=sim.device) - base_pos_w
    pos_b = quat_apply(base_quat_inv, rel_w)
    return (float(pos_b[0]), float(pos_b[1]), float(pos_b[2]))


def world_to_base_quat(
    sim, 
    robot, 
    quat_wxyz_w: Optional[Tuple[float, float, float, float]]
) -> Optional[Tuple[float, float, float, float]]:
    """Convert world quaternion to robot base frame.
    
    Args:
        sim: Simulation context
        robot: Robot articulation
        quat_wxyz_w: Quaternion in world frame (w, x, y, z), or None
        
    Returns:
        Quaternion in base frame (w, x, y, z), or None if input is None
    """
    if quat_wxyz_w is None:
        return None
    try:
        # IsaacLab math utils operate on XYZW; our data is WXYZ. Convert, multiply, convert back.
        from isaaclab.utils.math import quat_conjugate, quat_multiply, wxyz2xyzw, xyzw2wxyz  # type: ignore[attr-defined]
        base_quat_wxyz = robot.data.root_pose_w[0, 3:7]  # (w,x,y,z)
        base_quat_xyzw = wxyz2xyzw(base_quat_wxyz)
        base_quat_inv_xyzw = quat_conjugate(base_quat_xyzw)
        q_wxyz = torch.tensor(quat_wxyz_w, dtype=torch.float32, device=sim.device)
        q_xyzw = wxyz2xyzw(q_wxyz)
        qb_xyzw = quat_multiply(base_quat_inv_xyzw, q_xyzw)
        qb_wxyz = xyzw2wxyz(qb_xyzw)
        return (float(qb_wxyz[0]), float(qb_wxyz[1]), float(qb_wxyz[2]), float(qb_wxyz[3]))
    except Exception:
        return None


def yaw_from_quat_wxyz(q: Tuple[float, float, float, float]) -> float:
    """Extract yaw (rotation around Z axis) from quaternion.
    
    Args:
        q: Quaternion in (w, x, y, z) format
        
    Returns:
        Yaw angle in radians
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    s1 = 2.0 * (w * z + x * y)
    c1 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s1, c1)

