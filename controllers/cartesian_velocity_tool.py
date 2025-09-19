from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional, Literal

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

from controllers.base import ArmController, ArmControllerConfig, InputProvider


@dataclass
class CartesianVelocityJogConfig(ArmControllerConfig):
    """Config for Cartesian velocity jogging using Differential IK."""

    linear_speed_mps: float = 0.05
    ik_method: Literal["pinv", "svd", "trans", "dls"] = "dls"


class CartesianVelocityJogController(ArmController):
    """Controller that maps a 3D Cartesian velocity command to joint velocity via Diff-IK.

    The input provider is expected to return a 7D tensor per step: [dx, dy, dz, rx, ry, rz, g]
    Only the first three entries are used to construct a translation-only 6D twist.
    """

    def __init__(self, config: CartesianVelocityJogConfig, num_envs: int = 1, device: Optional[str] = None) -> None:
        super().__init__(config, num_envs=num_envs, device=device)
        self.config: CartesianVelocityJogConfig
        self._diff_ik: Optional[DifferentialIKController] = None
        self._arm_joint_ids = None
        self._ee_body_id = None
        self._ee_jacobi_idx = None

    def reset(self, robot) -> None:
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=self.config.use_relative_mode,
            ik_method=self.config.ik_method,
        )
        self._diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self._diff_ik.reset()

        # Resolve indices for EE and arm joints
        body_ids, _ = robot.find_bodies([self.config.ee_link_name])
        self._ee_body_id = int(body_ids[0])
        self._ee_jacobi_idx = self._ee_body_id - 1 if robot.is_fixed_base else self._ee_body_id
        arm_joint_ids, _ = robot.find_joints(self.config.arm_joint_regex)
        self._arm_joint_ids = arm_joint_ids

        # Hold current joint positions as position targets
        robot.set_joint_position_target(robot.data.joint_pos)

    def step(self, robot, dt: float) -> None:
        assert self._diff_ik is not None, "Controller not reset()"
        assert self._arm_joint_ids is not None, "Controller not reset()"
        assert self._ee_body_id is not None, "Controller not reset()"

        # Read input provider
        if self._input_provider is None:
            dx_cmd = torch.zeros(self.num_envs, 6, device=self.device)
        else:
            cmd7 = self._input_provider.advance()
            dx_cmd = torch.zeros(self.num_envs, 6, device=self.device)
            dx_cmd[:, 0:3] = cmd7[..., 0:3]

        # State
        jac = robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._arm_joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, self._ee_body_id]
        root_pose_w = robot.data.root_pose_w
        q_arm = robot.data.joint_pos[:, self._arm_joint_ids]

        # Relative pose frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Compute desired joint positions via IK and convert to velocity
        self._diff_ik.set_command(dx_cmd, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        q_des = self._diff_ik.compute(ee_pos_b, ee_quat_b, jac, q_arm)
        qdot_arm = (q_des - q_arm) / dt

        # Hold all joints at current position target, zero velocity for non-arm joints
        robot.set_joint_position_target(robot.data.joint_pos)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        robot.set_joint_velocity_target(qdot_arm, joint_ids=self._arm_joint_ids)

        # Gravity compensation
        gravity = robot.root_physx_view.get_gravity_compensation_forces()
        robot.set_joint_effort_target(gravity)

        # Write to sim
        robot.write_data_to_sim()


