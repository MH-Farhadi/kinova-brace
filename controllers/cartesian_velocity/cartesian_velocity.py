"""Cartesian velocity jogging controller implementation."""

from __future__ import annotations

import torch
from typing import Optional, Literal

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_box_plus

from ..base import ArmControllerConfig, ArmController, InputProvider
from ..safety import WorkspaceBounds, hold_orientation
from dataclasses import dataclass

@dataclass
class CartesianVelocityJogConfig(ArmControllerConfig):
    """Config for Cartesian velocity jogging using Differential IK and gripper control."""

    linear_speed_mps: float = 0.05
    ik_method: Literal["pinv", "svd", "trans", "dls"] = "dls"
    hold_orientation: bool = True
    # Gripper settings
    gripper_joint_regex: str = ".*_joint_finger_.*|.*_joint_finger_tip_.*"
    gripper_open_pos: float = 0.2
    gripper_close_pos: float = 1.2 


class CartesianVelocityJogController(ArmController):
    """Controller mapping Cartesian deltas to joint velocity via Diff-IK, with a gripper mode.

    Input provider should return up to 7D per step: [dx, dy, dz, rx, ry, rz, g]
    - Translation mode: uses [dx, dy, dz], holds orientation.
    - Rotation mode: uses [rx, ry, rz] (rotation vector), holds position.
    - Gripper mode: uses [g] if present (g>0 open, g<0 close) and holds pose.
    """

    def __init__(self, config: CartesianVelocityJogConfig, num_envs: int = 1, device: Optional[str] = None) -> None:
        super().__init__(config, num_envs=num_envs, device=device)
        self.config: CartesianVelocityJogConfig
        self._diff_ik: Optional[DifferentialIKController] = None
        self._arm_joint_ids = None
        self._gripper_joint_ids = None
        self._ee_body_id = None
        self._ee_jacobi_idx = None
        self._ee_quat_hold_b: Optional[torch.Tensor] = None
        self._step_count = 0
        self._mode: Literal["translate", "rotate", "gripper"] = "translate"

    def set_mode(self, mode: Literal["translate", "rotate", "gripper"]) -> None:
        if mode not in ("translate", "rotate", "gripper"):
            raise ValueError("mode must be 'translate', 'rotate', or 'gripper'")
        if self._mode != mode:
            self._mode = mode
            print(f"[CTRL] Mode set to: {self._mode}")

    def reset(self, robot) -> None:
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=self.config.use_relative_mode,
            ik_method=self.config.ik_method,
        )
        self._diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self._diff_ik.reset()

        # Resolve indices for EE, arm joints, and gripper joints
        body_ids, _ = robot.find_bodies([self.config.ee_link_name])
        self._ee_body_id = int(body_ids[0])
        self._ee_jacobi_idx = self._ee_body_id - 1 if robot.is_fixed_base else self._ee_body_id
        arm_joint_ids, _ = robot.find_joints(self.config.arm_joint_regex)
        self._arm_joint_ids = arm_joint_ids
        gripper_joint_ids, _ = robot.find_joints(self.config.gripper_joint_regex)
        self._gripper_joint_ids = gripper_joint_ids

        # Hold current joint positions as position targets
        robot.set_joint_position_target(robot.data.joint_pos)

        # Capture initial EE orientation in base frame to optionally hold during translation
        root_pose_w = robot.data.root_pose_w
        ee_pose_w = robot.data.body_pose_w[:, self._ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        self._ee_quat_hold_b = ee_quat_b.clone()

    def step(self, robot, dt: float) -> None:
        assert self._diff_ik is not None, "Controller not reset()"
        assert self._arm_joint_ids is not None, "Controller not reset()"
        assert self._ee_body_id is not None, "Controller not reset()"

        # Read input provider and robustly shape it to (N, D)
        if self._input_provider is None:
            cmd = torch.zeros(1, 6, device=self.device)
        else:
            cmd = self._input_provider.advance()
            if cmd.ndim == 1:
                cmd = cmd.view(1, -1)
        if str(cmd.device) != str(torch.device(self.device)):
            cmd = cmd.to(self.device)
        # ensure at least 6, keep extra (g) if present
        if cmd.shape[-1] < 6:
            pad = torch.zeros(cmd.shape[0], 6 - cmd.shape[-1], device=cmd.device, dtype=cmd.dtype)
            cmd = torch.cat([cmd, pad], dim=-1)

        dpos = cmd[..., 0:3]
        drot = cmd[..., 3:6]
        g_val: Optional[torch.Tensor] = None
        if cmd.shape[-1] >= 7:
            g_val = cmd[..., 6]

        # State
        jac = robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._arm_joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, self._ee_body_id]
        root_pose_w = robot.data.root_pose_w
        q_arm = robot.data.joint_pos[:, self._arm_joint_ids]

        # Relative pose frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Build desired pose based on mode
        ws = WorkspaceBounds(self.config.workspace_min, self.config.workspace_max)
        if self._mode == "translate":
            pos_des = ws.clamp(ee_pos_b + dpos, device=self.device)
            quat_des = hold_orientation(ee_quat_b, self._ee_quat_hold_b, True)
        elif self._mode == "rotate":
            pos_des = ws.clamp(ee_pos_b, device=self.device)
            quat_des = quat_box_plus(ee_quat_b, drot)
        else:  # gripper
            pos_des = ws.clamp(ee_pos_b, device=self.device)
            quat_des = ee_quat_b

        # Feed IK
        self._diff_ik.ee_pos_des[:] = pos_des
        self._diff_ik.ee_quat_des[:] = quat_des
        q_des = self._diff_ik.compute(ee_pos_b, ee_quat_b, jac, q_arm)
        qdot_arm = (q_des - q_arm) / dt

        # Hold all joints at current position target, zero velocity for non-arm joints
        robot.set_joint_position_target(robot.data.joint_pos)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        robot.set_joint_velocity_target(qdot_arm, joint_ids=self._arm_joint_ids)

        # Gripper control in gripper mode using g_val
        if self._gripper_joint_ids:
            try:
                num_gripper_joints = len(self._gripper_joint_ids)  # list-like
            except TypeError:
                num_gripper_joints = int(self._gripper_joint_ids.shape[0])  # tensor-like
            if self._mode == "gripper" and g_val is not None and num_gripper_joints > 0:
                open_cmd = (g_val.mean().item() > 0.0)
                target_val = self.config.gripper_open_pos if open_cmd else self.config.gripper_close_pos
                target = torch.full(
                    (robot.data.joint_pos.shape[0], num_gripper_joints),
                    fill_value=target_val,
                    dtype=robot.data.joint_pos.dtype,
                    device=self.device,
                )
                robot.set_joint_position_target(target, joint_ids=self._gripper_joint_ids)

        # Gravity compensation
        gravity = robot.root_physx_view.get_gravity_compensation_forces()
        robot.set_joint_effort_target(gravity)

        # Optional real-time EE position logging
        if getattr(self.config, "log_ee_pos", False):
            frame = getattr(self.config, "log_ee_frame", "world")
            every = max(1, int(getattr(self.config, "log_every_n_steps", 1)))
            if (self._step_count % every) == 0:
                if frame == "world":
                    pos = ee_pose_w[:, 0:3]
                else:
                    pos = ee_pos_b
                pos_np = pos.detach().to("cpu").numpy().tolist()
                print(f"[EE] frame={frame} xyz(m)={pos_np} mode={self._mode}")
        self._step_count += 1

        # Write to sim
        robot.write_data_to_sim() 