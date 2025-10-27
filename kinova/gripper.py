from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class GripperConfig:
    """Config for Kinova gripper mapping and dynamics.

    - joint_regex: regex(es) used to find gripper joints on the robot
    - open_position: joint position value representing open state
    - close_position: joint position value representing closed state
    - stiffness: drive stiffness to apply on gripper joints
    - damping: drive damping to apply on gripper joints
    - max_velocity: optional joint velocity limit for gripper drives
    - effort_limit: optional effort/force limit for gripper drives
    - split_base_and_tip: whether to separate base and tip finger joints
    - tip_ratio_on_close: fraction of base close value to apply on tips when closing
    """

    joint_regex: str = ".*_joint_finger_.*|.*_joint_finger_tip_.*"
    open_position: float = 0.0
    close_position: float = 1.2
    stiffness: float = 1.2
    damping: float = 0.01
    max_velocity: Optional[float] = None
    effort_limit: Optional[float] = None
    split_base_and_tip: bool = True
    tip_ratio_on_close: float = 0.35


class GripperController:
    """High-level gripper utilities bound to a robot articulation view.

    Expected robot interface (isaaclab.assets.Articulation):
    - find_joints(regex) -> (indices, names)
    - data.joint_pos, data.joint_vel
    - set_joint_position_target(tensor, joint_ids=...)
    - set_joint_velocity_target(tensor)
    - set_joint_effort_target(tensor)
    - root_physx_view.get_gravity_compensation_forces()
    """

    def __init__(self, cfg: GripperConfig, num_envs: int, device: str) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._base_joint_ids: list[int] | None = None
        self._tip_joint_ids: list[int] | None = None
        self._hold_pos_base: Optional[torch.Tensor] = None
        self._hold_pos_tip: Optional[torch.Tensor] = None

    def resolve_joints(self, robot) -> None:
        ids, names = robot.find_joints(self.cfg.joint_regex)
        # Normalize ids to plain Python list[int]
        try:
            if torch.is_tensor(ids):
                ids_list: list[int] = [int(v) for v in ids.view(-1).tolist()]
            else:
                ids_list = [int(v) for v in ids]
        except Exception:
            try:
                flat = ids.tolist()  # numpy-like
                if isinstance(flat, (list, tuple)):
                    ids_list = [int(v) for v in flat]
                else:
                    ids_list = [int(flat)]
            except Exception:
                ids_list = [int(v) for v in list(ids)]
        if not self.cfg.split_base_and_tip:
            self._base_joint_ids = ids_list
            self._tip_joint_ids = ids_list[:0]
            return
        base_ids: list[int] = []
        tip_ids: list[int] = []
        for jid, name in zip(ids_list, names):
            if "finger_tip" in name:
                tip_ids.append(int(jid))
            else:
                base_ids.append(int(jid))
        self._base_joint_ids = base_ids
        self._tip_joint_ids = tip_ids

    def reset(self, robot) -> None:
        if self._base_joint_ids is None:
            self.resolve_joints(robot)
        # Hold current gripper state
        if self._base_joint_ids:
            self._hold_pos_base = robot.data.joint_pos[:, self._base_joint_ids].clone()
        if self._tip_joint_ids:
            self._hold_pos_tip = robot.data.joint_pos[:, self._tip_joint_ids].clone()

    def apply_hold(self, robot) -> None:
        if self._base_joint_ids and self._hold_pos_base is not None:
            robot.set_joint_position_target(self._hold_pos_base, joint_ids=self._base_joint_ids)
        if self._tip_joint_ids and self._hold_pos_tip is not None:
            robot.set_joint_position_target(self._hold_pos_tip, joint_ids=self._tip_joint_ids)

    def command_open(self, robot) -> None:
        self._command_to(robot, base_val=self.cfg.open_position, tip_val=0.0)

    def command_close(self, robot) -> None:
        tip_val = min(self.cfg.close_position * self.cfg.tip_ratio_on_close, 0.4) if self._tip_joint_ids else self.cfg.close_position
        self._command_to(robot, base_val=self.cfg.close_position, tip_val=tip_val)

    def _command_to(self, robot, *, base_val: float, tip_val: float) -> None:
        if self._base_joint_ids:
            target_base = torch.full(
                (robot.data.joint_pos.shape[0], len(self._base_joint_ids)),
                fill_value=base_val,
                dtype=robot.data.joint_pos.dtype,
                device=self.device,
            )
            robot.set_joint_position_target(target_base, joint_ids=self._base_joint_ids)
            self._hold_pos_base = target_base.clone()
        if self._tip_joint_ids:
            target_tip = torch.full(
                (robot.data.joint_pos.shape[0], len(self._tip_joint_ids)),
                fill_value=tip_val,
                dtype=robot.data.joint_pos.dtype,
                device=self.device,
            )
            robot.set_joint_position_target(target_tip, joint_ids=self._tip_joint_ids)
            self._hold_pos_tip = target_tip.clone()

    def set_drive_gains(self, prim_path: str) -> None:
        """Optionally tune joint drive gains via schema utilities over a prim sub-tree.

        Use this if you want to push low-level joint drive properties directly to USD/PhysX.
        """
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.sim.schemas.schemas_cfg import JointDrivePropertiesCfg
            cfg = JointDrivePropertiesCfg(
                stiffness=self.cfg.stiffness,
                damping=self.cfg.damping,
                max_velocity=self.cfg.max_velocity,
                max_effort=self.cfg.effort_limit,
            )
            sim_utils.modify_joint_drive_properties(prim_path, cfg)
        except Exception:
            # Best-effort: schema may not apply on all robot assets
            pass


