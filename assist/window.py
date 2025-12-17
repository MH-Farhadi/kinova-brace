from __future__ import annotations
from collections import deque
from typing import Deque, List, Optional, Tuple

import torch

from .config import AssistConfig
from .schemas import (
    DetectedObject,
    Event,
    Pose,
    RobotState,
    RollingWindow,
    UserInput,
)


class RollingWindowBuilder:
    """Maintains a fixed-length rolling context at assist rate."""

    def __init__(self, cfg: AssistConfig):
        self.cfg = cfg
        self.period_s = 1.0 / float(cfg.assist_rate_hz)
        self.window_s = float(cfg.window_len_s)
        self.max_len = max(1, int(round(self.window_s / self.period_s)))
        self.robot_states: Deque[RobotState] = deque(maxlen=self.max_len)
        self.user_inputs: Deque[UserInput] = deque(maxlen=self.max_len)
        self.events: Deque[Event] = deque(maxlen=self.max_len)

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    def _read_robot_state(self, robot) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
        # World pose of EE and base/root
        ee_pose_w = robot.data.body_pose_w[:, robot.data.body_names.index("j2n6s300_end_effector")] if False else robot.data.body_pose_w[:, 0]  # placeholder if names missing
        # Better: directly via known indices set during controller.reset; for v0, read first EE if available
        pos_w = tuple(float(v) for v in ee_pose_w[0, 0:3].tolist())
        quat_w = tuple(float(v) for v in ee_pose_w[0, 3:7].tolist())
        lin_vel = tuple(float(v) for v in robot.data.body_vel_w[0, 0, 0:3].tolist()) if hasattr(robot.data, "body_vel_w") else (0.0, 0.0, 0.0)
        ang_vel = tuple(float(v) for v in robot.data.body_vel_w[0, 0, 3:6].tolist()) if hasattr(robot.data, "body_vel_w") else (0.0, 0.0, 0.0)
        return pos_w, quat_w, lin_vel, ang_vel

    def tick(
        self,
        now_ms: int,
        robot,
        last_user_cmd: Optional[torch.Tensor],
        objects: List[DetectedObject],
        robot_mode: str = "manual",
    ) -> RollingWindow:
        # Robot snapshot
        pos_w, quat_w, lin_vel, ang_vel = self._read_robot_state(robot)
        robot_state = RobotState(
            t_ms=now_ms,
            ee_pose=Pose(position_m=pos_w, orientation_wxyz=quat_w),
            ee_linear_vel_mps=lin_vel,
            ee_angular_vel_rps=ang_vel,
            gripper_open_frac=0.5,  # v0: placeholder; could be read from gripper controller
            mode=robot_mode,
        )
        self.robot_states.append(robot_state)

        # User snapshot
        if last_user_cmd is None:
            cmd6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            speed_scale = 1.0
            mode = "velocity"
        else:
            if last_user_cmd.ndim == 2:
                arr = last_user_cmd[0]
            else:
                arr = last_user_cmd
            arr = arr.detach().to("cpu")
            # Ensure 6 components
            padded = torch.zeros(6, dtype=arr.dtype)
            padded[: min(6, arr.numel())] = arr[: min(6, arr.numel())]
            cmd6 = tuple(float(v) for v in padded.tolist())  # type: ignore[assignment]
            speed_scale = 1.0
            mode = "velocity"
        user_input = UserInput(
            t_ms=now_ms,
            cartesian_vel_cmd=cmd6,  # type: ignore[arg-type]
            speed_scale=speed_scale,
            mode=mode,
        )
        self.user_inputs.append(user_input)

        # Build window
        win = RollingWindow(
            window_ms=int(self.window_s * 1000.0),
            robot_states=list(self.robot_states),
            user_inputs=list(self.user_inputs),
            objects=objects,
            events=list(self.events),
        )
        return win


