from __future__ import annotations

import json
import time
from typing import List

import torch

from assist.config import AssistConfig
from assist.policy import infer_policy
from assist.schemas import DetectedObject, Pose
from assist.window import RollingWindowBuilder
from assist.actions import build_align_and_grasp


class MockRobot:
    class Data:
        def __init__(self):
            self.body_pose_w = torch.zeros(1, 1, 7)
            # [x, y, z, w, x, y, z]
            self.body_pose_w[0, 0, 0:3] = torch.tensor([0.50, 0.10, 0.10])
            self.body_pose_w[0, 0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    def __init__(self):
        self.data = MockRobot.Data()


def make_object(id: str, xyz: list[float]) -> DetectedObject:
    return DetectedObject(
        id=id,
        label="object",
        color=None,
        pose=Pose(position_m=(float(xyz[0]), float(xyz[1]), float(xyz[2])), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)),
        bbox_xywh=None,
        confidence=1.0,
    )


def main():
    cfg = AssistConfig()
    robot = MockRobot()
    winb = RollingWindowBuilder(cfg)

    # Objects: red mug near +Y
    objs: List[DetectedObject] = [make_object("obj_red_mug_1", [0.60, 0.12, 0.10]), make_object("obj_green_can_2", [0.55, 0.20, 0.10])]

    # Simulate 1.2s of forward command toward red mug
    dt = 1.0 / cfg.assist_rate_hz
    now = int(time.time() * 1000)
    # Initialize window once to avoid unbound warnings
    win = winb.tick(now, robot, torch.zeros(1, 6), objs, robot_mode="manual")
    for i in range(12):
        now += int(dt * 1000)
        cmd = torch.zeros(1, 6)
        cmd[0, 1] = 0.05  # forward in +Y
        win = winb.tick(now, robot, cmd, objs, robot_mode="manual")

    pol = infer_policy(cfg, win)
    print("Policy output:", json.dumps({
        "kind": pol.kind,
        "text": pol.text,
        "confidence": round(pol.confidence, 2),
        "top": pol.top_object_id,
        "alt": pol.alt_object_id,
    }))

    # Build action plan toward top object
    if pol.top_object_id:
        top = next(o for o in win.objects if o.id == pol.top_object_id)
        ee = robot.data.body_pose_w[0, 0, 0:3].tolist()
        plan = build_align_and_grasp(cfg, (float(ee[0]), float(ee[1]), float(ee[2])), top.pose, dt=1.0/240.0)
        print(f"Action stream length: {len(plan.cmd_stream)} steps; gripper events: {plan.gripper_events}")
        if plan.cmd_stream:
            first = plan.cmd_stream[0][0, :6].tolist()
            print("First cmd (dx,dy,dz,rx,ry,rz):", [round(v, 4) for v in first])


if __name__ == "__main__":
    main()


