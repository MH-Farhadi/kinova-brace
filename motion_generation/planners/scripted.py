from __future__ import annotations

from typing import List, Tuple, Optional

from .base import BasePlanner
from utils import get_ee_pos_base_frame, yaw_from_quat_wxyz


class ScriptedPlanner(BasePlanner):
    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        x, y, z = float(target_pos_b[0]), float(target_pos_b[1]), float(target_pos_b[2])
        pre = (x, y, z + float(pregrasp_offset_m))
        grasp = (x, y, z + float(grasp_depth_m))
        lift = (x, y, z + float(lift_height_m))
        return [pre, grasp, lift]

    # High-level scripted execution helpers
    def execute_scripted_motion(
        self,
        *,
        sim,
        robot,
        controller,
        ctrl_cfg,
        grasp_pos_b: Tuple[float, float, float],
        grasp_quat_b_wxyz: Optional[Tuple[float, float, float, float]],
        grasp_quat_wxyz_w: Optional[Tuple[float, float, float, float]],
        dt: float,
        tolerance_m: float,
        inp,
    ) -> None:
        """Execute the scripted XY-align, yaw-rotate, and descend phases.

        This function centralizes the hard-coded scripted behavior so that the
        demo can remain thin and planner-agnostic.
        """
        # Phase 1: XY align at current Z
        ee_b = get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name)
        xy_align = (grasp_pos_b[0], grasp_pos_b[1], float(ee_b[2]))
        controller.set_mode("translate")
        inp.set_waypoints_b([xy_align])
        steps = 0
        while True:
            inp.set_current_pose_b(get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name))
            controller.step(robot, dt)
            sim.step()
            robot.update(dt)
            if len(inp._waypoints_b) == 0 or steps > 1000:
                break
            steps += 1

        # Phase 2: Rotate yaw to align with target yaw (from grasp quaternion if available)
        try:
            import math
            from isaaclab.utils.math import subtract_frame_transforms  # type: ignore[attr-defined]

            # Prefer target yaw in base frame if available
            target_yaw = (
                yaw_from_quat_wxyz(grasp_quat_b_wxyz)
                if grasp_quat_b_wxyz is not None
                else (yaw_from_quat_wxyz(grasp_quat_wxyz_w) if grasp_quat_wxyz_w is not None else None)
            )
            if target_yaw is None:
                raise RuntimeError("target yaw unavailable")

            # Current yaw in base frame from EE orientation
            root_pose_w = robot.data.root_pose_w
            ee_pose_w = robot.data.body_pose_w[:, int(robot.find_bodies([ctrl_cfg.ee_link_name])[0][0])]
            _, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            curr_yaw = yaw_from_quat_wxyz(
                (float(ee_quat_b[0, 0]), float(ee_quat_b[0, 1]), float(ee_quat_b[0, 2]), float(ee_quat_b[0, 3]))
            )
            dyaw = target_yaw - curr_yaw
            # Wrap to [-pi, pi]
            dyaw = (dyaw + math.pi) % (2.0 * math.pi) - math.pi

            controller.set_mode("rotate")
            # Distribute rotation over N steps
            step_ang = 0.02
            n_rot = max(1, int(abs(dyaw) / step_ang))
            inp.queue_rotate_z(dyaw, n_rot)
            for _ in range(n_rot):
                controller.step(robot, dt)
                sim.step()
                robot.update(dt)
        except Exception as e:
            print(f"[MG][EP][WARN] Yaw alignment skipped: {e}")

        # Phase 3: Descend straight down to target Z at aligned XY
        controller.set_mode("translate")
        descend = (grasp_pos_b[0], grasp_pos_b[1], grasp_pos_b[2])
        inp.set_waypoints_b([descend])
        steps = 0
        while True:
            inp.set_current_pose_b(get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name))
            controller.step(robot, dt)
            sim.step()
            robot.update(dt)
            if len(inp._waypoints_b) == 0 or steps > 500:
                break
            steps += 1


