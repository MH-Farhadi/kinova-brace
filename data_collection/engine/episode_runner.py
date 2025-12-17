from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms

from controllers import CartesianVelocityJogController, CartesianVelocityJogConfig
from ..config import RunConfig
from controllers.input.waypoint_follower import WaypointFollowerInput
from ..engine.waypoints import generate_waypoints_base
import importlib
from data_collection.core.logger import SessionLogWriter, TickLoggingConfig
from data_collection.core.objects import ObjectsTracker


@dataclass
class EpisodeOutcome:
    success: bool
    reason: str
    target_id: str
    target_label: str


class EpisodeRunner:
    """Scripted reach-to-grasp runner that uses a waypoint follower input provider."""

    def __init__(
        self,
        *,
        sim: sim_utils.SimulationContext,
        robot,
        controller: CartesianVelocityJogController,
        session_logger: SessionLogWriter,
        tick_cfg: TickLoggingConfig,
        tracker: ObjectsTracker,
        id_to_label: Dict[str, str],
        run_cfg: RunConfig,
    ) -> None:
        self.sim = sim
        self.robot = robot
        self.controller = controller
        self.logger = session_logger
        self.tick_cfg = tick_cfg
        self.tracker = tracker
        self.id_to_label = id_to_label
        self.cfg = run_cfg
        # Input provider setup
        dt = float(self.sim.get_physics_dt())
        step_pos = float(self.controller.config.linear_speed_mps) * dt
        self.input = WaypointFollowerInput(step_pos_m=step_pos, tol_m=self.cfg.planner.tolerance_m, device=str(self.sim.device))
        self.controller.set_input_provider(self.input)
        # Grasp provider: OBB-based with minor-axis alignment
        try:
            obb_mod = importlib.import_module("motion_generation.grasp_estimation.obb")
            ProviderCls = getattr(obb_mod, "ObbGraspPoseProvider", None) or getattr(obb_mod, "OBBGraspPoseProvider", None)
            if ProviderCls is None:
                raise AttributeError("ObbGraspPoseProvider/OBBGraspPoseProvider not found")
            self._grasp_provider = ProviderCls(align_to_min_width=True)
        except Exception as e:
            raise ImportError(f"[MG][ERROR] Failed to import OBB grasp provider: {e}")

    def _read_ee_pose_b(self) -> torch.Tensor:
        body_ids, _ = self.robot.find_bodies([self.controller.config.ee_link_name])
        ee_id = int(body_ids[0])
        ee_pose_w = self.robot.data.body_pose_w[:, ee_id]
        root_pose_w = self.robot.data.root_pose_w
        ee_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return ee_pos_b[0]  # (3,)

    def _world_to_base(self, pos_w: torch.Tensor) -> torch.Tensor:
        root_pose_w = self.robot.data.root_pose_w
        base_pos_w = root_pose_w[0, 0:3]
        base_quat_w = root_pose_w[0, 3:7]
        # Convert pos_w to base frame: subtract translation; rotate by inverse quat
        from isaaclab.utils.math import quat_conjugate, quat_apply

        base_quat_inv = quat_conjugate(base_quat_w)
        rel_w = pos_w - base_pos_w
        pos_b = quat_apply(base_quat_inv, rel_w)
        return pos_b

    def _select_target(self, objects: List[Dict[str, Any]], target_label: Optional[str]) -> Optional[Dict[str, Any]]:
        if len(objects) == 0:
            return None
        if target_label is None:
            return objects[0]
        # Prefer exact match on label (case-insensitive)
        for o in objects:
            if str(o.get("label", "")).lower() == target_label.lower():
                return o
        # Fallback: substring match
        for o in objects:
            if target_label.lower() in str(o.get("label", "")).lower():
                return o
        return objects[0]

    def _log_tick_if_due(self, dt_accum: Dict[str, float], dt: float) -> None:
        if self.logger is None:
            return
        if self.tracker is None:
            return
        dt_accum["t"] += dt
        period = 1.0 / float(self.tick_cfg.log_rate_hz)
        if dt_accum["t"] + 1e-9 >= period:
            dt_accum["t"] = 0.0
            # Build objects payload
            objs_raw: List[Dict[str, Any]] = []
            try:
                for o in self.tracker.snapshot():
                    lbl = self.id_to_label.get(o.id, o.label)
                    objs_raw.append({
                        "id": o.id,
                        "label": lbl,
                        "pose": {"position_m": list(o.pose.position_m), "orientation_wxyz": list(o.pose.orientation_wxyz)},
                        "confidence": o.confidence,
                    })
            except Exception:
                pass
            last_cmd = getattr(self.input, "last_cmd", None)
            self.logger.write_tick(robot=self.robot, controller=self.controller, objects=objs_raw, last_user_cmd=last_cmd, cfg=self.tick_cfg)

    def run_episode(self, *, target_label: Optional[str]) -> EpisodeOutcome:
        dt = float(self.sim.get_physics_dt())
        # Ensure controller and input are reset
        self.controller.reset(self.robot)
        self.input.reset()
        self.controller.set_mode("translate")

        # Snapshot objects and pick target
        objs = []
        try:
            objs = self.tracker.snapshot()
        except Exception:
            pass
        if len(objs) == 0:
            return EpisodeOutcome(False, "no_objects", "", "")
        target = self._select_target(
            [{"id": o.id, "label": self.id_to_label.get(o.id, o.label), "pose": o.pose} for o in objs],
            target_label,
        )
        target_id = str(target["id"]) if target is not None else ""
        target_label_res = str(target.get("label", "")) if target is not None else ""
        if target is None:
            return EpisodeOutcome(False, "no_target", target_id, target_label_res)

        # Build waypoints in base frame using OBB-based grasp estimation
        # Reconstruct full prim path from tracker by matching trailing id
        prim_path = None
        try:
            for p in getattr(self.tracker, "prim_paths", []):
                if p.endswith("/" + target_id):
                    prim_path = p
                    break
        except Exception:
            prim_path = None
        if prim_path is not None:
            grasp_pos_w, _ = self._grasp_provider.get_grasp_pose_w(object_prim_path=prim_path, robot_prim_path=None)
            pos_w = torch.tensor(grasp_pos_w, dtype=torch.float32, device=self.sim.device)
        else:
            pos_w = torch.tensor(target["pose"].position_m, dtype=torch.float32, device=self.sim.device)
        pos_b = self._world_to_base(pos_w)
        waypoints = generate_waypoints_base(
            target_pos_b=(float(pos_b[0]), float(pos_b[1]), float(pos_b[2])),
            pregrasp_offset_m=self.cfg.task.pregrasp_offset_m,
            grasp_depth_m=self.cfg.task.grasp_depth_m,
            lift_height_m=self.cfg.task.lift_height_m,
        )
        # First phase: approach only to pregrasp and grasp (exclude lift)
        self.input.set_waypoints_b(waypoints[:2])

        # Open gripper briefly before approach
        self.controller.set_mode("gripper")
        self.input.queue_gripper(+1.0, steps=10)
        # A few steps to process open command
        for _ in range(10):
            self.input.set_current_pose_b(self._read_ee_pose_b())
            self.controller.step(self.robot, dt)
            self.sim.step()
            self.robot.update(dt)
        self.controller.set_mode("translate")

        steps = 0
        max_steps = int(self.cfg.episode.max_steps_per_phase) * 3
        dt_accum = {"t": 0.0}

        # Approach and descend to grasp
        while steps < max_steps:
            self.input.set_current_pose_b(self._read_ee_pose_b())
            self.controller.step(self.robot, dt)
            self.sim.step()
            self.robot.update(dt)
            self._log_tick_if_due(dt_accum, dt)

            # Done when no more waypoints queued
            if len(self.input._waypoints_b) == 0:  # internal state ok for runner
                break
            steps += 1

        # Close gripper
        self.controller.set_mode("gripper")
        self.input.queue_gripper(-1.0, steps=60)
        for _ in range(60):
            self.input.set_current_pose_b(self._read_ee_pose_b())
            self.controller.step(self.robot, dt)
            self.sim.step()
            self.robot.update(dt)
            self._log_tick_if_due(dt_accum, dt)

        # Allow additional settle time holding the closed shape
        for _ in range(20):
            self.input.set_current_pose_b(self._read_ee_pose_b())
            self.controller.step(self.robot, dt)
            self.sim.step()
            self.robot.update(dt)
            self._log_tick_if_due(dt_accum, dt)

        # Lift: set a single lift waypoint after gripper is closed
        self.controller.set_mode("translate")
        lift_pt = (waypoints[-1][0], waypoints[-1][1], waypoints[-1][2])
        self.input.set_waypoints_b([lift_pt])
        steps = 0
        while steps < int(self.cfg.episode.max_steps_per_phase):
            self.input.set_current_pose_b(self._read_ee_pose_b())
            self.controller.step(self.robot, dt)
            self.sim.step()
            self.robot.update(dt)
            self._log_tick_if_due(dt_accum, dt)
            if len(self.input._waypoints_b) == 0:
                break
            steps += 1

        # Evaluate success: object z increased by at least lift_height_m * 0.5
        success = False
        reason = "unknown"
        try:
            after = self.tracker.snapshot()
            curr = next((o for o in after if o.id == target_id), None)
            if curr is not None:
                z_after = float(curr.pose.position_m[2])
                z_target = float(target["pose"].position_m[2])
                if (z_after - z_target) >= (self.cfg.task.lift_height_m * 0.5):
                    success = True
                    reason = "lifted"
                else:
                    success = False
                    reason = "not_lifted_enough"
            else:
                success = False
                reason = "target_missing"
        except Exception:
            success = False
            reason = "eval_error"

        return EpisodeOutcome(success, reason, target_id, target_label_res)


