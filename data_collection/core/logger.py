from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, subtract_frame_transforms


@dataclass
class TickLoggingConfig:
    log_rate_hz: int = 10
    workspace_min: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None
    workspace_max: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None
    ee_link_name: str = "j2n6s300_end_effector"
    arm_joint_regex: str = "j2n6s300_joint_[1-6]"
    log_joint_data: bool = False  # Enable joint positions/velocities logging (for VLA training)


class SessionLogWriter:
    """Writer for metadata.json, ticks.jsonl, and events.jsonl following the v0 schema."""

    def __init__(self, root: Path, session_name: Optional[str] = None) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_name or f"session_{ts}"
        self.root = Path(root) / self.session_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.ticks_f = (self.root / "ticks.jsonl").open("a", buffering=1)
        self.events_f = (self.root / "events.jsonl").open("a", buffering=1)
        self.tick_idx = 0
        self._last_action_ms: Optional[int] = None
        self._last_safety_ms: Optional[int] = None
        # Caches
        self._ee_body_id: Optional[int] = None
        self._ee_jacobi_idx: Optional[int] = None
        self._arm_joint_ids: Optional[List[int]] = None
        self._obj_last_pos: Dict[str, Tuple[List[float], int]] = {}

    def close(self) -> None:
        try:
            self.ticks_f.close()
        except Exception:
            pass
        try:
            self.events_f.close()
        except Exception:
            pass

    def write_metadata(
        self,
        *,
        sim_dt: float,
        physics_substeps: int,
        seed: int,
        robot_name: str,
        ee_link: str,
        arm_joint_regex: str,
        log_rate_hz: int,
        window_len_s: float,
    ) -> None:
        meta = {
            "version": "data_collection_v0",
            "session_id": self.session_id,
            "started_ms": int(time.time() * 1000),
            "env": {"sim": True, "sim_dt": float(sim_dt), "physics_substeps": int(physics_substeps), "seed": int(seed)},
            "robot": {"name": robot_name, "ee_link": ee_link, "arm_joint_regex": arm_joint_regex},
            "config": {"log_rate_hz": int(log_rate_hz), "window_len_s": float(window_len_s)},
        }
        (self.root / "metadata.json").write_text(json.dumps(_format_numbers(meta, ndigits=4), indent=2))

    def _resolve_robot_indices(self, robot, ee_link_name: str, arm_joint_regex: str) -> None:
        if self._ee_body_id is None:
            body_ids, _ = robot.find_bodies([ee_link_name])
            self._ee_body_id = int(body_ids[0])
            self._ee_jacobi_idx = self._ee_body_id - 1 if robot.is_fixed_base else self._ee_body_id
        if self._arm_joint_ids is None:
            arm_joint_ids, _ = robot.find_joints(arm_joint_regex)
            # Normalize to Python ints
            try:
                if torch.is_tensor(arm_joint_ids):
                    self._arm_joint_ids = [int(v) for v in arm_joint_ids.view(-1).tolist()]
                else:
                    self._arm_joint_ids = [int(v) for v in arm_joint_ids]
            except Exception:
                self._arm_joint_ids = [int(v) for v in list(arm_joint_ids)]

    @staticmethod
    def _quat_inv(q: torch.Tensor) -> torch.Tensor:
        return quat_conjugate(q)

    def write_tick(
        self,
        *,
        robot,
        controller,
        objects: List[Dict[str, Any]],
        last_user_cmd: Optional[torch.Tensor],
        cfg: TickLoggingConfig,
        image_path: Optional[str] = None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        self._resolve_robot_indices(robot, cfg.ee_link_name, cfg.arm_joint_regex)
        ee_id = int(self._ee_body_id) if self._ee_body_id is not None else 0
        jac_idx = int(self._ee_jacobi_idx) if self._ee_jacobi_idx is not None else 0
        arm_ids = self._arm_joint_ids or []

        # World/base poses
        ee_pose_w = robot.data.body_pose_w[:, ee_id]
        root_pose_w = robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Velocities
        if hasattr(robot.data, "body_vel_w"):
            lin = robot.data.body_vel_w[0, ee_id, 0:3]
            ang = robot.data.body_vel_w[0, ee_id, 3:6]
        else:
            lin = torch.zeros(3, device=ee_pose_w.device)
            ang = torch.zeros(3, device=ee_pose_w.device)

        # Gripper state: infer from finger joints if possible
        gr_state = "open"
        try:
            joint_pos = robot.data.joint_pos[0]
            names = robot.data.joint_names
            idxs = [i for i, n in enumerate(names) if ("finger" in str(n))]
            if idxs:
                vals = torch.tensor([joint_pos[i] for i in idxs], dtype=torch.float32)
                gr_state = "close" if vals.mean().item() > 0.5 else "open"
        except Exception:
            pass

        # Safety metrics
        try:
            jac = robot.root_physx_view.get_jacobians()[:, jac_idx, :, arm_ids]
            svals = torch.linalg.svdvals(jac)
            s_min = float(svals[..., -1].item())
        except Exception:
            s_min = 0.0
        ws_min = cfg.workspace_min or (None, None, None)
        ws_max = cfg.workspace_max or (None, None, None)
        ee_b = ee_pos_b[0]
        clamped_axes = [
            (ws_min[0] is not None and float(ee_b[0]) <= float(ws_min[0]) + 1e-6)
            or (ws_max[0] is not None and float(ee_b[0]) >= float(ws_max[0]) - 1e-6),
            (ws_min[1] is not None and float(ee_b[1]) <= float(ws_min[1]) + 1e-6)
            or (ws_max[1] is not None and float(ee_b[1]) >= float(ws_max[1]) - 1e-6),
            (ws_min[2] is not None and float(ee_b[2]) <= float(ws_min[2]) + 1e-6)
            or (ws_max[2] is not None and float(ee_b[2]) >= float(ws_max[2]) - 1e-6),
        ]
        try:
            limits = robot.data.soft_joint_pos_limits[0, arm_ids]
            q = robot.data.joint_pos[0, arm_ids]
            near = bool(torch.any((q - limits[:, 0]) < 0.10) or torch.any((limits[:, 1] - q) < 0.10))
        except Exception:
            near = False

        # User
        if last_user_cmd is not None:
            u = last_user_cmd.view(-1)
            cmd6 = [float(v) for v in (u[:6].detach().to("cpu").tolist() + [0, 0, 0, 0, 0, 0])][:6]
            deadman = bool(torch.linalg.norm(u[:6]).item() > 1e-6)
        else:
            cmd6 = [0.0] * 6
            deadman = False

        # Objects and rel metrics
        obj_list: List[Dict[str, Any]] = []
        ee_pos_w = ee_pose_w[0, 0:3]
        base_quat_w = root_pose_w[0, 3:7]
        base_quat_inv = self._quat_inv(base_quat_w)
        for o in objects:
            pos_w = torch.tensor(o["pose"]["position_m"], dtype=torch.float32, device=root_pose_w.device)
            rel_w = pos_w - root_pose_w[0, 0:3]
            pos_b = quat_apply(base_quat_inv, rel_w)
            rel_to_ee_w = pos_w - ee_pos_w
            rel_to_ee_b = quat_apply(base_quat_inv, rel_to_ee_w)
            dist = float(torch.linalg.norm(rel_to_ee_b).item())
            dir_b = rel_to_ee_b / (torch.linalg.norm(rel_to_ee_b) + 1e-8)

            if sum(abs(c) for c in cmd6[:3]) > 0:
                v = torch.tensor(cmd6[0:3], dtype=torch.float32, device=dir_b.device)
                v = v / (torch.linalg.norm(v) + 1e-8)
                cosang = float(torch.clamp((v * dir_b).sum(), -1.0, 1.0).item())
                angle_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
            else:
                angle_deg = 180.0

            vel_w = [0.0, 0.0, 0.0]
            prev = self._obj_last_pos.get(o["id"]) if "id" in o else None
            if prev is not None:
                prev_pos, prev_t = prev
                dt_s = max(1e-3, (now_ms - prev_t) / 1000.0)
                vel_w = [float((pos_w[i].item() - prev_pos[i]) / dt_s) for i in range(3)]
            self._obj_last_pos[o["id"]] = (
                [float(pos_w[0].item()), float(pos_w[1].item()), float(pos_w[2].item())],
                now_ms,
            )

            obj_list.append(
                {
                    "id": o.get("id", ""),
                    "label": o.get("label", "object"),
                    "pose_w": {
                        "position_m": [float(pos_w[0]), float(pos_w[1]), float(pos_w[2])],
                        "orientation_wxyz": o["pose"].get("orientation_wxyz", [1, 0, 0, 0]),
                    },
                    "pose_b": {"position_m": [float(pos_b[0]), float(pos_b[1]), float(pos_b[2])], "orientation_wxyz": [1, 0, 0, 0]},
                    "confidence": float(o.get("confidence", 1.0)),
                    "tracking_quality": float(o.get("tracking_quality", 1.0)),
                    "stability_s": float(o.get("stability_s", 0.0)),
                    "rel_to_ee": {
                        "distance_m": dist,
                        "dir_b": [float(dir_b[0]), float(dir_b[1]), float(dir_b[2])],
                        "angle_to_cmd_deg": angle_deg,
                    },
                    "vel_w_mps": vel_w,
                }
            )

        rec = {
            "time_since_last_action_ms": 0 if self._last_action_ms is None else max(0, now_ms - self._last_action_ms),
            "time_since_last_safety_ms": 0 if self._last_safety_ms is None else max(0, now_ms - self._last_safety_ms),
        }

        # Build robot record with base fields
        robot_record = {
            "ee_pose_w": {
                "position_m": [float(ee_pose_w[0, 0].item()), float(ee_pose_w[0, 1].item()), float(ee_pose_w[0, 2].item())],
                "orientation_wxyz": [
                    float(ee_pose_w[0, 3].item()),
                    float(ee_pose_w[0, 4].item()),
                    float(ee_pose_w[0, 5].item()),
                    float(ee_pose_w[0, 6].item()),
                ],
            },
            "ee_pose_b": {
                "position_m": [float(ee_pos_b[0, 0].item()), float(ee_pos_b[0, 1].item()), float(ee_pos_b[0, 2].item())],
                "orientation_wxyz": [
                    float(ee_quat_b[0, 0].item()),
                    float(ee_quat_b[0, 1].item()),
                    float(ee_quat_b[0, 2].item()),
                    float(ee_quat_b[0, 3].item()),
                ],
            },
            "ee_linear_vel_mps": [float(lin[0].item()), float(lin[1].item()), float(lin[2].item())],
            "ee_angular_vel_rps": [float(ang[0].item()), float(ang[1].item()), float(ang[2].item())],
            "gripper": {
                "state": gr_state,
                "pose_w": {
                    "position_m": [float(ee_pose_w[0, 0].item()), float(ee_pose_w[0, 1].item()), float(ee_pose_w[0, 2].item())],
                    "orientation_wxyz": [
                        float(ee_pose_w[0, 3].item()),
                        float(ee_pose_w[0, 4].item()),
                        float(ee_pose_w[0, 5].item()),
                        float(ee_pose_w[0, 6].item()),
                    ],
                },
                "pose_b": {
                    "position_m": [float(ee_pos_b[0, 0].item()), float(ee_pos_b[0, 1].item()), float(ee_pos_b[0, 2].item())],
                    "orientation_wxyz": [
                        float(ee_quat_b[0, 0].item()),
                        float(ee_quat_b[0, 1].item()),
                        float(ee_quat_b[0, 2].item()),
                        float(ee_quat_b[0, 3].item()),
                    ],
                },
            },
            "safety": {"jacobian_s_min": s_min, "workspace_clamped_axes": clamped_axes, "near_joint_limit": near},
        }

        # Add joint data if enabled (for VLA training)
        if cfg.log_joint_data and arm_ids:
            try:
                # Get joint positions
                joint_positions = [
                    float(robot.data.joint_pos[0, jid].item()) 
                    for jid in arm_ids
                ]
                joint_names = [
                    str(robot.data.joint_names[jid]) 
                    for jid in arm_ids
                ]
                
                # Get joint velocities
                joint_velocities = []
                if hasattr(robot.data, "joint_vel"):
                    joint_velocities = [
                        float(robot.data.joint_vel[0, jid].item()) 
                        for jid in arm_ids
                    ]
                else:
                    joint_velocities = [0.0] * len(arm_ids)
                
                robot_record["joints"] = {
                    "positions": joint_positions,
                    "velocities": joint_velocities,
                    "names": joint_names,
                }
            except Exception:
                # If joint data extraction fails, skip it
                pass

        record = {
            "t_ms": now_ms,
            "tick_idx": self.tick_idx,
            "robot": robot_record,
            "user": {"joystick": {"cartesian_vel_cmd": cmd6, "speed_scale": 1.0}, "mode": "velocity", "deadman": deadman},
            "objects": obj_list,
            "recency": rec,
        }
        
        # Add image path if provided
        if image_path is not None:
            record["image"] = {"path": image_path}

        self.ticks_f.write(json.dumps(_format_numbers(record, ndigits=4)) + "\n")
        self.tick_idx += 1

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        now_ms = int(time.time() * 1000)
        if event_type == "action_start":
            self._last_action_ms = now_ms
        if event_type == "safety_event":
            self._last_safety_ms = now_ms
        rec = {"event_id": f"evt_{now_ms}", "t_ms": now_ms, "type": event_type, "data": data}
        self.events_f.write(json.dumps(_format_numbers(rec, ndigits=4)) + "\n")


def _format_numbers(obj: Any, ndigits: int = 4) -> Any:
    """Recursively format floats to strings with fixed decimals (zero-padded)."""

    if isinstance(obj, float):
        val = round(obj, ndigits)
        if val == -0.0:
            val = 0.0
        return f"{val:.{ndigits}f}"
    if isinstance(obj, list):
        return [_format_numbers(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return [_format_numbers(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {k: _format_numbers(v, ndigits) for k, v in obj.items()}
    return obj


