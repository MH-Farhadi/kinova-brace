from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BasePlanner
from .scripted import ScriptedPlanner


@dataclass
class _CuroboVLARuntime:
    """Lazy cuRobo runtime handles.

    We keep imports fully lazy so this module can be imported in non-Isaac Python.
    """

    MotionGen: Any | None = None
    MotionGenConfig: Any | None = None


class CuroboVLAPlanner(BasePlanner):
    """cuRobo MotionGen planner wrapper intended for VLA reach-to-grasp.

    Goals:
    - Prefer cuRobo's collision-aware planning (when available/configured)
    - Return a dense *EE waypoint list in base frame* so we can execute it with the existing
      Cartesian jog controller + `WaypointFollowerInput` (no new controller needed).
    - Fall back gracefully when cuRobo is unavailable.
    """

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._available = True
        self._rt = _CuroboVLARuntime()
        self._mg: Any | None = None
        self._last_cfg_path: str | None = None

        # Best-effort availability check. We don't hard-require cuRobo at import time.
        try:
            self._import_motion_gen_symbols()
        except Exception:
            self._available = False

    def _import_motion_gen_symbols(self) -> None:
        if self._rt.MotionGen is not None and self._rt.MotionGenConfig is not None:
            return
        # Try multiple import paths across cuRobo / Isaac Sim versions.
        try:
            from nvidia.curobo.motion_gen import MotionGen, MotionGenConfig  # type: ignore
        except Exception:
            try:
                from curobo.motion_gen import MotionGen, MotionGenConfig  # type: ignore
            except Exception:
                try:
                    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig  # type: ignore
                except Exception as e:
                    raise ImportError(f"cuRobo MotionGen import failed: {e}")
        self._rt.MotionGen = MotionGen
        self._rt.MotionGenConfig = MotionGenConfig

    def _default_cfg_path(self) -> Optional[str]:
        # Expected layout: <ctx.config_dir>/cuRobo/j2n6s300.yaml
        cfg_dir = str(getattr(self.ctx, "config_dir", "") or "")
        if cfg_dir:
            candidate = Path(cfg_dir) / "cuRobo" / "j2n6s300.yaml"
            if candidate.exists():
                return str(candidate)
        return None

    def _patch_cfg_dict(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Patch known-bad config fields so the repo works out-of-the-box.

        The bundled YAML historically contained an absolute URDF path from a dev machine.
        We rewrite it to the repo-local URDF if available.
        """
        try:
            urdf_local = Path(getattr(self.ctx, "config_dir", "")) / "cuRobo" / "kinovaJacoJ2N6S300.urdf"
            if urdf_local.exists():
                # Common schema: robot_cfg.kinematics.urdf_path
                rcfg = cfg.get("robot_cfg") or {}
                kin = rcfg.get("kinematics") or {}
                if isinstance(kin, dict):
                    kin["urdf_path"] = str(urdf_local)
                    rcfg["kinematics"] = kin
                    cfg["robot_cfg"] = rcfg

                # Alternate schema: robot_cfg.kin_chain.urdf_path
                rcfg2 = cfg.get("robot_cfg") or {}
                kin_chain = rcfg2.get("kin_chain") or {}
                if isinstance(kin_chain, dict) and "urdf_path" in kin_chain:
                    kin_chain["urdf_path"] = str(urdf_local)
                    rcfg2["kin_chain"] = kin_chain
                    cfg["robot_cfg"] = rcfg2
        except Exception:
            pass
        return cfg

    def _ensure_motion_gen(self) -> bool:
        if not self._available:
            return False
        if self._mg is not None:
            return True
        try:
            self._import_motion_gen_symbols()
        except Exception as e:
            print(f"[MG][CUROBO_VLA][WARN] cuRobo import failed: {e}")
            self._available = False
            return False

        MotionGen = self._rt.MotionGen
        MotionGenConfig = self._rt.MotionGenConfig
        if MotionGen is None or MotionGenConfig is None:
            self._available = False
            return False

        # Resolve config path: env override wins, else repo default.
        try:
            import os

            cfg_path = os.environ.get("CUROBO_ROBOT_CFG", "") or ""
        except Exception:
            cfg_path = ""
        if not cfg_path:
            cfg_path = self._default_cfg_path() or ""
        if not cfg_path:
            print("[MG][CUROBO_VLA][WARN] No cuRobo config found. Expected planners_config/cuRobo/j2n6s300.yaml")
            return False

        self._last_cfg_path = cfg_path

        # Load configuration with broad compatibility across cuRobo versions.
        mg_cfg: Any | None = None
        try:
            if hasattr(MotionGenConfig, "load_from_file"):
                mg_cfg = MotionGenConfig.load_from_file(cfg_path)  # type: ignore[attr-defined]
            elif hasattr(MotionGenConfig, "from_yaml_file"):
                mg_cfg = MotionGenConfig.from_yaml_file(cfg_path)  # type: ignore[attr-defined]
            elif hasattr(MotionGenConfig, "from_yaml"):
                mg_cfg = MotionGenConfig.from_yaml(cfg_path)  # type: ignore[attr-defined]
        except Exception:
            mg_cfg = None

        if mg_cfg is None:
            # Manual YAML load fallback + patch absolute URDF paths
            try:
                import yaml  # type: ignore

                with open(cfg_path, "r") as f:
                    cfg_dict = yaml.safe_load(f) or {}
                if not isinstance(cfg_dict, dict):
                    raise ValueError("YAML root must be a dict")
                cfg_dict = self._patch_cfg_dict(cfg_dict)
                if hasattr(MotionGenConfig, "from_dict"):
                    mg_cfg = MotionGenConfig.from_dict(cfg_dict)  # type: ignore[attr-defined]
                else:
                    mg_cfg = MotionGenConfig(cfg_dict)  # type: ignore[call-arg]
            except Exception as e:
                # Some builds expose a direct factory on MotionGen.
                if hasattr(MotionGen, "from_yaml_file"):
                    try:
                        self._mg = MotionGen.from_yaml_file(cfg_path)  # type: ignore[attr-defined]
                        print(f"[MG][CUROBO_VLA] MotionGen initialized via from_yaml_file('{cfg_path}').")
                        return True
                    except Exception:
                        pass
                print(f"[MG][CUROBO_VLA][WARN] Failed to build MotionGen config from '{cfg_path}': {e}")
                return False

        try:
            self._mg = MotionGen(mg_cfg)
            # Warmup can be very expensive (and can appear like a UI "freeze").
            # Make it opt-in via env var.
            try:
                import os

                do_warmup = str(os.environ.get("CUROBO_WARMUP", "0")).lower() in ("1", "true", "yes", "y")
            except Exception:
                do_warmup = False
            if do_warmup and hasattr(self._mg, "warmup"):
                print("[MG][CUROBO_VLA] MotionGen warmup starting (CUROBO_WARMUP=1)...")
                self._mg.warmup()
                print("[MG][CUROBO_VLA] MotionGen warmup complete.")
            print(f"[MG][CUROBO_VLA] MotionGen initialized from '{cfg_path}'.")
            return True
        except Exception as e:
            print(f"[MG][CUROBO_VLA][WARN] MotionGen init failed: {e}")
            self._mg = None
            return False

    def _extract_ee_waypoints_from_result(self, result: Any) -> List[Tuple[float, float, float]]:
        """Best-effort extraction of base-frame EE positions from cuRobo result."""
        try:
            import torch
        except Exception:
            torch = None  # type: ignore

        # Common cuRobo fields (varies by version)
        candidates = [
            "ee_traj_b",
            "ee_traj",
            "eef_traj",
            "ee_pos_traj_b",
            "ee_pos_traj",
        ]
        traj = None
        for name in candidates:
            if hasattr(result, name):
                traj = getattr(result, name)
                if traj is not None:
                    break

        if traj is None:
            return []

        try:
            if torch is not None and isinstance(traj, torch.Tensor):
                t = traj
                if t.ndim == 3 and t.shape[1] == 1:
                    t = t[:, 0, :]
                if t.ndim == 2 and t.shape[1] >= 3:
                    pts = t[:, 0:3].detach().cpu().tolist()
                    return [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
        except Exception:
            pass

        try:
            pts = []
            for row in traj:
                pts.append((float(row[0]), float(row[1]), float(row[2])))
            return pts
        except Exception:
            return []

    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        if not self._ensure_motion_gen():
            return ScriptedPlanner(self.ctx).plan_waypoints_b(
                target_pos_b=target_pos_b,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )
        x, y, z = map(float, target_pos_b)
        return [
            (x, y, z + float(pregrasp_offset_m)),
            (x, y, z + float(grasp_depth_m)),
            (x, y, z + float(lift_height_m)),
        ]

    def plan_to_pose_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        target_quat_b_wxyz: Optional[Tuple[float, float, float, float]],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        if not self._ensure_motion_gen():
            return ScriptedPlanner(self.ctx).plan_to_pose_b(
                target_pos_b=target_pos_b,
                target_quat_b_wxyz=target_quat_b_wxyz,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )
        try:
            import torch

            gx, gy, gz = map(float, target_pos_b)
            goal_pos = torch.tensor([[gx, gy, gz + float(pregrasp_offset_m)]], dtype=torch.float32)
            if target_quat_b_wxyz is None:
                goal_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            else:
                w, x, y, z = target_quat_b_wxyz
                goal_quat = torch.tensor([[float(w), float(x), float(y), float(z)]], dtype=torch.float32)

            goal = {"position": goal_pos, "orientation": goal_quat}

            if hasattr(self._mg, "plan_single"):
                result = self._mg.plan_single(goal_pose=goal)
            elif hasattr(self._mg, "plan"):
                result = self._mg.plan(goal_pose=goal)
            else:
                raise RuntimeError("MotionGen has no recognized plan method")

            if result is None:
                raise RuntimeError("Empty cuRobo plan result")

            ee_pts = self._extract_ee_waypoints_from_result(result)

            # Downsample for stability
            if len(ee_pts) > 0:
                max_pts = 120
                stride = max(1, len(ee_pts) // max_pts)
                ee_pts = ee_pts[::stride]

            # Ensure we end with explicit pregrasp, then append grasp + lift.
            ee_pts = ee_pts or []
            ee_pts.append((gx, gy, gz + float(pregrasp_offset_m)))
            ee_pts.append((gx, gy, gz + float(grasp_depth_m)))
            ee_pts.append((gx, gy, gz + float(lift_height_m)))

            print(f"[MG][CUROBO_VLA] Planned to pregrasp with {len(ee_pts)} EE waypoints (cfg={self._last_cfg_path}).")
            return ee_pts
        except Exception as e:
            print(f"[MG][CUROBO_VLA][WARN] cuRobo planning failed ({e}); falling back to position waypoints.")
            return self.plan_waypoints_b(
                target_pos_b=target_pos_b,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )


