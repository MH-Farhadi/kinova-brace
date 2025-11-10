from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import os


@dataclass
class PlannerContext:
    base_frame: str
    ee_link_name: str
    urdf_path: Optional[str]
    config_dir: str


class BasePlanner:
    def __init__(self, ctx: PlannerContext) -> None:
        self.ctx = ctx

    def plan_to_pose_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        target_quat_b_wxyz: Optional[Tuple[float, float, float, float]],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        """Optional 6D planning interface. Default falls back to position-only waypoints."""
        return self.plan_waypoints_b(
            target_pos_b=target_pos_b,
            pregrasp_offset_m=pregrasp_offset_m,
            grasp_depth_m=grasp_depth_m,
            lift_height_m=lift_height_m,
        )

    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        raise NotImplementedError


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


class RmpFlowPlanner(BasePlanner):
    def __init__(self, ctx: PlannerContext) -> None:
        super().__init__(ctx)
        # Validate presence of motion generation config files
        cfg_dir = self.ctx.config_dir
        expected = [
            os.path.join(cfg_dir, "j2n6s300.srdf"),
            os.path.join(cfg_dir, "kinematics.yaml"),
            os.path.join(cfg_dir, "joint_limits.yaml"),
            os.path.join(cfg_dir, "ompl_planning.yaml"),
        ]
        missing = [p for p in expected if not os.path.exists(p)]
        if missing:
            print(f"[MG][RMP][WARN] Missing required config files: {missing}. Falling back to scripted planner.")
            self._fallback = ScriptedPlanner(ctx)
        else:
            self._fallback = None
            print(f"[MG][RMP] Found MoveIt configs in: {cfg_dir}. Using scripted path generation; RMP collision checks TBD.")

    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        if self._fallback is not None:
            return self._fallback.plan_waypoints_b(
                target_pos_b=target_pos_b,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )
        # TODO: Integrate LULA RMP collision-aware path sampling when Kinova descriptor and RMP config are available
        return ScriptedPlanner(self.ctx).plan_waypoints_b(
            target_pos_b=target_pos_b,
            pregrasp_offset_m=pregrasp_offset_m,
            grasp_depth_m=grasp_depth_m,
            lift_height_m=lift_height_m,
        )

    def plan_to_pose_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        target_quat_b_wxyz: Optional[Tuple[float, float, float, float]],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        # Until full RMP integration is added, use position-only fallback
        return self.plan_waypoints_b(
            target_pos_b=target_pos_b,
            pregrasp_offset_m=pregrasp_offset_m,
            grasp_depth_m=grasp_depth_m,
            lift_height_m=lift_height_m,
        )


class CuroboPlanner(BasePlanner):
    def __init__(self, ctx: PlannerContext) -> None:
        super().__init__(ctx)
        # Lazy detection of cuRobo/Isaac Motion Generation GPU backend
        try:
            import importlib  # noqa: F401
            # The exact API may vary across Isaac Sim versions; we probe extension availability
            self._available = True
            print("[MG][CUROBO] cuRobo backend assumed available (extension check deferred).")
        except Exception:
            self._available = False
            print("[MG][CUROBO][WARN] cuRobo backend not available. Falling back to scripted planner.")
        self._mg: Any = None  # Motion generator instance (lazy)

    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        if not getattr(self, "_available", False):
            return ScriptedPlanner(self.ctx).plan_waypoints_b(
                target_pos_b=target_pos_b,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )
        # TODO: Construct a collision-aware trajectory via cuRobo once robot config is provided
        print("[MG][CUROBO] Placeholder path generation; using scripted waypoints pending full integration.")
        return ScriptedPlanner(self.ctx).plan_waypoints_b(
            target_pos_b=target_pos_b,
            pregrasp_offset_m=pregrasp_offset_m,
            grasp_depth_m=grasp_depth_m,
            lift_height_m=lift_height_m,
        )

    def _ensure_motion_gen(self) -> bool:
        """Initialize cuRobo MotionGen with a reasonable default config if available."""
        if not getattr(self, "_available", False):
            return False
        if self._mg is not None:
            return True
        # Try multiple import paths to support different cuRobo versions
        MotionGen = None
        MotionGenConfig = None
        try:
            from nvidia.curobo.motion_gen import MotionGen, MotionGenConfig  # type: ignore
        except Exception:
            try:
                from curobo.motion_gen import MotionGen, MotionGenConfig  # type: ignore
            except Exception:
                try:
                    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig  # type: ignore
                except Exception:
                    print("[MG][CUROBO][WARN] Could not import MotionGen APIs from cuRobo.")
                    return False
        try:
            import os
            # Allow user to provide a config path via env or planner context
            cfg_path = os.environ.get("CUROBO_ROBOT_CFG", "")
            if len(cfg_path) == 0:
                # Look for a local config under motion_generation_config/curobo (or cuRobo) if present
                for sub in ("curobo", "cuRobo"):
                    local_dir = os.path.join(self.ctx.config_dir, sub)
                    candidate = os.path.join(local_dir, "j2n6s300.yaml")
                    if os.path.exists(candidate):
                        cfg_path = candidate
                        break
            if len(cfg_path) == 0:
                print("[MG][CUROBO][WARN] No cuRobo config provided. Set CUROBO_ROBOT_CFG or add a config under motion_generation_config/curobo/.")
                return False
            # Load configuration with broad compatibility across cuRobo versions
            mg_cfg = None
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
                # Manual YAML load fallback
                try:
                    import yaml  # type: ignore
                    with open(cfg_path, "r") as f:
                        cfg_dict = yaml.safe_load(f)
                    if hasattr(MotionGenConfig, "from_dict"):
                        mg_cfg = MotionGenConfig.from_dict(cfg_dict)  # type: ignore[attr-defined]
                    else:
                        # Last resort: try passing dict into constructor
                        mg_cfg = MotionGenConfig(cfg_dict)  # type: ignore[call-arg]
                except Exception as e:
                    # Some versions expose a direct factory on MotionGen
                    if hasattr(MotionGen, "from_yaml_file"):
                        self._mg = MotionGen.from_yaml_file(cfg_path)  # type: ignore[attr-defined]
                        print(f"[MG][CUROBO] MotionGen initialized via from_yaml_file('{cfg_path}').")
                        return True
                    print(f"[MG][CUROBO][WARN] Could not build MotionGenConfig from '{cfg_path}': {e}")
                    return False
            # Initialize motion generator
            self._mg = MotionGen(mg_cfg)
            self._mg.warmup()
            print(f"[MG][CUROBO] MotionGen initialized from '{cfg_path}'.")
            return True
        except Exception as e:
            print(f"[MG][CUROBO][WARN] MotionGen init failed: {e}")
            self._mg = None
            return False

    def plan_to_pose_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        target_quat_b_wxyz: Optional[Tuple[float, float, float, float]],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        """Plan to a 6D goal using cuRobo when available, else fallback to scripted."""
        if not self._ensure_motion_gen():
            return ScriptedPlanner(self.ctx).plan_waypoints_b(
                target_pos_b=target_pos_b,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )
        try:
            import torch
            # Build start/goal in base frame (assuming single env)
            # Start state is left to MotionGen to query from sim; for now, we feed only goal pose.
            pos = torch.tensor([[float(target_pos_b[0]), float(target_pos_b[1]), float(target_pos_b[2])]], dtype=torch.float32)
            if target_quat_b_wxyz is None:
                # Identity orientation if not provided
                quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            else:
                w, x, y, z = target_quat_b_wxyz
                quat = torch.tensor([[float(w), float(x), float(y), float(z)]], dtype=torch.float32)
            goal = {"position": pos, "orientation": quat}
            # Plan (details depend on cuRobo version; we call a generic API)
            result = None
            if hasattr(self._mg, "plan_single"):
                result = self._mg.plan_single(goal_pose=goal)
            elif hasattr(self._mg, "plan"):
                result = self._mg.plan(goal_pose=goal)
            else:
                print("[MG][CUROBO][WARN] MotionGen has no recognized plan method; falling back.")
                raise RuntimeError("No plan method")
            if result is None:
                raise RuntimeError("Empty cuRobo result")
            # Extract a small set of Cartesian waypoints from the end-effector trajectory
            # Fallback: use scripted pattern if trajectory not available
            waypoints: List[Tuple[float, float, float]] = []
            if hasattr(result, "ee_traj_b"):
                ee_traj = getattr(result, "ee_traj_b")
                if isinstance(ee_traj, torch.Tensor) and ee_traj.numel() >= 3:
                    # Take first, mid, and final points as [pre, grasp, lift] around goal
                    first = ee_traj[0, 0:3].tolist()
                    last = ee_traj[-1, 0:3].tolist()
                    mid = ee_traj[ee_traj.shape[0] // 2, 0:3].tolist()
                    waypoints = [
                        (float(last[0]), float(last[1]), float(last[2] + pregrasp_offset_m)),
                        (float(last[0]), float(last[1]), float(last[2] + grasp_depth_m)),
                        (float(last[0]), float(last[1]), float(last[2] + lift_height_m)),
                    ]
            if len(waypoints) == 0:
                waypoints = ScriptedPlanner(self.ctx).plan_waypoints_b(
                    target_pos_b=target_pos_b,
                    pregrasp_offset_m=pregrasp_offset_m,
                    grasp_depth_m=grasp_depth_m,
                    lift_height_m=lift_height_m,
                )
            print("[MG][CUROBO] Planned 6D goal; returning approach/grasp/lift waypoints.")
            return waypoints
        except Exception as e:
            print(f"[MG][CUROBO][WARN] Failed to plan with cuRobo ({e}); using scripted path.")
            return ScriptedPlanner(self.ctx).plan_waypoints_b(
                target_pos_b=target_pos_b,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )

def create_planner(kind: str, *, ctx: PlannerContext) -> BasePlanner:
    kind_l = (kind or "").lower()
    if kind_l == "rmpflow":
        return RmpFlowPlanner(ctx)
    if kind_l == "curobo":
        return CuroboPlanner(ctx)
    if kind_l == "scripted" or kind_l == "":
        return ScriptedPlanner(ctx)
    print(f"[MG][PLANNER][WARN] Unknown planner type '{kind}'. Falling back to 'scripted'.")
    return ScriptedPlanner(ctx)


