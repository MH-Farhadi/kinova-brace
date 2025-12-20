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
        self._unavailable_reason: str | None = None
        # Runtime-cached state (set by the caller, e.g. vla_v1) so we can satisfy
        # cuRobo APIs that require a start_state for planning.
        self._last_start_joint_pos: List[float] | None = None
        self._last_joint_names: List[str] | None = None
        # Dynamic world model (optional). Some cuRobo builds accept this at plan time.
        self._world_model: Any | None = None

        # Best-effort availability check. We don't hard-require cuRobo at import time.
        try:
            self._import_motion_gen_symbols()
        except Exception as e:
            self._available = False
            self._unavailable_reason = str(e)
            print(f"[MG][CUROBO_VLA][WARN] cuRobo MotionGen is not importable; will fall back to scripted. ({e})")

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

    def _normalize_robot_cfg_schema(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize config so cuRobo builds expecting robot_cfg.kinematics can still work.

        We have seen cuRobo versions that expect different schemas:
        - robot_cfg.kinematics.{urdf_path, base_link, ee_link, cspace...}
        - robot_cfg.kin_chain.{urdf_path, base_link, ee_link, joint_names...}

        This function ensures that if kin_chain exists but kinematics is missing,
        we synthesize a minimal kinematics dict with base_link/ee_link populated.
        """
        try:
            rcfg = cfg.get("robot_cfg")
            if not isinstance(rcfg, dict):
                return cfg
            kin = rcfg.get("kinematics")
            if isinstance(kin, dict) and (kin.get("base_link") or kin.get("ee_link")):
                return cfg
            kin_chain = rcfg.get("kin_chain")
            if not isinstance(kin_chain, dict):
                return cfg
            # Build a minimal kinematics block from kin_chain.
            kinematics = {
                "urdf_path": kin_chain.get("urdf_path"),
                "base_link": kin_chain.get("base_link"),
                "ee_link": kin_chain.get("ee_link"),
            }
            # Preserve joint_names into cspace if present.
            jn = kin_chain.get("joint_names")
            if jn is not None:
                kinematics["cspace"] = {"joint_names": jn}
            rcfg["kinematics"] = kinematics
            cfg["robot_cfg"] = rcfg
            return cfg
        except Exception:
            return cfg

    def _tensor_args_from_cfg(self, cfg: Dict[str, Any]) -> Any:
        """Best-effort TensorDeviceType construction for cuRobo."""
        try:
            import torch
            from curobo.types.tensor import TensorDeviceType  # type: ignore

            ta = cfg.get("tensor_args") if isinstance(cfg, dict) else None
            dev = None
            if isinstance(ta, dict):
                dev = ta.get("device", None)
            device = torch.device(str(dev) if dev else "cuda:0")
            return TensorDeviceType(device=device)
        except Exception:
            return None

    def _ensure_motion_gen(self) -> bool:
        if not self._available:
            if self._unavailable_reason:
                print(f"[MG][CUROBO_VLA][WARN] cuRobo unavailable: {self._unavailable_reason}")
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
                cfg_dict = self._normalize_robot_cfg_schema(cfg_dict)
                # Newer cuRobo versions (like the NVLabs repo) use MotionGenConfig.load_from_robot_config.
                if hasattr(MotionGenConfig, "load_from_robot_config"):
                    # Expect cfg_dict to contain robot_cfg: {kinematics: ...}
                    robot_cfg = cfg_dict.get("robot_cfg", None)
                    if not isinstance(robot_cfg, dict):
                        raise ValueError("Expected 'robot_cfg' dict in YAML for MotionGenConfig.load_from_robot_config")
                    # cuRobo expects robot_cfg to be nested under key 'robot_cfg'
                    robot_cfg_wrapped = {"robot_cfg": robot_cfg}

                    # Some cuRobo builds require base_link/base_link_name passed explicitly (not only inside YAML).
                    base_link = None
                    try:
                        kin = robot_cfg.get("kinematics", None)
                        if isinstance(kin, dict) and kin.get("base_link"):
                            base_link = str(kin.get("base_link"))
                    except Exception:
                        base_link = None
                    if base_link is None:
                        try:
                            kin_chain = robot_cfg.get("kin_chain", None)
                            if isinstance(kin_chain, dict) and kin_chain.get("base_link"):
                                base_link = str(kin_chain.get("base_link"))
                        except Exception:
                            base_link = None

                    # One-time diagnostic to confirm which schema we ended up with.
                    try:
                        if getattr(self, "_printed_schema_diag", False) is False:
                            self._printed_schema_diag = True  # type: ignore[attr-defined]
                            kin_keys = list((robot_cfg.get("kinematics") or {}).keys()) if isinstance(robot_cfg.get("kinematics"), dict) else []
                            kc_keys = list((robot_cfg.get("kin_chain") or {}).keys()) if isinstance(robot_cfg.get("kin_chain"), dict) else []
                            print(f"[MG][CUROBO_VLA] robot_cfg schema: kinematics_keys={kin_keys} kin_chain_keys={kc_keys} base_link={base_link}")
                    except Exception:
                        pass

                    # Optional world collision config: allow env var to point to cuRobo's built-in world configs.
                    # Example:
                    # - built-in: export CUROBO_WORLD_CFG=collision_primitives_3d.yml
                    # - custom file: export CUROBO_WORLD_CFG=/abs/path/to/world_demo.yml
                    try:
                        import os

                        world_cfg_name = os.environ.get("CUROBO_WORLD_CFG", "") or ""
                    except Exception:
                        world_cfg_name = ""
                    world_model: Any | None = None
                    if world_cfg_name:
                        # If user passed an absolute/relative path, load YAML and pass as dict.
                        if ("/" in world_cfg_name) or world_cfg_name.endswith((".yml", ".yaml")) and Path(world_cfg_name).exists():
                            try:
                                import yaml  # type: ignore

                                world_model = yaml.safe_load(Path(world_cfg_name).read_text()) or {}
                            except Exception:
                                world_model = world_cfg_name
                        else:
                            # Otherwise treat as a built-in cuRobo world config name (relative to cuRobo's world config dir).
                            world_model = world_cfg_name
                    tensor_args = self._tensor_args_from_cfg(cfg_dict)
                    ee_link = str(getattr(self.ctx, "ee_link_name", "")) or None

                    # Call with best-effort signature compatibility across cuRobo versions.
                    def _try_load_from_robot_config(payload: Dict[str, Any], *, with_tensor: bool) -> Any:
                        kwargs: Dict[str, Any] = {"world_model": world_model}
                        if ee_link is not None:
                            kwargs["ee_link_name"] = ee_link
                        if with_tensor and tensor_args is not None:
                            kwargs["tensor_args"] = tensor_args
                        # Try base_link variants if we have one
                        if base_link:
                            for key in ("base_link", "base_link_name", "robot_base_link", "robot_base_link_name"):
                                try:
                                    return MotionGenConfig.load_from_robot_config(payload, **{**kwargs, key: base_link})
                                except TypeError:
                                    # unexpected kw
                                    continue
                        # Fallback: no explicit base_link kw
                        return MotionGenConfig.load_from_robot_config(payload, **kwargs)

                    # Attempt wrapped + unwrapped payloads (different cuRobo builds expect different roots).
                    tried = []
                    last_err: Exception | None = None
                    for payload in (robot_cfg_wrapped, robot_cfg):
                        for with_tensor in (True, False):
                            try:
                                mg_cfg = _try_load_from_robot_config(payload, with_tensor=with_tensor)
                                break
                            except Exception as e:
                                last_err = e
                                tried.append((type(payload).__name__, with_tensor))
                                mg_cfg = None
                        if mg_cfg is not None:
                            break
                    if mg_cfg is None and last_err is not None:
                        raise last_err
                elif hasattr(MotionGenConfig, "from_dict"):
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
                # IMPORTANT: if config build fails, mark unavailable so we don't retry every plan.
                self._available = False
                self._unavailable_reason = f"MotionGenConfig init failed: {e}"
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

    def update_world_from_prim_paths(self, *, sim, robot, prim_paths: List[str]) -> bool:
        """Update cuRobo's internal collision world from live USD prims (best-effort).

        This is the missing link between Isaac's physics scene and cuRobo's planner:
        cuRobo plans against its own world model, so we must translate the current
        stage obstacles into cuRobo collision primitives.
        """
        if not self._ensure_motion_gen():
            return False
        mg = self._mg
        if mg is None:
            return False

        try:
            from motion_generation.curobo_world import build_curobo_world_cuboids

            world_model = build_curobo_world_cuboids(sim=sim, robot=robot, prim_paths=list(prim_paths))
        except Exception as e:
            print(f"[MG][CUROBO_VLA][WARN] Failed building cuRobo world from prims: {e}")
            return False

        # Always cache the world model; even if we can't "update" the MotionGen instance,
        # some cuRobo versions accept a world/world_model argument at plan time.
        self._world_model = world_model

        # Best-effort across cuRobo versions/APIs.
        try:
            if hasattr(mg, "update_world"):
                mg.update_world(world_model)  # type: ignore[misc]
                return True
        except Exception:
            pass

        try:
            if hasattr(mg, "update_world_model"):
                mg.update_world_model(world_model)  # type: ignore[misc]
                return True
        except Exception:
            pass

        try:
            wcc = getattr(mg, "world_coll_checker", None)
            if wcc is not None and hasattr(wcc, "update_world"):
                wcc.update_world(world_model)  # type: ignore[misc]
                return True
        except Exception:
            pass

        # Avoid spamming this warning every plan loop; it can materially slow down Kit UI.
        try:
            if getattr(self, "_printed_no_world_update_api", False) is False:
                self._printed_no_world_update_api = True  # type: ignore[attr-defined]
                print("[MG][CUROBO_VLA][WARN] No compatible cuRobo world update API found on MotionGen.")
        except Exception:
            print("[MG][CUROBO_VLA][WARN] No compatible cuRobo world update API found on MotionGen.")
        return False

    def set_start_state(self, *, joint_pos: List[float], joint_names: Optional[List[str]]) -> None:
        """Cache the current arm joint state for planning calls that require start_state."""
        self._last_start_joint_pos = [float(v) for v in joint_pos]
        self._last_joint_names = [str(n) for n in (joint_names or [])] if joint_names is not None else None

    def plan_joint_trajectory(
        self,
        *,
        start_joint_pos: List[float],
        joint_names: Optional[List[str]],
        target_pos_b: Tuple[float, float, float],
        target_quat_b_wxyz: Optional[Tuple[float, float, float, float]],
        pregrasp_offset_m: float,
    ) -> Any:
        """Plan a joint trajectory using cuRobo MotionGen (returns MotionGenResult).

        This is the preferred execution path for interactive demos because MotionGenResult
        naturally contains a joint-space interpolated trajectory (interpolated_plan).
        """
        if not self._ensure_motion_gen():
            raise RuntimeError("cuRobo MotionGen unavailable")
        # New cuRobo API: MotionGen.plan_single(start_state: JointState, goal_pose: Pose, ...)
        try:
            import time
            import torch
            from curobo.types.math import Pose  # type: ignore
            from curobo.types.state import JointState  # type: ignore

            gx, gy, gz = map(float, target_pos_b)
            goal_pos = torch.tensor([[gx, gy, gz + float(pregrasp_offset_m)]], dtype=torch.float32, device="cuda:0")
            if target_quat_b_wxyz is None:
                goal_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda:0")
            else:
                w, x, y, z = target_quat_b_wxyz
                goal_quat = torch.tensor([[float(w), float(x), float(y), float(z)]], dtype=torch.float32, device="cuda:0")
            goal_pose = Pose(position=goal_pos, quaternion=goal_quat)

            start_state = JointState.from_position(
                torch.tensor([start_joint_pos], dtype=torch.float32, device="cuda:0"),
                joint_names=joint_names,
            )
            # plan_single signature varies across cuRobo builds. We try to pass a world model if present,
            # but always fall back to a plain plan_single call.
            t_start = time.perf_counter()
            result = None
            if self._world_model is not None:
                for kw in ("world_model", "world", "world_cfg"):
                    try:
                        result = self._mg.plan_single(  # type: ignore[call-arg]
                            start_state=start_state,
                            goal_pose=goal_pose,
                            **{kw: self._world_model},
                        )
                        break
                    except TypeError:
                        result = None
            if result is None:
                result = self._mg.plan_single(start_state=start_state, goal_pose=goal_pose)  # type: ignore[call-arg]
            t_ms = (time.perf_counter() - t_start) * 1000.0
            if t_ms > 1.0:
                print(f"[MG][CUROBO_VLA] plan_single time={t_ms:.1f}ms world_model={'yes' if self._world_model is not None else 'no'}")
            return result
        except Exception as e:
            # Fall back to older API (dict goal_pose) if present
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
                    return self._mg.plan_single(goal_pose=goal)
                if hasattr(self._mg, "plan"):
                    return self._mg.plan(goal_pose=goal)
            except Exception:
                pass
            raise RuntimeError(f"cuRobo plan_joint_trajectory failed: {e}")

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
            gx, gy, gz = map(float, target_pos_b)
            # Prefer joint-trajectory planning if we have a cached start state.
            if self._last_start_joint_pos is None:
                raise RuntimeError("No cached start_state; call set_start_state(...) before planning.")

            joint_names = self._last_joint_names
            result = self.plan_joint_trajectory(
                start_joint_pos=self._last_start_joint_pos,
                joint_names=joint_names,
                target_pos_b=target_pos_b,
                target_quat_b_wxyz=target_quat_b_wxyz,
                pregrasp_offset_m=pregrasp_offset_m,
            )

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


