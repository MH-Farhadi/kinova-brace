from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from isaaclab.app import AppLauncher

# Ensure project root on sys.path for modular imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from controllers import (  # noqa: E402
    CartesianVelocityJogConfig,
    CartesianVelocityJogController,
)
from motion_generation.planners import PlannerContext, create_planner  # noqa: E402
from controllers.input.waypoint_follower import WaypointFollowerInput  # noqa: E402
from motion_generation.grasp_estimation.replicator import ReplicatorGraspProvider  # noqa: E402
from utils import (  # noqa: E402
    enable_optional_planner_extensions,
    reset_robot_to_origin,
    get_ee_pos_base_frame,
    world_to_base_pos,
    world_to_base_quat,
    yaw_from_quat_wxyz,
    stabilize_with_hold,
)


def run_grasp_loop_demo(args: argparse.Namespace) -> int:
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Enable optional planner extensions
    enable_optional_planner_extensions()

    # Import modules requiring active app
    import isaaclab.sim as sim_utils
    from environments.reach_to_grasp.utils import design_scene
    from environments.reach_to_grasp.config import DEFAULT_SCENE, DEFAULT_CAMERA
    from environments.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix

    # Setup sim
    phys = PhysicsConfig(device=args.device)
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    if not args.headless:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    print("[MG] App launched; building scene...")
    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]
    robot_prim_path = None
    try:
        robot_prim_path = str(getattr(getattr(robot, "cfg", None), "prim_path", None))
    except Exception:
        robot_prim_path = None

    # Controller
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(getattr(args, "speed", 0.7)),
        workspace_min=(0.20, -0.45, 0.01),
        workspace_max=(0.6, 0.45, 0.35),
        log_ee_pos=False,
        log_ee_frame="world",
        log_every_n_steps=9999,
    )
    print(f"[MG] Controller speed={ctrl_cfg.linear_speed_mps} m/s, ee_link={ctrl_cfg.ee_link_name}")
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))

    # Planner
    cfg_dir = str((Path(__file__).resolve().parent / "motion_generation_config").resolve())
    planner = create_planner(
        str(getattr(args, "planner", "scripted")),
        ctx=PlannerContext(
            base_frame="base_link",
            ee_link_name=str(ctrl_cfg.ee_link_name),
            urdf_path=None,
            config_dir=cfg_dir,
        ),
    )

    # Grasp provider selection
    default_rep_yaml = str((Path(__file__).resolve().parent / "motion_generation_config" / "gripper_configs" / "j2n6s300_topdown.yaml").resolve())
    grasp_kind = str(getattr(args, "grasp", "obb")).lower()
    rep_yaml = getattr(args, "rep_config_yaml", default_rep_yaml if grasp_kind == "replicator" else None)
    grasp_provider = None
    if grasp_kind == "replicator":
        try:
            grasp_provider = ReplicatorGraspProvider(
                gripper_prim_path=getattr(args, "rep_gripper_prim_path", robot_prim_path),
                config_yaml_path=rep_yaml,
                sampler_config=None,
                max_candidates=int(getattr(args, "rep_max_candidates", 16)),
            )
            print("[MG] Grasp provider: Replicator")
        except Exception as e:
            print(f"[MG][WARN] Replicator unavailable ({e}); falling back to AABB.")
            grasp_kind = "aabb"
    if grasp_kind != "replicator" or grasp_provider is None:
        try:
            obb_mod = importlib.import_module("motion_generation.grasp_estimation.obb")
            ProviderCls = getattr(obb_mod, "ObbGraspPoseProvider", None) or getattr(obb_mod, "OBBGraspPoseProvider", None)
            if ProviderCls is None:
                raise AttributeError("ObbGraspPoseProvider/OBBGraspPoseProvider not found")
            grasp_provider = ProviderCls(align_to_min_width=True)
            print("[MG] Grasp provider: OBB (minor-axis aligned)")
        except Exception as e:
            raise ImportError(f"[MG][ERROR] Failed to import OBB grasp provider: {e}")

    # Object loader defaults
    dataset_dirs = [str(d) for d in getattr(args, "objects_dataset", [])]
    if len(dataset_dirs) == 0:
        try:
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # type: ignore
            ycb_dir = f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
        except Exception:
            ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"
        dataset_dirs = [ycb_dir]
        print(f"[MG] Using default YCB dataset: {dataset_dirs[0]}")
    else:
        print(f"[MG] Using custom object datasets: {dataset_dirs}")
    phys_loader_kwargs = object_loader_kwargs_from_physix(phys)
    loader_cfg = ObjectLoaderConfig(
        dataset_dirs=dataset_dirs,
        bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
        min_distance=float(getattr(args, "min_distance", 0.1)),
        uniform_scale_range=((float(args.scale_min), float(args.scale_max)) if getattr(args, "scale_min", None) is not None and getattr(args, "scale_max", None) is not None else None),
        **phys_loader_kwargs,
    )
    loader = ObjectLoader(loader_cfg)

    # Loop
    prev_prim_paths: List[str] = []
    num_episodes = int(getattr(args, "num_episodes", 3))
    pregrasp = float(getattr(args, "pregrasp", 0.10))
    lift_h = float(getattr(args, "lift", 0.15))
    tolerance_m = float(getattr(args, "tolerance", 0.005))
    planner_kind = str(getattr(args, "planner", "scripted")).lower()

    print(f"[MG] Starting grasp loop: episodes={num_episodes} grasp={grasp_kind} planner={getattr(args, 'planner', 'scripted')}")
    successes = 0
    for ep in range(num_episodes):
        print(f"[MG][EP {ep+1}/{num_episodes}] Resetting world and robot...")
        # Remove previously spawned objects
        try:
            import omni.usd  # type: ignore[attr-defined]
            stage = omni.usd.get_context().get_stage()
            for p in prev_prim_paths:
                try:
                    if stage.GetPrimAtPath(p).IsValid():
                        stage.RemovePrim(p)
                except Exception:
                    pass
        except Exception:
            pass
        prev_prim_paths = []

        # Reset sim and robot, then spawn
        sim.reset()
        reset_robot_to_origin(sim, robot, (float(scene_origins[0][0]), float(scene_origins[0][1]), float(scene_origins[0][2])))
        controller.reset(robot)
        try:
            n = int(getattr(args, "num_objects", 1))
        except Exception:
            n = 1
        try:
            prev_prim_paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=n)
            print(f"[MG][EP] Spawned objects: {prev_prim_paths}")
        except Exception as e:
            print(f"[MG][EP][WARN] Object spawn failed: {e}")
            prev_prim_paths = []
        if len(prev_prim_paths) == 0:
            print("[MG][EP] No objects spawned; skipping episode.")
            continue

        # Get physics timestep
        dt = float(sim.get_physics_dt())

        # Allow objects to stabilize while holding robot at start position
        stabilize_steps = int(getattr(args, "stabilize_steps", 120))
        if stabilize_steps > 0:
            stabilize_with_hold(sim, robot, stabilize_steps, dt)

        # Select first object
        target_prim = prev_prim_paths[0]
        print(f"[MG][EP] Target prim: {target_prim}")

        # Grasp pose (world)
        pos_w, quat_wxyz_w = grasp_provider.get_grasp_pose_w(object_prim_path=target_prim, robot_prim_path=robot_prim_path)
        print(f"[MG][EP] Grasp pose (world): pos={pos_w} quat(wxyz)={quat_wxyz_w}")

        # Convert to base
        pos_b = world_to_base_pos(sim, robot, pos_w)
        quat_b = world_to_base_quat(sim, robot, quat_wxyz_w)
        print(f"[MG][EP] Grasp pose (base): pos={pos_b} quat_b(wxyz)={quat_b}")

        # Open gripper briefly
        controller.set_mode("gripper")
        inp = WaypointFollowerInput(step_pos_m=float(ctrl_cfg.linear_speed_mps) * dt, tol_m=tolerance_m, device=str(sim.device))
        controller.set_input_provider(inp)
        inp.queue_gripper(+1.0, steps=10)
        for _ in range(10):
            controller.step(robot, dt)
            sim.step(); robot.update(dt)

        if planner_kind == "scripted":
            # Phase 1: XY align at current Z
            ee_b = get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name)
            xy_align = (pos_b[0], pos_b[1], float(ee_b[2]))
            controller.set_mode("translate")
            inp.set_waypoints_b([xy_align])
            steps = 0
            while True:
                inp.set_current_pose_b(get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name))
                controller.step(robot, dt)
                sim.step(); robot.update(dt)
                if len(inp._waypoints_b) == 0 or steps > 2000:
                    break
                steps += 1

            # Phase 2: Rotate yaw to align with target yaw (from grasp quaternion if available)
            try:
                import math
                # Prefer target yaw in base frame
                target_yaw = yaw_from_quat_wxyz(quat_b) if quat_b is not None else (yaw_from_quat_wxyz(quat_wxyz_w) if quat_wxyz_w is not None else None)
                if target_yaw is None:
                    raise RuntimeError("target yaw unavailable")
                # Current yaw in base frame
                # Get current EE orientation in base frame
                root_pose_w = robot.data.root_pose_w
                ee_pose_w = robot.data.body_pose_w[:, int(robot.find_bodies([ctrl_cfg.ee_link_name])[0][0])]
                from isaaclab.utils.math import subtract_frame_transforms  # type: ignore[attr-defined]
                _, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                curr_yaw = yaw_from_quat_wxyz((float(ee_quat_b[0,0]), float(ee_quat_b[0,1]), float(ee_quat_b[0,2]), float(ee_quat_b[0,3])))
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
                    sim.step(); robot.update(dt)
            except Exception as e:
                print(f"[MG][EP][WARN] Yaw alignment skipped: {e}")

            # Phase 3: Descend straight down to target Z at aligned XY
            controller.set_mode("translate")
            descend = (pos_b[0], pos_b[1], pos_b[2])
            inp.set_waypoints_b([descend])
            steps = 0
            while True:
                inp.set_current_pose_b(get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name))
                controller.step(robot, dt)
                sim.step(); robot.update(dt)
                if len(inp._waypoints_b) == 0 or steps > 2000:
                    break
                steps += 1
        else:
            # Planner-driven approach + grasp
            try:
                if hasattr(planner, "plan_to_pose_b"):
                    waypoints: List[Tuple[float, float, float]] = getattr(planner, "plan_to_pose_b")(
                        target_pos_b=pos_b,
                        target_quat_b_wxyz=quat_b,
                        pregrasp_offset_m=pregrasp,
                        grasp_depth_m=0.00,
                        lift_height_m=lift_h,
                    )
                else:
                    raise RuntimeError("6D planning interface not available")
            except Exception as e:
                print(f"[MG][EP][WARN] plan_to_pose_b failed ({e}); using position-only waypoints.")
                waypoints = planner.plan_waypoints_b(
                    target_pos_b=pos_b,
                    pregrasp_offset_m=pregrasp,
                    grasp_depth_m=0.00,
                    lift_height_m=lift_h,
                )
            print(f"[MG][EP] Waypoints: {waypoints}")
            controller.set_mode("translate")
            inp.set_waypoints_b(waypoints[:2])
            steps = 0
            while True:
                inp.set_current_pose_b(get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name))
                controller.step(robot, dt)
                sim.step(); robot.update(dt)
                if len(inp._waypoints_b) == 0 or steps > 2000:
                    break
                steps += 1

        # Close gripper
        controller.set_mode("gripper")
        inp.queue_gripper(-1.0, steps=60)
        for _ in range(60):
            inp.set_current_pose_b(get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name))
            controller.step(robot, dt)
            sim.step(); robot.update(dt)

        # Lift
        controller.set_mode("translate")
        lift_pt = (pos_b[0], pos_b[1], pos_b[2] + lift_h)
        inp.set_waypoints_b([lift_pt])
        steps = 0
        while True:
            inp.set_current_pose_b(get_ee_pos_base_frame(robot, ctrl_cfg.ee_link_name))
            controller.step(robot, dt)
            sim.step(); robot.update(dt)
            if len(inp._waypoints_b) == 0 or steps > 2000:
                break
            steps += 1

        # Quick success heuristic: lifted z increased
        print(f"[MG][EP] Completed episode {ep+1}.")

    print(f"[MG] Completed: episodes={num_episodes}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Motion generation grasp loop demo")
    # Reuse the stable demo CLI for spawn/object/controller settings
    from scripts.cli import add_demo_cli_args
    add_demo_cli_args(ap)
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--objects-dataset", type=str, nargs="*", default=[])
    ap.add_argument("--pregrasp", type=float, default=0.10)
    ap.add_argument("--lift", type=float, default=0.15)
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--stabilize-steps", type=int, default=120, help="Number of sim steps to wait after spawning objects for physics to stabilize")
    ap.add_argument("--planner", type=str, default="scripted", choices=["scripted", "rmpflow", "curobo", "lula"])
    ap.add_argument("--grasp", type=str, default="obb", choices=["obb", "replicator"])
    ap.add_argument("--rep-gripper-prim-path", type=str, default=None)
    ap.add_argument("--rep-config-yaml", type=str, default=None)
    AppLauncher.add_app_launcher_args(ap)
    args_cli = ap.parse_args()
    raise SystemExit(run_grasp_loop_demo(args_cli))


