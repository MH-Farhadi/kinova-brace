from __future__ import annotations

import argparse
import importlib
import sys
import random
from pathlib import Path
from typing import List, Tuple
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from controllers import (  
    CartesianVelocityJogConfig,
    CartesianVelocityJogController,
)
from motion_generation.planners import PlannerContext, create_planner  
from controllers.input.waypoint_follower import WaypointFollowerInput  
from motion_generation.grasp_estimation.replicator import ReplicatorGraspProvider  
from motion_generation.mogen import MotionGenerationAgent
from utils import (  
    enable_optional_planner_extensions,
    reset_robot_to_origin,
    get_ee_pos_base_frame,
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
            grasp_kind = "obb"
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
        uniform_scale_range=(
            (float(args.scale_min), float(args.scale_max))
            if getattr(args, "scale_min", None) is not None and getattr(args, "scale_max", None) is not None
            else None
        ),
        include_labels=[
            # "banana",
            "bleach_cleanser",
            # "bowl",
            # "cracker_box",
            # "extra_large_clamp",
            # "foam_brick",
            "gelatin_box",
            # "large_clamp",
            # "large_marker",
            "master_chef_can",
            "mug",
            "mustard_bottle",
            # "pitcher_base",
            "potted_meat_can",
            # "power_drill",
            "pudding_box",
            # "scissors",
            "sugar_box",
            "tomato_soup_can",
            "tuna_fish_can",
            # "wood_block",
        ],
        **phys_loader_kwargs,
    )
    loader = ObjectLoader(loader_cfg)

    # High-level motion generation helper for label-based target selection and grasp queries
    agent = MotionGenerationAgent(
        sim=sim,
        robot=robot,
        controller=controller,
        planner=planner,
        grasp_provider=grasp_provider,
        loader=loader,
        robot_prim_path=robot_prim_path,
    )

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
        stabilize_steps = int(getattr(args, "stabilize_steps", 500))
        if stabilize_steps > 0:
            stabilize_with_hold(sim, robot, stabilize_steps, dt)

        # Randomly select a target object from the spawned set and compute latest grasp pose
        target_prim = random.choice(prev_prim_paths)
        pos_w, quat_wxyz_w, pos_b, quat_b = agent.compute_current_grasp_for_prim(target_prim)
        print(f"[MG][EP] Target prim: {target_prim}")
        print(f"[MG][EP] Grasp pose (world): pos={pos_w} quat(wxyz)={quat_wxyz_w}")
        print(f"[MG][EP] Grasp pose (base): pos={pos_b} quat_b(wxyz)={quat_b}")

        # Open gripper briefly
        controller.set_mode("gripper")
        inp = WaypointFollowerInput(
            step_pos_m=float(ctrl_cfg.linear_speed_mps) * dt,
            tol_m=tolerance_m,
            device=str(sim.device),
        )
        controller.set_input_provider(inp)
        inp.queue_gripper(+1.0, steps=10)
        for _ in range(10):
            controller.step(robot, dt)
            sim.step()
            robot.update(dt)

        if planner_kind == "scripted":
            # Delegate scripted motion phases (XY align, yaw rotate, descend) to the ScriptedPlanner
            from motion_generation.planners import ScriptedPlanner  # local import to avoid circular deps

            assert isinstance(planner, ScriptedPlanner), "Scripted planner expected when planner_kind='scripted'"
            planner.execute_scripted_motion(
                sim=sim,
                robot=robot,
                controller=controller,
                ctrl_cfg=ctrl_cfg,
                grasp_pos_b=pos_b,
                grasp_quat_b_wxyz=quat_b,
                grasp_quat_wxyz_w=quat_wxyz_w,
                dt=dt,
                tolerance_m=tolerance_m,
                inp=inp,
            )
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
    # Import CLI configuration from motion_generation.cli
    from motion_generation.cli import add_motion_gen_cli_args
    
    ap = argparse.ArgumentParser(description="Motion generation grasp loop demo")
    add_motion_gen_cli_args(ap)
    args_cli = ap.parse_args()
    raise SystemExit(run_grasp_loop_demo(args_cli))


