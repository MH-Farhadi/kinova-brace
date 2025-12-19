from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from data_collection.envs.registry import get_envs
from data_collection.profiles.spec import ProfileSpec


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", type=str, default="reach_to_grasp_VLA", choices=sorted(get_envs().keys()))
    parser.add_argument("--logs-root", type=str, default="logs/data_collection")
    parser.add_argument("--log-rate-hz", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--control", type=str, default="keyboard", choices=["keyboard", "idle", "planner"])
    parser.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"], help="Image format for saving")

    # Planner-driven reach-to-grasp (VLA)
    parser.add_argument(
        "--planner",
        type=str,
        default="curobo_vla",
        choices=["curobo_vla", "curobo", "lula", "rmpflow", "scripted"],
        help="Planner backend for --control planner (default: curobo_vla)",
    )
    parser.add_argument("--target-label", type=str, default=None, help="Optional target object label filter")
    parser.add_argument("--pregrasp", type=float, default=0.10, help="Pre-grasp offset above object top (m)")
    parser.add_argument(
        "--grasp-depth",
        type=float,
        default=-0.04,
        help="Grasp depth relative to the estimated object top (m). Negative goes downward.",
    )
    parser.add_argument(
        "--ignore-grasp-orientation",
        action="store_true",
        default=True,
        help="Plan with position-only goals (faster/more robust than constraining EE orientation). Default: true.",
    )
    parser.add_argument("--lift", type=float, default=0.15, help="Lift height after grasp (m)")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Waypoint tolerance for planner control (m)")
    parser.add_argument("--stabilize-steps", type=int, default=300, help="Hold steps before first plan (physics settle)")
    parser.add_argument("--gripper-open-steps", type=int, default=10, help="Steps to open gripper before approach")
    parser.add_argument("--gripper-close-steps", type=int, default=60, help="Steps to close gripper at grasp")

    # Episode control (NEW in v1.1): end after grasp+lift, then reset and repeat.
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to run (planner mode)")
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=3000,
        help="Hard cap on physics steps per episode to avoid getting stuck",
    )
    parser.add_argument(
        "--render-rate-hz",
        type=float,
        default=60.0,
        help="Max render rate for interactive runs. Physics still steps at sim dt; rendering is throttled.",
    )
    parser.add_argument(
        "--respawn-each-episode",
        action="store_true",
        help=(
            "Re-randomize object poses each episode. "
            "Note: for PhysX GPU stability we do NOT destroy/recreate rigid bodies; we teleport existing ones."
        ),
    )

    # NEW in v1: dynamic cuRobo world sync (USD -> cuRobo cuboids)
    parser.add_argument(
        "--curobo-world-from-scene",
        action="store_true",
        help="Update cuRobo world collision model from current USD scene prim bounds before planning.",
    )
    parser.add_argument(
        "--curobo-world-include-table",
        action="store_true",
        default=True,
        help="Include /World/Origin1/Table as an obstacle when building cuRobo world (default: true).",
    )


def run(args: argparse.Namespace) -> int:
    # Ensure kinova-isaac root is first on sys.path (Kit may mutate sys.path).
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[2]
    root_str = str(ROOT)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    _env_mod = sys.modules.get("environments")
    if _env_mod is not None and not hasattr(_env_mod, "__path__"):
        del sys.modules["environments"]
    # Same collision as in collect_data.py: Isaac may preload `cv2.utils` which shadows our `utils/`.
    _utils_mod = sys.modules.get("utils")
    if _utils_mod is not None:
        _utils_file = str(getattr(_utils_mod, "__file__", "") or "")
        if _utils_file and root_str not in _utils_file:
            for _k in list(sys.modules.keys()):
                if _k == "utils" or _k.startswith("utils."):
                    del sys.modules[_k]

    # Heavy imports must happen only after Kit is started.
    from isaaclab.app import AppLauncher
    from data_collection.core.input_mux import CommandMuxInputProvider
    from data_collection.core.logger import SessionLogWriter, TickLoggingConfig
    from data_collection.core.objects import ObjectsTracker

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch  # noqa: E402
    try:
        import numpy as np  # noqa: E402
    except Exception as e:
        print(f"[VLA_V1] ERROR: numpy is required for image saving but could not be imported: {e}")
        print("[VLA_V1] Please install numpy in your environment (e.g., `pip install numpy`) and retry.")
        return 2

    import carb  # noqa: E402

    carb_settings = carb.settings.get_settings()
    enable_cameras = bool(getattr(args, "enable_cameras", False))
    carb_settings.set_bool("/isaaclab/cameras_enabled", enable_cameras)
    print(f"[VLA_V1] enable_cameras flag value: {enable_cameras}")
    print(f"[VLA_V1] carb /isaaclab/cameras_enabled={carb_settings.get('/isaaclab/cameras_enabled')}")

    import importlib
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    from controllers import CartesianVelocityJogConfig, CartesianVelocityJogController
    from environments.utils.object_loader import ObjectLoader, ObjectLoaderConfig, SpawnBounds
    from environments.utils.physix import PhysicsConfig, apply_to_simulation_cfg, object_loader_kwargs_from_physix

    env_spec = get_envs()[str(getattr(args, "env", "reach_to_grasp_VLA"))]
    env_cfg_mod = importlib.import_module(f"{env_spec.module_base}.config")
    env_utils_mod = importlib.import_module(f"{env_spec.module_base}.utils")
    DEFAULT_SCENE = getattr(env_cfg_mod, "DEFAULT_SCENE")
    DEFAULT_CAMERA = getattr(env_cfg_mod, "DEFAULT_CAMERA", None)
    DEFAULT_TOP_DOWN_CAMERA = getattr(env_cfg_mod, "DEFAULT_TOP_DOWN_CAMERA", None)
    design_scene = getattr(env_utils_mod, "design_scene")
    create_topdown_camera = getattr(importlib.import_module("environments.utils.camera"), "create_topdown_camera")

    # Setup sim
    phys = PhysicsConfig(device=str(getattr(args, "device", "cuda:0")))
    sim_cfg = sim_utils.SimulationCfg(device=phys.device)
    apply_to_simulation_cfg(sim_cfg, phys)
    sim = sim_utils.SimulationContext(sim_cfg)
    if (not getattr(args, "headless", False)) and DEFAULT_CAMERA is not None:
        sim.set_camera_view(DEFAULT_CAMERA.eye, DEFAULT_CAMERA.target)

    # Build scene and robot
    scene_entities, scene_origins = design_scene(DEFAULT_SCENE)
    robot = scene_entities["kinova_j2n6s300"]

    # Create top-down camera prim
    if DEFAULT_TOP_DOWN_CAMERA is not None:
        create_topdown_camera(DEFAULT_TOP_DOWN_CAMERA)
        print(f"[VLA_V1] Top-down camera created at: {DEFAULT_TOP_DOWN_CAMERA.prim_path}")

    # Spawn objects (v1 uses helpers so we can respawn per episode)
    spawned_paths: list[str] = []
    id_to_label: Dict[str, str] = {}
    loader: ObjectLoader | None = None

    def _yaw_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
        import math

        half = 0.5 * float(yaw_rad)
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    def _spawn_objects() -> tuple[list[str], Dict[str, str]]:
        if getattr(args, "no_objects", False):
            return [], {}

        try:
            ycb_dir = f"{ISAAC_NUCLEUS_DIR}/Props/YCB"
        except Exception:
            ycb_dir = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Props/YCB"

        scale_range = None
        if getattr(args, "scale_min", None) is not None and getattr(args, "scale_max", None) is not None:
            scale_range = (float(args.scale_min), float(args.scale_max))

        nonlocal loader
        if loader is None:
            phys_loader_kwargs = object_loader_kwargs_from_physix(phys)
            loader_cfg = ObjectLoaderConfig(
                dataset_dirs=[ycb_dir],
                bounds=SpawnBounds(min_xyz=tuple(args.spawn_min), max_xyz=tuple(args.spawn_max)),
                min_distance=float(getattr(args, "min_distance", 0.1)),
                uniform_scale_range=scale_range,
                **phys_loader_kwargs,
            )
            loader = ObjectLoader(loader_cfg)

        paths = loader.spawn(parent_prim_path="/World/Origin1", num_objects=int(getattr(args, "num_objects", 0)))
        try:
            prim_to_label = loader.get_last_spawn_labels()
            lbl_map = {str(p).split("/")[-1]: str(lbl) for p, lbl in prim_to_label.items()}
        except Exception:
            lbl_map = {}
        return paths, lbl_map

    # IMPORTANT: spawn objects once up-front so planner mode has a loader + targets available.
    spawned_paths, id_to_label = _spawn_objects()

    def _rerandomize_object_poses(paths: list[str]) -> None:
        """Teleport existing objects to new random poses (no delete/recreate).

        This avoids PhysX GPU tensor crashes that can happen if we destroy and recreate
        dynamic rigid bodies while tensor views exist.
        """
        if not paths:
            return
        try:
            import random
            import math
            import torch
            from isaacsim.core.simulation_manager import SimulationManager
        except Exception:
            return

        # Bounds are in meters relative to parent prim; in this env parent is /World/Origin1,
        # which is identity in world for single-origin usage.
        bmin = tuple(float(v) for v in getattr(args, "spawn_min", (0.30, -0.20, 0.81)))
        bmax = tuple(float(v) for v in getattr(args, "spawn_max", (0.55, 0.20, 0.81)))
        min_dist = float(getattr(args, "min_distance", 0.10))

        # Rejection sample XY, snap Z near configured surface.
        positions: list[tuple[float, float, float]] = []
        tries = 0
        while len(positions) < len(paths) and tries < 2000:
            tries += 1
            x = random.uniform(bmin[0], bmax[0])
            y = random.uniform(bmin[1], bmax[1])
            z = random.uniform(bmin[2], bmax[2])
            cand = (x, y, z)
            ok = True
            for p in positions:
                dx = cand[0] - p[0]
                dy = cand[1] - p[1]
                dz = cand[2] - p[2]
                if math.sqrt(dx * dx + dy * dy + dz * dz) < min_dist:
                    ok = False
                    break
            if ok:
                positions.append(cand)

        if len(positions) != len(paths):
            # Fallback: allow overlaps rather than failing.
            positions = [
                (
                    random.uniform(bmin[0], bmax[0]),
                    random.uniform(bmin[1], bmax[1]),
                    random.uniform(bmin[2], bmax[2]),
                )
                for _ in paths
            ]

        sim_view = SimulationManager.get_physics_sim_view()
        for prim_path, pos in zip(paths, positions):
            try:
                rb_view = sim_view.create_rigid_body_view(str(prim_path))
                yaw = random.uniform(-math.pi, math.pi)
                qw, qx, qy, qz = _yaw_quat_wxyz(yaw)
                # RigidBodyView transform format: [x,y,z,qx,qy,qz,qw]
                tf = torch.tensor([[float(pos[0]), float(pos[1]), float(pos[2]), float(qx), float(qy), float(qz), float(qw)]], device=sim.device)
                if hasattr(rb_view, "set_transforms"):
                    rb_view.set_transforms(tf)
                if hasattr(rb_view, "set_linear_velocities"):
                    rb_view.set_linear_velocities(torch.zeros((1, 3), device=sim.device))
                if hasattr(rb_view, "set_angular_velocities"):
                    rb_view.set_angular_velocities(torch.zeros((1, 3), device=sim.device))
            except Exception:
                # Best-effort: skip failures (keeps episode alive)
                continue

    # Camera sensor for image capture (same pattern as vla_v0)
    camera_sensor = None
    if DEFAULT_TOP_DOWN_CAMERA is not None:
        try:
            camera_cfg = CameraCfg(
                prim_path=DEFAULT_TOP_DOWN_CAMERA.prim_path,
                offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                spawn=None,
                data_types=["rgb"],
                width=DEFAULT_TOP_DOWN_CAMERA.resolution[0],
                height=DEFAULT_TOP_DOWN_CAMERA.resolution[1],
            )
            camera_sensor = Camera(cfg=camera_cfg)
            print(f"[VLA_V1] Camera sensor created: {camera_cfg.width}x{camera_cfg.height}")
            print(f"[VLA_V1] Attached to existing camera prim: {DEFAULT_TOP_DOWN_CAMERA.prim_path}")
        except Exception as create_err:
            print(f"[VLA_V1] ERROR: Failed to create Camera object: {create_err}")
            camera_sensor = None

    # Reset sim and robot
    def _reset_sim_and_robot() -> None:
        sim.reset()
        origin0 = torch.tensor(scene_origins[0], device=sim.device)
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += origin0
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()

    _reset_sim_and_robot()

    if camera_sensor is not None:
        try:
            camera_sensor.reset()
            print("[VLA_V1] Camera sensor reset OK (post sim.reset)")
        except Exception as e:
            print(f"[VLA_V1] ERROR: Camera sensor reset failed post sim.reset: {e}")
            camera_sensor = None

    # Controller
    ctrl_cfg = CartesianVelocityJogConfig(
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        device=str(sim.device),
        use_relative_mode=True,
        linear_speed_mps=float(getattr(args, "speed", 0.7)),
        workspace_min=(0.20, -0.45, 0.01),
        workspace_max=(0.6, 0.45, 0.35),
        log_ee_pos=bool(getattr(args, "print_ee", False)),
        log_ee_frame=str(getattr(args, "ee_frame", "world")),
        log_every_n_steps=int(getattr(args, "print_interval", 1)),
    )
    controller = CartesianVelocityJogController(ctrl_cfg, num_envs=1, device=str(sim.device))
    controller.set_mode("translate")
    controller.reset(robot)

    # Input/control
    mux_input = CommandMuxInputProvider()
    control_mode = str(getattr(args, "control", "keyboard"))
    if (not getattr(args, "headless", False)) and control_mode == "keyboard":
        from controllers.input.keyboard import Se3KeyboardInput

        keyboard = Se3KeyboardInput(
            pos_sensitivity_per_step=ctrl_cfg.linear_speed_mps * sim.get_physics_dt(),
            rot_sensitivity_rad_per_step=float(getattr(args, "rot_speed", 2.0)) * sim.get_physics_dt(),
        )
        mux_input.set_base(keyboard)
    elif control_mode == "planner":
        from pathlib import Path as _Path

        from controllers.input.waypoint_follower import WaypointFollowerInput
        from motion_generation.grasp_estimation.obb import ObbGraspPoseProvider
        from motion_generation.mogen import MotionGenerationAgent
        from motion_generation.planners import PlannerContext, create_planner
        from utilities import get_ee_pos_base_frame

        if loader is None or len(spawned_paths) == 0:
            print("[VLA_V1][PLANNER] ERROR: planner control requires spawned objects to grasp.")
            print("[VLA_V1][PLANNER] Ensure you run WITHOUT --no-objects and WITH --num-objects >= 1.")
            simulation_app.close()
            return 2

        cfg_dir = str((_Path(__file__).resolve().parents[2] / "motion_generation" / "planners" / "planners_config").resolve())
        planner = create_planner(
            str(getattr(args, "planner", "curobo_vla")),
            ctx=PlannerContext(
                base_frame="base_link",
                ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
                urdf_path=str((_Path(cfg_dir) / "cuRobo" / "kinovaJacoJ2N6S300.urdf").resolve()),
                config_dir=cfg_dir,
            ),
        )
        print(f"[VLA_V1][PLANNER] Control enabled. planner={getattr(args, 'planner', 'curobo_vla')} cfg_dir={cfg_dir}")
        grasp_provider = ObbGraspPoseProvider(align_to_min_width=True)

        robot_prim_path: Optional[str] = None
        try:
            robot_prim_path = str(getattr(getattr(robot, "cfg", None), "prim_path", None))
        except Exception:
            robot_prim_path = None

        agent = MotionGenerationAgent(
            sim=sim,
            robot=robot,
            controller=controller,
            planner=planner,
            grasp_provider=grasp_provider,
            loader=loader,  # type: ignore[name-defined]
            robot_prim_path=robot_prim_path,
        )

        dt_local = float(sim.get_physics_dt())
        wp = WaypointFollowerInput(
            step_pos_m=float(ctrl_cfg.linear_speed_mps) * dt_local,
            tol_m=float(getattr(args, "tolerance", 0.005)),
            device=str(sim.device),
        )
        mux_input.set_base(wp)

        # Cache arm joint ids/names for cuRobo start_state construction.
        _arm_joint_ids = None
        _arm_joint_names: list[str] = []
        try:
            arm_joint_ids, _ = robot.find_joints(controller.config.arm_joint_regex)
            # Normalize tensor/list to python ints
            if hasattr(arm_joint_ids, "view"):
                _arm_joint_ids = [int(v) for v in arm_joint_ids.view(-1).tolist()]
            else:
                _arm_joint_ids = [int(v) for v in list(arm_joint_ids)]
            _arm_joint_names = [str(robot.data.joint_names[i]) for i in _arm_joint_ids]
        except Exception:
            _arm_joint_ids = None
            _arm_joint_names = []

        vla_planner_state = {
            "stage": "init_open",  # init_open -> approach -> close -> lift -> done
            "target_prim": None,
            "lift_pt": None,
            "open_left": int(getattr(args, "gripper_open_steps", 10)),
            "close_left": int(getattr(args, "gripper_close_steps", 60)),
            "stabilize_left": int(getattr(args, "stabilize_steps", 300)),
        }
    controller.set_input_provider(mux_input)

    # Tracker + logger
    tracker = ObjectsTracker(prim_paths=spawned_paths)
    tick_cfg = TickLoggingConfig(
        log_rate_hz=int(getattr(args, "log_rate_hz", 10)),
        workspace_min=getattr(controller.config.safety_cfg, "workspace_min", None),
        workspace_max=getattr(controller.config.safety_cfg, "workspace_max", None),
        ee_link_name=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        arm_joint_regex=controller.config.arm_joint_regex,
    )
    session_logger = SessionLogWriter(root=Path(str(getattr(args, "logs_root", "logs/data_collection"))))

    images_dir = session_logger.root / "images"
    images_dir.mkdir(exist_ok=True)
    image_format = getattr(args, "image_format", "png")

    session_logger.write_metadata(
        sim_dt=sim.get_physics_dt(),
        physics_substeps=int(getattr(sim.cfg, "sub_steps", 4)),
        seed=0,
        robot_name="kinova_j2n6s300",
        ee_link=str(getattr(args, "ee_link", "j2n6s300_end_effector")),
        arm_joint_regex=controller.config.arm_joint_regex,
        log_rate_hz=tick_cfg.log_rate_hz,
        window_len_s=2.0,
    )

    print(f"[VLA_V1] Data collection started!")
    print(f"[VLA_V1] Session directory: {session_logger.root}")
    print(f"[VLA_V1] Images directory: {images_dir}")
    print(f"[VLA_V1] Logging rate: {tick_cfg.log_rate_hz} Hz")
    print(f"[VLA_V1] Duration: {getattr(args, 'duration_s', 30.0)} seconds")
    print(f"[VLA_V1] Control mode: {getattr(args, 'control', 'keyboard')}")
    print(f"[VLA_V1] Camera sensor: {'Active' if camera_sensor is not None else 'Inactive'}")

    # Run (episodes)
    dt = float(sim.get_physics_dt())
    period = 1.0 / float(tick_cfg.log_rate_hz)
    t0 = time.time()
    images_captured = 0
    num_episodes = int(getattr(args, "num_episodes", 10))
    max_steps_ep = int(getattr(args, "max_steps_per_episode", 3000))
    render_rate_hz = float(getattr(args, "render_rate_hz", 60.0))
    render_stride = max(1, int(round((1.0 / max(1e-9, render_rate_hz)) / max(1e-9, dt))))

    for ep in range(num_episodes):
        if not simulation_app.is_running():
            break

        # Reset sim/robot for a clean episode start.
        _reset_sim_and_robot()
        if camera_sensor is not None:
            try:
                camera_sensor.reset()
            except Exception:
                pass

        # Re-randomize object poses each episode (no destroy/recreate).
        if bool(getattr(args, "respawn_each_episode", False)):
            _rerandomize_object_poses(spawned_paths)

        # Re-init per-episode follower state to avoid leftover queued waypoints/gripper commands.
        if control_mode == "planner":
            wp.set_waypoints_b([])
            try:
                setattr(wp, "_gripper_queue", [])  # type: ignore[attr-defined]
            except Exception:
                pass
            vla_planner_state.update(
                {
                    "stage": "init_open",
                    "target_prim": None,
                    "lift_pt": None,
                    "open_left": int(getattr(args, "gripper_open_steps", 10)),
                    "close_left": int(getattr(args, "gripper_close_steps", 60)),
                    "stabilize_left": int(getattr(args, "stabilize_steps", 300)),
                    "open_queued": False,
                    "close_queued": False,
                    "lift_queued": False,
                }
            )
            session_logger.log_event("episode_start", {"episode_idx": int(ep), "num_objects": int(len(spawned_paths))})

        accum = 0.0
        steps = 0
        while simulation_app.is_running() and steps < max_steps_ep:
            steps += 1

            # Planner-driven reach-to-grasp state machine (optional)
            if control_mode == "planner":
                try:
                    wp.set_current_pose_b(get_ee_pos_base_frame(robot, str(getattr(args, "ee_link", "j2n6s300_end_effector"))))
                except Exception:
                    pass

                try:
                    stab_left = int(vla_planner_state.get("stabilize_left", 0))
                    if stab_left > 0:
                        vla_planner_state["stabilize_left"] = stab_left - 1
                    else:
                        stage = str(vla_planner_state.get("stage", "idle"))

                        if stage == "init_open":
                            open_left = int(vla_planner_state.get("open_left", 0))
                            if not vla_planner_state.get("open_queued", False):
                                session_logger.log_event("action_start", {"action": "GRIPPER_OPEN", "steps": int(open_left), "episode_idx": int(ep)})
                                try:
                                    controller.set_mode("gripper")
                                    wp.queue_gripper(+1.0, steps=open_left)
                                except Exception:
                                    pass
                                vla_planner_state["open_queued"] = True
                            controller.set_mode("gripper")
                            if open_left > 0:
                                vla_planner_state["open_left"] = open_left - 1
                            else:
                                session_logger.log_event("action_end", {"action": "GRIPPER_OPEN", "episode_idx": int(ep)})
                                vla_planner_state["stage"] = "idle"

                        if str(vla_planner_state.get("stage", "idle")) == "idle":
                            if len(spawned_paths) != 0:
                                target_label = getattr(args, "target_label", None)
                                target_prim, _pos_w, _quat_wxyz_w, pos_b, quat_b = agent.compute_current_grasp_for_label(
                                    label=target_label,
                                    prim_paths=spawned_paths,
                                )

                                # Provide cuRobo with a proper start_state (required by some MotionGen builds).
                                try:
                                    if hasattr(planner, "set_start_state") and _arm_joint_ids is not None and len(_arm_joint_ids) > 0:
                                        q = robot.data.joint_pos[0, _arm_joint_ids].detach().to("cpu").tolist()
                                        planner.set_start_state(joint_pos=[float(v) for v in q], joint_names=_arm_joint_names)
                                except Exception:
                                    pass

                                # Sync cuRobo world from scene before planning.
                                if bool(getattr(args, "curobo_world_from_scene", False)):
                                    obstacle_prims = list(spawned_paths)
                                    if bool(getattr(args, "curobo_world_include_table", True)):
                                        obstacle_prims = ["/World/Origin1/Table"] + obstacle_prims
                                    ok = False
                                    try:
                                        if hasattr(planner, "update_world_from_prim_paths"):
                                            ok = bool(
                                                planner.update_world_from_prim_paths(sim=sim, robot=robot, prim_paths=obstacle_prims)
                                            )
                                    except Exception:
                                        ok = False
                                    session_logger.log_event(
                                        "action_start",
                                        {"action": "CUROBO_UPDATE_WORLD", "ok": bool(ok), "n_prims": int(len(obstacle_prims)), "episode_idx": int(ep)},
                                    )
                                if ep == 0 and steps < 5:
                                    print(f"[VLA_V1][CUROBO] update_world_from_scene ok={ok} n_prims={len(obstacle_prims)}")

                                session_logger.log_event(
                                    "action_start",
                                    {"action": "PLAN_TO_PREGRASP", "planner": str(getattr(args, "planner", "curobo_vla")), "target_prim": str(target_prim), "episode_idx": int(ep)},
                                )
                                waypoints = planner.plan_to_pose_b(
                                    target_pos_b=pos_b,
                                    target_quat_b_wxyz=None if bool(getattr(args, "ignore_grasp_orientation", True)) else quat_b,
                                    pregrasp_offset_m=float(getattr(args, "pregrasp", 0.10)),
                                    grasp_depth_m=float(getattr(args, "grasp_depth", -0.04)),
                                    lift_height_m=float(getattr(args, "lift", 0.15)),
                                )
                                session_logger.log_event("action_end", {"action": "PLAN_TO_PREGRASP", "n_waypoints": int(len(waypoints)), "episode_idx": int(ep)})

                                if len(waypoints) >= 2:
                                    lift_pt = waypoints[-1]
                                    approach_pts = waypoints[:-1]
                                else:
                                    lift_pt = (
                                        float(pos_b[0]),
                                        float(pos_b[1]),
                                        float(pos_b[2] + float(getattr(args, "lift", 0.15))),
                                    )
                                    approach_pts = waypoints

                                vla_planner_state["target_prim"] = target_prim
                                vla_planner_state["lift_pt"] = lift_pt
                                vla_planner_state["stage"] = "approach"
                                vla_planner_state["lift_queued"] = False

                                controller.set_mode("translate")
                                session_logger.log_event("action_start", {"action": "EXECUTE_WAYPOINTS", "n_waypoints": int(len(approach_pts)), "episode_idx": int(ep)})
                                wp.set_waypoints_b([(float(x), float(y), float(z)) for (x, y, z) in approach_pts])

                        if str(vla_planner_state.get("stage", "")) == "approach":
                            if len(getattr(wp, "_waypoints_b", [])) == 0:
                                session_logger.log_event("action_end", {"action": "EXECUTE_WAYPOINTS", "episode_idx": int(ep)})
                                vla_planner_state["stage"] = "close"
                                vla_planner_state["close_queued"] = False
                                vla_planner_state["close_left"] = int(getattr(args, "gripper_close_steps", 60))

                        if str(vla_planner_state.get("stage", "")) == "close":
                            close_left = int(vla_planner_state.get("close_left", 0))
                            if not vla_planner_state.get("close_queued", False):
                                session_logger.log_event("action_start", {"action": "GRIPPER_CLOSE", "steps": int(close_left), "episode_idx": int(ep)})
                                try:
                                    controller.set_mode("gripper")
                                    wp.queue_gripper(-1.0, steps=close_left)
                                except Exception:
                                    pass
                                vla_planner_state["close_queued"] = True
                            controller.set_mode("gripper")
                            if close_left > 0:
                                vla_planner_state["close_left"] = close_left - 1
                            else:
                                session_logger.log_event("action_end", {"action": "GRIPPER_CLOSE", "episode_idx": int(ep)})
                                vla_planner_state["stage"] = "lift"
                                vla_planner_state["lift_queued"] = False

                        if str(vla_planner_state.get("stage", "")) == "lift":
                            lift_pt = vla_planner_state.get("lift_pt", None)
                            if lift_pt is not None and not vla_planner_state.get("lift_queued", False):
                                controller.set_mode("translate")
                                session_logger.log_event(
                                    "action_start",
                                    {"action": "LIFT", "target": [float(lift_pt[0]), float(lift_pt[1]), float(lift_pt[2])], "episode_idx": int(ep)},
                                )
                                wp.set_waypoints_b([(float(lift_pt[0]), float(lift_pt[1]), float(lift_pt[2]))])
                                vla_planner_state["lift_queued"] = True
                            if lift_pt is not None and vla_planner_state.get("lift_queued", False) and len(getattr(wp, "_waypoints_b", [])) == 0:
                                session_logger.log_event("action_end", {"action": "LIFT", "episode_idx": int(ep)})
                                vla_planner_state["stage"] = "done"

                        if str(vla_planner_state.get("stage", "")) == "done":
                            session_logger.log_event("episode_end", {"episode_idx": int(ep), "steps": int(steps)})
                            break
                except Exception as e:
                    if session_logger.tick_idx < 3:
                        print(f"[VLA_V1][PLANNER][WARN] Planner control error: {e}")

            controller.step(robot, dt)

            # Rendering throttling:
            # - In interactive mode, rendering every physics step (e.g., 240Hz) can tank FPS and cause
            #   lots of GPU/CPU sync with low utilization.
            # - Always render on tick boundaries (camera/logging), otherwise render at render_rate_hz.
            do_tick = (accum + dt + 1e-9) >= period
            do_render = bool(do_tick) or (steps % render_stride == 0)
            sim.step(render=bool(do_render))
            robot.update(dt)

            if camera_sensor is not None:
                try:
                    if hasattr(camera_sensor, "update"):
                        camera_sensor.update(dt)
                except Exception:
                    pass

            accum += dt
            if accum + 1e-9 >= period:
                accum = 0.0
            objs_raw = []
            try:
                for o in tracker.snapshot():
                    lbl = id_to_label.get(o.id, o.label)
                    objs_raw.append(
                        {
                            "id": o.id,
                            "label": lbl,
                            "pose": {"position_m": list(o.pose.position_m), "orientation_wxyz": list(o.pose.orientation_wxyz)},
                            "confidence": o.confidence,
                        }
                    )
            except Exception:
                pass

            image_path = None
            if camera_sensor is not None:
                try:
                    cam_data = camera_sensor.data
                    rgb_data = None
                    if cam_data.output is not None:
                        rgb_data = cam_data.output.get("rgb")
                    if rgb_data is not None:
                        if len(rgb_data.shape) == 4:
                            rgb_np = rgb_data[0].cpu().numpy()
                        elif len(rgb_data.shape) == 3:
                            rgb_np = rgb_data.cpu().numpy()
                        else:
                            raise ValueError(f"Unexpected RGB data shape: {rgb_data.shape}")

                        if rgb_np.max() <= 1.0:
                            rgb_np = (rgb_np * 255).astype(np.uint8)
                        else:
                            rgb_np = rgb_np.astype(np.uint8)

                        image_filename = f"image_{session_logger.tick_idx:06d}.{image_format}"
                        out_path = images_dir / image_filename
                        try:
                            from PIL import Image

                            Image.fromarray(rgb_np).save(str(out_path))
                        except Exception:
                            try:
                                import cv2

                                if len(rgb_np.shape) == 3 and rgb_np.shape[2] == 3:
                                    rgb_np_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(str(out_path), rgb_np_bgr)
                                else:
                                    cv2.imwrite(str(out_path), rgb_np)
                            except Exception:
                                np.save(str(out_path).replace(f".{image_format}", ".npy"), rgb_np)
                                out_path = out_path.with_suffix(".npy")

                        image_path = f"images/{image_filename}"
                        images_captured += 1
                except Exception:
                    image_path = None

            session_logger.write_tick(
                robot=robot,
                controller=controller,
                objects=objs_raw,
                last_user_cmd=mux_input.last_cmd,
                cfg=tick_cfg,
                image_path=image_path,
            )
        # If we hit the max steps, still end the episode cleanly.
        if control_mode == "planner" and str(vla_planner_state.get("stage", "")) != "done":
            session_logger.log_event("episode_end", {"episode_idx": int(ep), "steps": int(steps), "truncated": True})

    print(f"\n[VLA_V1] Data collection completed!")
    print(f"[VLA_V1] Total ticks logged: {session_logger.tick_idx}")
    print(f"[VLA_V1] Total images captured: {images_captured}")

    simulation_app.close()
    try:
        session_logger.close()
    except Exception:
        pass
    return 0


PROFILE = ProfileSpec(
    name="vla_v1",
    add_cli_args=add_cli_args,
    run=run,
)


