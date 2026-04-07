"""Isaac Lab Gymnasium environment for 2D planar BRACE training.

Wraps the Kinova Jaco2 (J2N6S300) robot in Isaac Sim with coloured goal
and obstacle cuboids on a tabletop.  All policy-facing observations and
actions are 2D (XY plane), matching the lightweight ``ReachGraspEnv`` so
that expert, belief, and arbitration models transfer without changes.

IMPORTANT
---------
This module imports Isaac Lab / Isaac Sim packages and **must** be loaded
*after* ``isaaclab.app.AppLauncher`` has been called.  The training
scripts (``train_isaac_expert.py``, ``train_isaac_arbitration.py``)
handle the launch sequence before importing this module.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# Isaac Lab imports --------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg as _CuboidCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import (
    PreviewSurfaceCfg as _PreviewSurfaceCfg,
)

# Default Jaco2 joint positions — arm forward, EE near table surface
_JACO2_DEFAULT_JOINTS = {
    "j2n6s300_joint_1": 0.0,
    "j2n6s300_joint_2": math.pi,
    "j2n6s300_joint_3": 1.8 * math.pi,
    "j2n6s300_joint_4": 0.0,
    "j2n6s300_joint_5": 1.75 * math.pi,
    "j2n6s300_joint_6": 0.5 * math.pi,
    "j2n6s300_joint_finger_1": 0.0,
    "j2n6s300_joint_finger_2": 0.0,
    "j2n6s300_joint_finger_3": 0.0,
    "j2n6s300_joint_finger_tip_1": 0.0,
    "j2n6s300_joint_finger_tip_2": 0.0,
    "j2n6s300_joint_finger_tip_3": 0.0,
}

# BRACE imports ------------------------------------------------------------
from brace_kinova.envs.isaac_config import IsaacBraceEnvConfig, IsaacBraceSceneConfig
from brace_kinova.envs.scenarios import ScenarioConfig, SCENARIOS


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def setup_brace_scene(
    scene_cfg: IsaacBraceSceneConfig,
) -> tuple[dict, list]:
    """Build the Isaac Sim scene for BRACE training.

    Creates ground plane, dome light, Thorlabs table, and Kinova Jaco2.
    Returns the scene entity dict and origin list (single origin).
    """
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets import KINOVA_JACO2_N6S300_CFG
    from dataclasses import replace as dc_replace

    prim_utils = importlib.import_module("isaacsim.core.utils.prims")

    # Ground + light
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # Table
    table_usd = scene_cfg.table_usd_path or (
        f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
    )
    table_cfg = sim_utils.UsdFileCfg(usd_path=table_usd, scale=scene_cfg.table_scale)
    table_cfg.func("/World/Origin1/Table", table_cfg, translation=scene_cfg.table_translation)

    # Robot
    robot_cfg = dc_replace(KINOVA_JACO2_N6S300_CFG, prim_path=scene_cfg.robot_prim_path)
    robot_cfg.init_state.pos = (0.0, 0.0, scene_cfg.robot_base_height)
    joint_pos = scene_cfg.robot_default_joint_pos or _JACO2_DEFAULT_JOINTS
    robot_cfg.init_state.joint_pos = joint_pos
    robot = Articulation(cfg=robot_cfg)

    entities = {"kinova_j2n6s300": robot}
    return entities, origins


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IsaacReachGraspEnv(gym.Env):
    """2D planar reach-and-grasp in Isaac Sim with Kinova Jaco2.

    Observation (continuous Box) — identical to ``ReachGraspEnv``::

        EE position XY (2) + EE velocity XY (2) + gripper (1)
        + relative XY to each object (2 × n_objects)
        + relative XY to each obstacle (2 × n_obstacles)
        + distance to nearest obstacle (1)
        + progress toward each object (n_objects)

    Action (continuous Box, shape=(3,))::

        vx, vy  – Cartesian velocity in [-1, 1], scaled by max_velocity
        gripper – < 0 open, >= 0 close
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        sim: Any,
        robot: Articulation,
        config: IsaacBraceEnvConfig,
        scenario: str | ScenarioConfig = "full_complexity",
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.sim = sim
        self.robot = robot
        self.cfg = config
        self.render_mode = render_mode
        self._device = str(sim.device)

        # Scenario
        if isinstance(scenario, str):
            self.scenario = SCENARIOS[scenario]
        else:
            self.scenario = scenario
        self.n_objects = self.scenario.n_objects
        self.n_obstacles = self.scenario.n_obstacles

        self.ws_diag = math.hypot(
            self.cfg.x_max - self.cfg.x_min,
            self.cfg.y_max - self.cfg.y_min,
        )

        # Spaces (match ReachGraspEnv)
        obs_dim = 5 + 2 * self.n_objects + 2 * self.n_obstacles + 1 + self.n_objects
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32,
        )

        self.np_random, _ = gym.utils.seeding.np_random(config.seed)

        # Controller setup
        self._setup_ik()

        # Visual markers for goals / obstacles
        self._goal_prim_paths: list[str] = []
        self._obstacle_prim_paths: list[str] = []
        self._spawn_visual_objects()

        # Mutable state
        self._step_count = 0
        self._ee_pos_xy = np.zeros(2, dtype=np.float32)
        self._ee_vel_xy = np.zeros(2, dtype=np.float32)
        self._prev_ee_pos_xy = np.zeros(2, dtype=np.float32)
        self._gripper_state = 0.0
        self._object_positions = np.zeros((self.n_objects, 2), dtype=np.float32)
        self._obstacle_positions = np.zeros((self.n_obstacles, 2), dtype=np.float32)
        self._true_goal_idx = 0
        self._prev_dist_to_goal = 0.0
        self._initial_dist_to_goal = 0.0
        self._prev_heading = 0.0
        self._collision_count = 0

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    def _setup_ik(self) -> None:
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method=self.cfg.ik_method,
        )
        self._ik = DifferentialIKController(ik_cfg, num_envs=1, device=self._device)
        self._ik.reset()

        body_ids, _ = self.robot.find_bodies([self.cfg.ee_link_name])
        self._ee_body_id = int(body_ids[0])
        self._ee_jacobi_idx = (
            self._ee_body_id - 1 if self.robot.is_fixed_base else self._ee_body_id
        )
        arm_ids, _ = self.robot.find_joints(self.cfg.arm_joint_regex)
        self._arm_joint_ids = arm_ids

        gripper_ids, _ = self.robot.find_joints(self.cfg.gripper_joint_regex)
        self._gripper_joint_ids = gripper_ids

    def _read_ee_pose_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pos_b (1,3), quat_b (1,4)) of the EE in robot base frame."""
        root_w = self.robot.data.root_pose_w
        ee_w = self.robot.data.body_pose_w[:, self._ee_body_id]
        return subtract_frame_transforms(
            root_w[:, 0:3], root_w[:, 3:7],
            ee_w[:, 0:3], ee_w[:, 3:7],
        )

    # ------------------------------------------------------------------
    # Scene objects
    # ------------------------------------------------------------------

    def _spawn_visual_objects(self) -> None:
        """Create coloured cuboids as visual markers for goals and obstacles."""
        prim_utils = importlib.import_module("isaacsim.core.utils.prims")
        root = self.cfg.scene.objects_prim_root

        if not _prim_exists(root):
            prim_utils.create_prim(root, "Xform")

        sc = self.cfg.scene
        for i in range(self.cfg.max_n_objects):
            path = f"{root}/Goal_{i:02d}"
            if _prim_exists(path):
                self._goal_prim_paths.append(path)
                continue
            cfg = _CuboidCfg(
                size=sc.goal_box_size,
                visual_material=_PreviewSurfaceCfg(diffuse_color=sc.goal_color),
            )
            cfg.func(path, cfg, translation=(0.0, 0.0, -5.0))
            self._goal_prim_paths.append(path)

        for i in range(self.cfg.max_n_obstacles):
            path = f"{root}/Obstacle_{i:02d}"
            if _prim_exists(path):
                self._obstacle_prim_paths.append(path)
                continue
            cfg = _CuboidCfg(
                size=sc.obstacle_box_size,
                visual_material=_PreviewSurfaceCfg(diffuse_color=sc.obstacle_color),
            )
            cfg.func(path, cfg, translation=(0.0, 0.0, -5.0))
            self._obstacle_prim_paths.append(path)

    def _set_prim_translation(self, prim_path: str, xyz: tuple) -> None:
        """Teleport a visual prim to *xyz* in world frame via USD Xform ops."""
        import omni.usd  # noqa: F811 – deferred import
        from pxr import UsdGeom, Gf

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    def _base_xy_to_world(self, xy: np.ndarray) -> tuple:
        """Convert base-frame XY to a world-frame (x, y, z) tuple."""
        root_w = self.robot.data.root_pose_w[0].cpu()
        wx = float(root_w[0]) + float(xy[0])
        wy = float(root_w[1]) + float(xy[1])
        wz = self.cfg.scene.object_spawn_z
        return (wx, wy, wz)

    # ------------------------------------------------------------------
    # Random placement
    # ------------------------------------------------------------------

    def _sample_positions(
        self, n: int, existing: list[np.ndarray], min_sep: float,
    ) -> np.ndarray:
        positions: list[np.ndarray] = []
        x_lo, x_hi = self.cfg.object_x_range
        y_lo, y_hi = self.cfg.object_y_range
        for _ in range(n):
            for _attempt in range(500):
                pos = np.array(
                    [self.np_random.uniform(x_lo, x_hi),
                     self.np_random.uniform(y_lo, y_hi)],
                    dtype=np.float32,
                )
                if all(np.linalg.norm(pos - o) >= min_sep for o in existing + positions):
                    positions.append(pos)
                    break
            else:
                positions.append(np.array(
                    [self.np_random.uniform(x_lo, x_hi),
                     self.np_random.uniform(y_lo, y_hi)],
                    dtype=np.float32,
                ))
        return np.array(positions, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    @property
    def true_goal_position(self) -> np.ndarray:
        return self._object_positions[self._true_goal_idx]

    @property
    def n_goals(self) -> int:
        return self.n_objects

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self._collision_count = 0

        # Reset robot joints to default configuration
        self._reset_robot()

        # Stabilise the robot for a few physics steps (holds joints)
        self._stabilize(self.cfg.stabilize_steps)

        # Capture orientation to hold constant during planar motion
        pos_b, quat_b = self._read_ee_pose_base()
        self._hold_quat = quat_b.clone()

        # Drive EE down to the planar working height before the episode starts
        self._home_to_start(n_steps=120)

        # Re-read actual EE state after homing
        pos_b, _ = self._read_ee_pose_base()
        ee_xy = pos_b[0, :2].cpu().numpy().astype(np.float32)
        self._ee_pos_xy = ee_xy.copy()
        self._prev_ee_pos_xy = ee_xy.copy()
        self._ee_vel_xy = np.zeros(2, dtype=np.float32)
        self._gripper_state = 0.0

        # Random object / obstacle placement
        ee_list = [self._ee_pos_xy.copy()]
        self._object_positions = self._sample_positions(
            self.n_objects, ee_list, self.cfg.min_object_separation,
        )
        if self.n_obstacles > 0:
            existing = ee_list + list(self._object_positions)
            self._obstacle_positions = self._sample_positions(
                self.n_obstacles, existing, self.cfg.min_obstacle_object_separation,
            )
        else:
            self._obstacle_positions = np.zeros((0, 2), dtype=np.float32)

        self._true_goal_idx = int(self.np_random.integers(0, self.n_objects))
        self._prev_dist_to_goal = float(
            np.linalg.norm(self._ee_pos_xy - self.true_goal_position)
        )
        self._initial_dist_to_goal = self._prev_dist_to_goal
        self._prev_heading = 0.0

        # Move visual cuboids
        self._place_visual_objects()

        obs = self._get_obs()
        info = {
            "true_goal_idx": self._true_goal_idx,
            "object_positions": self._object_positions.copy(),
            "obstacle_positions": self._obstacle_positions.copy(),
            "n_objects": self.n_objects,
            "n_obstacles": self.n_obstacles,
        }
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0)
        vx = float(action[0]) * self.cfg.max_velocity
        vy = float(action[1]) * self.cfg.max_velocity
        gripper_close = float(action[2]) >= 0.0

        # Compute target EE position for this control step
        pos_b, _ = self._read_ee_pose_base()
        cur = pos_b[0].cpu()
        target_x = float(cur[0]) + vx * self.cfg.control_dt
        target_y = float(cur[1]) + vy * self.cfg.control_dt
        target_x = np.clip(target_x, self.cfg.x_min, self.cfg.x_max)
        target_y = np.clip(target_y, self.cfg.y_min, self.cfg.y_max)
        target_z = self.cfg.z_fixed

        target_pos = torch.tensor(
            [[target_x, target_y, target_z]], dtype=torch.float32, device=self._device,
        )

        # Track the target over several physics sub-steps
        for _ in range(self.cfg.decimation):
            self._ik_substep(target_pos, gripper_close)

        # Read resulting state
        pos_b_new, _ = self._read_ee_pose_base()
        new_xy = pos_b_new[0, :2].cpu().numpy().astype(np.float32)
        self._ee_vel_xy = (new_xy - self._prev_ee_pos_xy) / self.cfg.control_dt
        self._prev_ee_pos_xy = self._ee_pos_xy.copy()
        self._ee_pos_xy = new_xy
        self._gripper_state = 1.0 if gripper_close else 0.0
        self._step_count += 1

        # Termination checks
        terminated = False
        truncated = False
        reward = 0.0

        collision = self._check_collision()
        if collision:
            self._collision_count += 1
            terminated = True
            reward = -self.cfg.reward_collision_penalty

        grasped = self._check_grasp()
        if grasped and not terminated:
            terminated = True
            reward = self.cfg.reward_goal_bonus

        if not terminated:
            reward = self._compute_reward(vx, vy)

        if self._step_count >= self.cfg.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "collision": collision,
            "grasped": grasped,
            "true_goal_idx": self._true_goal_idx,
            "dist_to_goal": float(np.linalg.norm(self._ee_pos_xy - self.true_goal_position)),
            "collision_count": self._collision_count,
            "step_count": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        rel_objects = self._object_positions - self._ee_pos_xy[np.newaxis, :]
        parts: list[np.ndarray] = [
            self._ee_pos_xy,
            self._ee_vel_xy,
            np.array([self._gripper_state], dtype=np.float32),
            rel_objects.flatten(),
        ]
        if self.n_obstacles > 0:
            rel_obs = self._obstacle_positions - self._ee_pos_xy[np.newaxis, :]
            obs_dists = np.linalg.norm(rel_obs, axis=1)
            min_obs_dist = float(obs_dists.min())
            parts.append(rel_obs.flatten())
        else:
            min_obs_dist = 1.0

        parts.append(np.array([min_obs_dist], dtype=np.float32))
        dists = np.linalg.norm(
            self._object_positions - self._ee_pos_xy[np.newaxis, :], axis=1,
        )
        progress = 1.0 - np.clip(dists / max(self.ws_diag, 1e-6), 0.0, 1.0)
        parts.append(progress.astype(np.float32))
        return np.concatenate(parts)

    def get_expert_obs(self) -> np.ndarray:
        """Observation with appended one-hot goal (for expert training)."""
        base = self._get_obs()
        onehot = np.zeros(self.n_objects, dtype=np.float32)
        onehot[self._true_goal_idx] = 1.0
        return np.concatenate([base, onehot])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, vx: float, vy: float) -> float:
        d_goal = float(np.linalg.norm(self._ee_pos_xy - self.true_goal_position))
        d_max = max(self._initial_dist_to_goal, 1e-6)

        heading = math.atan2(vy, vx) if (abs(vx) + abs(vy)) > 1e-6 else self._prev_heading
        delta_heading = heading - self._prev_heading

        if self.n_obstacles > 0:
            obs_dists = np.linalg.norm(
                self._obstacle_positions - self._ee_pos_xy[np.newaxis, :], axis=1,
            )
            min_obs_dist = float(obs_dists.min())
        else:
            min_obs_dist = 1.0

        reward = (
            self.cfg.reward_progress_weight * (self._prev_dist_to_goal - d_goal) / d_max
            - self.cfg.reward_heading_penalty * delta_heading ** 2
            - self.cfg.reward_obstacle_penalty * math.exp(-min_obs_dist / self.cfg.reward_d_safe)
        )
        self._prev_dist_to_goal = d_goal
        self._prev_heading = heading
        return reward

    # ------------------------------------------------------------------
    # Collision / grasp
    # ------------------------------------------------------------------

    def _check_collision(self) -> bool:
        if self.n_obstacles == 0:
            return False
        dists = np.linalg.norm(
            self._obstacle_positions - self._ee_pos_xy[np.newaxis, :], axis=1,
        )
        return bool(np.any(dists < self.cfg.collision_threshold))

    def _check_grasp(self) -> bool:
        dist = np.linalg.norm(self._ee_pos_xy - self.true_goal_position)
        return bool(dist < self.cfg.grasp_threshold and self._gripper_state > 0.5)

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _ik_substep(self, target_pos: torch.Tensor, gripper_close: bool) -> None:
        """Run one physics substep: IK → joint targets → sim.step."""
        pos_b, quat_b = self._read_ee_pose_base()

        # Jacobian
        jac = self.robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, self._arm_joint_ids
        ]
        q_arm = self.robot.data.joint_pos[:, self._arm_joint_ids]

        # Desired pose
        self._ik.ee_pos_des[:] = target_pos
        self._ik.ee_quat_des[:] = self._hold_quat

        q_des = self._ik.compute(pos_b, quat_b, jac, q_arm)
        qdot = (q_des - q_arm) / self.cfg.physics_dt

        # Arm velocity
        self.robot.set_joint_position_target(self.robot.data.joint_pos)
        self.robot.set_joint_velocity_target(torch.zeros_like(self.robot.data.joint_vel))
        self.robot.set_joint_velocity_target(qdot, joint_ids=self._arm_joint_ids)

        # Gripper position
        grip_val = self.cfg.gripper_close_pos if gripper_close else self.cfg.gripper_open_pos
        grip_target = torch.full(
            (1, len(self._gripper_joint_ids)),
            grip_val,
            device=self._device,
            dtype=torch.float32,
        )
        self.robot.set_joint_position_target(grip_target, joint_ids=self._gripper_joint_ids)

        # Gravity compensation
        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()
        self.robot.set_joint_effort_target(gravity)

        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.cfg.physics_dt)

    def _reset_robot(self) -> None:
        """Teleport robot back to its default joint configuration."""
        origin = torch.tensor([0.0, 0.0, 0.0], device=self._device)
        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] += origin
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos,
            self.robot.data.default_joint_vel,
        )
        self.robot.reset()

    def _stabilize(self, steps: int) -> None:
        """Hold the robot at its current joint positions for *steps* physics steps."""
        target = self.robot.data.joint_pos.clone()
        for _ in range(steps):
            self.robot.set_joint_position_target(target)
            self.robot.set_joint_velocity_target(torch.zeros_like(self.robot.data.joint_vel))
            gravity = self.robot.root_physx_view.get_gravity_compensation_forces()
            self.robot.set_joint_effort_target(gravity)
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)

    def _home_to_start(self, n_steps: int = 120) -> None:
        """Gradually move the EE to the planar start pose via IK.

        This prevents a violent snap from the default joint config
        (which may have the EE high up) to the z_fixed working plane.
        """
        target = torch.tensor(
            [[self.cfg.ee_initial_x, self.cfg.ee_initial_y, self.cfg.z_fixed]],
            dtype=torch.float32, device=self._device,
        )
        for _ in range(n_steps):
            self._ik_substep(target, gripper_close=False)

    def _place_visual_objects(self) -> None:
        """Move goal / obstacle cuboids to their sampled XY positions."""
        hidden = (0.0, 0.0, -5.0)
        for i in range(self.cfg.max_n_objects):
            if i < self.n_objects:
                self._set_prim_translation(
                    self._goal_prim_paths[i],
                    self._base_xy_to_world(self._object_positions[i]),
                )
            else:
                self._set_prim_translation(self._goal_prim_paths[i], hidden)

        for i in range(self.cfg.max_n_obstacles):
            if i < self.n_obstacles:
                self._set_prim_translation(
                    self._obstacle_prim_paths[i],
                    self._base_xy_to_world(self._obstacle_positions[i]),
                )
            else:
                self._set_prim_translation(self._obstacle_prim_paths[i], hidden)

    # ------------------------------------------------------------------
    # Scenario management
    # ------------------------------------------------------------------

    def set_scenario(self, scenario: str | ScenarioConfig) -> None:
        """Switch scenario (used by the curriculum manager)."""
        if isinstance(scenario, str):
            self.scenario = SCENARIOS[scenario]
        else:
            self.scenario = scenario
        self.n_objects = self.scenario.n_objects
        self.n_obstacles = self.scenario.n_obstacles
        obs_dim = 5 + 2 * self.n_objects + 2 * self.n_obstacles + 1 + self.n_objects
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

    def render(self):
        return None

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Expert variant
# ---------------------------------------------------------------------------

class IsaacExpertReachGraspEnv(IsaacReachGraspEnv):
    """Variant that appends a one-hot true-goal vector to observations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        expert_dim = self.observation_space.shape[0] + self.n_objects
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(expert_dim,), dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        # Must match ``ExpertReachGraspEnv``: base features + one-hot, not
        # ``get_expert_obs()`` (which calls ``_get_obs()`` → infinite recursion).
        base = super()._get_obs()
        goal_one_hot = np.zeros(self.n_objects, dtype=np.float32)
        goal_one_hot[self._true_goal_idx] = 1.0
        return np.concatenate([base, goal_one_hot])

    def get_expert_obs(self) -> np.ndarray:
        """Same as policy observation (one-hot already in ``_get_obs``)."""
        return self._get_obs()

    def set_scenario(self, scenario: str | ScenarioConfig) -> None:
        super().set_scenario(scenario)
        base_dim = 5 + 2 * self.n_objects + 2 * self.n_obstacles + 1 + self.n_objects
        expert_dim = base_dim + self.n_objects
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(expert_dim,), dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _prim_exists(prim_path: str) -> bool:
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        return stage.GetPrimAtPath(prim_path).IsValid()
    except Exception:
        return False
