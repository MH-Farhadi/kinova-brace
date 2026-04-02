"""ROS 1 node for real-time BRACE inference on Kinova Gen3.

Loads trained models (expert, belief, arbitration) and runs the full
BRACE pipeline at ~27 Hz. Subscribes to DualSense input and object
detections, publishes blended Cartesian velocity to the Kinova arm.

IMPORTANT: This requires ROS 1 (rospy), NOT ROS 2 (rclpy).
ROS dependencies are optional — this module can be imported for
testing without ROS installed (will raise at runtime if ROS missing).

Usage (via rosrun):
    rosrun brace_kinova brace_node.py _config_path:=configs/arbitration.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

try:
    import rospy
    from geometry_msgs.msg import Twist, TwistStamped, PoseArray, PoseStamped
    from std_msgs.msg import Float32, Float32MultiArray
    from sensor_msgs.msg import Joy

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from brace_kinova.models.bayesian_inference import BayesianGoalInference
from brace_kinova.models.expert_policy import ExpertPolicy, PotentialFieldExpert
from brace_kinova.models.arbitration_policy import action_to_gamma


class BraceNode:
    """ROS 1 node for real-time BRACE shared autonomy.

    Subscribes to:
        /joy                 — DualSense PS5 controller input
        /object_positions    — detected object poses (from YOLO pipeline)
        /obstacle_positions  — detected obstacle poses
        /ee_state            — current EE pose from kortex_driver

    Publishes:
        /my_gen3/in/cartesian_velocity  — blended Cartesian velocity command
        /brace/belief                   — current belief distribution
        /brace/gamma                    — current assistance level
    """

    INFERENCE_RATE_HZ = 27.0
    DEADZONE = 0.1
    MAX_VELOCITY = 0.15

    def __init__(self, config_path: str = "brace_kinova/configs/arbitration.yaml"):
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS 1 (rospy) is not installed. Cannot run BraceNode.")

        rospy.init_node("brace_node")
        self.config_path = config_path

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        device_str = rospy.get_param("~device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"[BRACE] Using device: {self.device}")

        self._load_models()
        self._init_state()
        self._setup_ros()

    def _load_models(self) -> None:
        bayes_cfg = self.config.get("bayesian", {})
        n_goals = bayes_cfg.get("n_goals", 3)

        self.belief_module = BayesianGoalInference(
            n_goals=n_goals,
            initial_beta=bayes_cfg.get("initial_beta", 2.0),
            initial_w_theta=bayes_cfg.get("initial_w_theta", 0.8),
            initial_w_dist=bayes_cfg.get("initial_w_dist", 0.2),
        ).to(self.device)
        self.belief_module.eval()

        belief_path = self.config.get("training", {}).get(
            "save_dir", "./checkpoints"
        ) + "/bayesian_inference_finetuned.pt"
        if Path(belief_path).exists():
            self.belief_module.load_state_dict(
                torch.load(belief_path, map_location=self.device)
            )
            rospy.loginfo(f"[BRACE] Loaded belief from {belief_path}")

        policy_path = self.config.get("training", {}).get(
            "save_dir", "./checkpoints"
        ) + "/arbitration_policy.zip"
        if Path(policy_path).exists():
            from stable_baselines3 import PPO
            self.arbitration_model = PPO.load(policy_path, device=self.device)
            rospy.loginfo(f"[BRACE] Loaded policy from {policy_path}")
        else:
            self.arbitration_model = None
            rospy.logwarn("[BRACE] No arbitration policy found, using fixed gamma=0.5")

        expert_path = self.config.get("expert_path", "checkpoints/expert_sac.zip")
        if Path(expert_path).exists():
            self.expert = ExpertPolicy(expert_path, device=str(self.device))
        else:
            self.expert = PotentialFieldExpert()
            rospy.logwarn("[BRACE] Using PotentialFieldExpert fallback")

        self.n_goals = n_goals

    def _init_state(self) -> None:
        self.belief = np.ones(self.n_goals, dtype=np.float32) / self.n_goals
        self.ee_pos = np.zeros(2, dtype=np.float32)
        self.ee_vel = np.zeros(2, dtype=np.float32)
        self.human_input = np.zeros(2, dtype=np.float32)
        self.object_positions = np.zeros((self.n_goals, 2), dtype=np.float32)
        self.obstacle_positions = np.zeros((0, 2), dtype=np.float32)
        self.gamma = 0.5
        self.gamma_mode = "ai"

    def _setup_ros(self) -> None:
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback, queue_size=1)
        self.objects_sub = rospy.Subscriber(
            "/object_positions", PoseArray, self.objects_callback, queue_size=1
        )
        self.obstacles_sub = rospy.Subscriber(
            "/obstacle_positions", PoseArray, self.obstacles_callback, queue_size=1
        )
        self.ee_sub = rospy.Subscriber(
            "/ee_state", PoseStamped, self.ee_callback, queue_size=1
        )

        self.cmd_pub = rospy.Publisher(
            "/my_gen3/in/cartesian_velocity", TwistStamped, queue_size=1
        )
        self.belief_pub = rospy.Publisher(
            "/brace/belief", Float32MultiArray, queue_size=1
        )
        self.gamma_pub = rospy.Publisher(
            "/brace/gamma", Float32, queue_size=1
        )

        rate = self.INFERENCE_RATE_HZ
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / rate), self.inference_callback
        )
        rospy.loginfo(f"[BRACE] Running at {rate} Hz")

    def joy_callback(self, msg: Joy) -> None:
        """Parse DualSense PS5 controller input from /joy topic."""
        if len(msg.axes) < 2:
            return

        lx = msg.axes[0]
        ly = msg.axes[1]

        if abs(lx) < self.DEADZONE and abs(ly) < self.DEADZONE:
            self.human_input[:] = 0.0
        else:
            self.human_input[0] = lx * self.MAX_VELOCITY
            self.human_input[1] = ly * self.MAX_VELOCITY

        if self.gamma_mode == "manual" and len(msg.axes) >= 6:
            l2 = msg.axes[4]
            r2 = msg.axes[5]
            if l2 > 0.1:
                self.gamma = max(0.0, self.gamma - 0.003)
            if r2 > 0.1:
                self.gamma = min(1.0, self.gamma + 0.003)

    def objects_callback(self, msg: PoseArray) -> None:
        poses = []
        for pose in msg.poses[:self.n_goals]:
            poses.append([pose.position.x, pose.position.y])
        if poses:
            self.object_positions = np.array(poses, dtype=np.float32)

    def obstacles_callback(self, msg: PoseArray) -> None:
        poses = []
        for pose in msg.poses:
            poses.append([pose.position.x, pose.position.y])
        self.obstacle_positions = np.array(poses, dtype=np.float32) if poses else np.zeros((0, 2))

    def ee_callback(self, msg: PoseStamped) -> None:
        self.ee_pos[0] = msg.pose.position.x
        self.ee_pos[1] = msg.pose.position.y

    def inference_callback(self, timer_event) -> None:
        """Main BRACE inference loop — runs at ~27 Hz."""
        h_input = self.human_input.copy()

        if np.linalg.norm(h_input) > 1e-6 and self.belief_module is not None:
            with torch.no_grad():
                t_h = torch.from_numpy(h_input).unsqueeze(0).to(self.device)
                t_ee = torch.from_numpy(self.ee_pos).unsqueeze(0).to(self.device)
                t_goals = torch.from_numpy(self.object_positions).unsqueeze(0).to(self.device)
                t_prior = torch.from_numpy(self.belief).unsqueeze(0).to(self.device)

                t_belief, _ = self.belief_module(
                    t_h, t_ee, t_goals, t_prior
                )
                self.belief = t_belief.squeeze(0).cpu().numpy()

        if self.gamma_mode == "ai" and self.arbitration_model is not None:
            obs = self._build_observation()
            fused = np.concatenate([obs, self.belief]).astype(np.float32)
            action, _ = self.arbitration_model.predict(fused, deterministic=True)
            self.gamma = action_to_gamma(float(action[0]))

        best_goal_idx = int(np.argmax(self.belief))
        best_goal = self.object_positions[best_goal_idx]

        if isinstance(self.expert, PotentialFieldExpert):
            w_action = self.expert.predict(
                self.ee_pos.copy(), best_goal, self.obstacle_positions
            )
        else:
            expert_obs = self._build_expert_obs(best_goal_idx)
            w_action = self.expert.predict(expert_obs)

        w_vel = w_action[:2] * self.MAX_VELOCITY

        blended_vx = (1.0 - self.gamma) * h_input[0] + self.gamma * w_vel[0]
        blended_vy = (1.0 - self.gamma) * h_input[1] + self.gamma * w_vel[1]

        cmd = TwistStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.twist.linear.x = float(blended_vx)
        cmd.twist.linear.y = float(blended_vy)
        cmd.twist.linear.z = 0.0
        self.cmd_pub.publish(cmd)

        belief_msg = Float32MultiArray()
        belief_msg.data = self.belief.tolist()
        self.belief_pub.publish(belief_msg)

        gamma_msg = Float32()
        gamma_msg.data = self.gamma
        self.gamma_pub.publish(gamma_msg)

    def _build_observation(self) -> np.ndarray:
        """Build the state observation vector (excluding belief)."""
        rel_objects = self.object_positions - self.ee_pos[np.newaxis, :]
        parts = [
            self.ee_pos,
            self.ee_vel,
            np.array([0.0], dtype=np.float32),
            rel_objects.flatten(),
        ]
        if self.obstacle_positions.shape[0] > 0:
            rel_obs = self.obstacle_positions - self.ee_pos[np.newaxis, :]
            obs_dists = np.linalg.norm(rel_obs, axis=1)
            parts.append(rel_obs.flatten())
            parts.append(np.array([obs_dists.min()], dtype=np.float32))
        else:
            parts.append(np.array([1.0], dtype=np.float32))

        dists = np.linalg.norm(self.object_positions - self.ee_pos[np.newaxis, :], axis=1)
        progress = 1.0 - np.clip(dists / 0.6, 0, 1)
        parts.append(progress.astype(np.float32))

        return np.concatenate(parts)

    def _build_expert_obs(self, goal_idx: int) -> np.ndarray:
        base_obs = self._build_observation()
        one_hot = np.zeros(self.n_goals, dtype=np.float32)
        one_hot[goal_idx] = 1.0
        return np.concatenate([base_obs, one_hot])

    def run(self) -> None:
        rospy.loginfo("[BRACE] Node is running...")
        rospy.spin()


def main():
    config_path = "brace_kinova/configs/arbitration.yaml"
    if ROS_AVAILABLE:
        config_path = rospy.get_param("~config_path", config_path)
    elif len(sys.argv) > 1:
        config_path = sys.argv[1]

    node = BraceNode(config_path)
    node.run()


if __name__ == "__main__":
    main()
