# BRACE Kinova

This repository combines two related codebases for shared-autonomy research with a Kinova robot:

1. **BRACE** (**Bayesian Reinforcement Assistance with Context Encoding**) — a modular Python stack for training and deploying assistance-arbitration policies on a 2D planar reach-and-grasp task, under `brace_kinova/`.
2. **Kinova Isaac simulation** — Isaac Lab demos, controllers, motion generation, and data collection for a Kinova Jaco2 (J2N6S300), under `KINOVA_CODEBASE/`.

---

## Quick Start

### 1. Install BRACE dependencies

```bash
pip install -r brace_kinova/requirements.txt
```

### 2. Train/Resume on lightweight 2D environment (no Isaac Sim needed)

```bash
# Expert (SAC) - start
python -m brace_kinova.training.train_expert \
  --config brace_kinova/configs/expert.yaml

# Expert (SAC) - resume from checkpoint (+ optional replay buffer)
python -m brace_kinova.training.train_expert \
  --config brace_kinova/configs/expert.yaml \
  --resume_model checkpoints/expert_sac_time_<TIMESTAMP>_<STEPS>.zip \
  --resume_replay_buffer checkpoints/expert_sac_time_<TIMESTAMP>_<STEPS>.replay_buffer.pkl

# Belief pretraining - start
python -m brace_kinova.training.train_belief \
  --config brace_kinova/configs/belief.yaml

# Belief pretraining - resume
python -m brace_kinova.training.train_belief \
  --config brace_kinova/configs/belief.yaml \
  --resume_checkpoint checkpoints/bayesian_inference_time_<TIMESTAMP>_epoch<NUM>.ckpt.pt

# Arbitration (PPO + curriculum) - start
python -m brace_kinova.training.train_arbitration \
  --config brace_kinova/configs/arbitration.yaml

# Arbitration (PPO + curriculum) - resume policy
python -m brace_kinova.training.train_arbitration \
  --config brace_kinova/configs/arbitration.yaml \
  --resume_model checkpoints/arbitration_policy_time_<TIMESTAMP>_<STEPS>.zip
```

### 3. Train/Resume on Isaac Sim with the Kinova Jaco2

Requires Isaac Sim 4.5+ and Isaac Lab. Run via the Isaac Lab launcher (replace path as needed):

```bash
# Expert (SAC) - start
/home/kye/IsaacLab/isaaclab.sh -p -m brace_kinova.training.train_isaac_expert \
  --config brace_kinova/configs/isaac_expert.yaml

# Expert (SAC) - resume from checkpoint (+ optional replay buffer)
/home/kye/IsaacLab/isaaclab.sh -p -m brace_kinova.training.train_isaac_expert \
  --config brace_kinova/configs/isaac_expert.yaml \
  --resume_model checkpoints/isaac_expert_sac_time_<TIMESTAMP>_<STEPS>.zip \
  --resume_replay_buffer checkpoints/isaac_expert_sac_time_<TIMESTAMP>_<STEPS>.replay_buffer.pkl

# Belief pretraining (same synthetic data pipeline)
python -m brace_kinova.training.train_belief \
  --config brace_kinova/configs/belief.yaml

# Isaac arbitration - start
/home/kye/IsaacLab/isaaclab.sh -p -m brace_kinova.training.train_isaac_arbitration \
  --config brace_kinova/configs/isaac_arbitration.yaml

# Isaac arbitration - resume policy
/home/kye/IsaacLab/isaaclab.sh -p -m brace_kinova.training.train_isaac_arbitration \
  --config brace_kinova/configs/isaac_arbitration.yaml \
  --resume_model checkpoints/isaac_arbitration_policy_time_<TIMESTAMP>_<STEPS>.zip
```

All training scripts now support hourly wall-clock checkpoints via:
- `training.checkpoint_every_seconds` in YAML configs (default `3600`)

### 4. Monitor training

```bash
tensorboard --logdir=./logs
```

---

## Project Layout

```
.
├── brace_kinova/                     # BRACE training / eval / deployment
│   ├── envs/
│   │   ├── reach_grasp_env.py        #   Lightweight 2D Gymnasium env
│   │   ├── isaac_env.py              #   Isaac Lab Gymnasium env (Jaco2)
│   │   ├── isaac_config.py           #   Isaac env configuration
│   │   ├── scenarios.py              #   Curriculum scenario definitions
│   │   └── wrappers.py
│   ├── models/
│   │   ├── bayesian_inference.py     #   Learnable Bayesian goal inference
│   │   ├── arbitration_policy.py     #   PPO scalar gamma policy
│   │   ├── expert_policy.py          #   SAC expert wrapper
│   │   └── simulated_human.py        #   Min-jerk + AR(1) noise
│   ├── training/
│   │   ├── train_expert.py           #   SAC (2D)
│   │   ├── train_belief.py           #   Supervised NLL
│   │   ├── train_arbitration.py      #   PPO + curriculum (2D)
│   │   ├── train_isaac_expert.py     #   SAC (Isaac Sim)
│   │   ├── train_isaac_arbitration.py#   PPO + curriculum (Isaac Sim)
│   │   ├── rewards.py
│   │   ├── curriculum.py
│   │   └── callbacks.py
│   ├── evaluation/
│   ├── ros_interface/
│   ├── configs/
│   │   ├── env.yaml                  #   2D env config
│   │   ├── expert.yaml / belief.yaml / arbitration.yaml
│   │   ├── isaac_env.yaml            #   Isaac env config
│   │   ├── isaac_expert.yaml / isaac_arbitration.yaml
│   │   └── ...
│   ├── requirements.txt
│   └── README.md                     #   Detailed package docs
│
├── KINOVA_CODEBASE/                  # Isaac Lab Kinova tooling
│   ├── demo.py
│   ├── controllers/                  #   Cartesian velocity jog + safety
│   ├── environments/                 #   Scene design + object loading
│   ├── motion_generation/            #   RMPflow / cuRobo / LULA planners
│   ├── data_collection/              #   VLA-style dataset collection
│   ├── utilities/                    #   Transforms, robot reset
│   ├── copilot_demo/                 #   LLM copilot demo
│   └── README.md
│
├── Paper/                            # LaTeX source for the research paper
├── OLD/                              # Archived legacy material
├── PROMPT.md                         # Original Cursor agent design brief
├── IMPLEMENTATION_SUMMARY.md
└── README.md                         # ← you are here
```

---

## Two Environment Backends

| Feature              | 2D (`ReachGraspEnv`)        | Isaac (`IsaacReachGraspEnv`)        |
|----------------------|-----------------------------|--------------------------------------|
| Physics              | Kinematic integration       | Full Isaac Sim (PhysX)               |
| Robot model          | Point EE + binary gripper   | Kinova Jaco2 (6-DoF + 3-finger)     |
| Control              | Velocity → position step    | Diff-IK → joint velocity + grav comp|
| Speed                | ~50k steps/s (CPU)          | ~200 steps/s (GPU, single env)       |
| Collision detection  | Distance-based              | Distance-based (same logic)          |
| Obs / Action space   | Identical                   | Identical                            |

Both environments produce the **same observation and action vectors**, so
trained models (expert, belief, arbitration) transfer between backends
without retraining.

---

## BRACE System

BRACE jointly learns:

- **Bayesian Goal Inference**: Recursive belief over N candidate goals using a
  noisy-rational (Boltzmann) likelihood with learnable parameters (β, w_θ, w_d).
- **Arbitration Policy (PPO)**: Outputs scalar γ ∈ [0,1] from `[state, belief]`;
  blends human and expert: `a = (1−γ)h + γw`.
- **Expert Policy (SAC)**: Frozen during arbitration training; sees true goal.

Training follows a 5-stage curriculum from simple reaching to full complexity
with multiple goals and obstacles.

---

## ROS 1 Deployment

For real-robot deployment with a DualSense PS5 controller:

```bash
rosrun brace_kinova brace_node.py _config_path:=brace_kinova/configs/arbitration.yaml
```

Requires `rospy`, `ros_kortex`, and a connected Kinova arm.

---

## Notes

- Detailed package documentation: [`brace_kinova/README.md`](brace_kinova/README.md)
- Kinova Isaac tooling docs: [`KINOVA_CODEBASE/README.md`](KINOVA_CODEBASE/README.md)
- Implementation breakdown: [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md)
- Design brief: [`PROMPT.md`](PROMPT.md)
