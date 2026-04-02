# BRACE Kinova — Package Documentation

**BRACE** (Bayesian Reinforcement Assistance with Context Encoding) implements
shared-autonomy assistance arbitration for a Kinova Jaco2 robot performing
planar reach-and-grasp tasks.  The system learns *when* and *how much* to assist
a human operator by jointly training a Bayesian goal-inference module and a
PPO-based arbitration policy.

---

## Architecture Overview

```
┌────────────┐    h_t     ┌───────────────┐   belief    ┌─────────────────┐
│  Human /   │──────────▶│   Bayesian     │───────────▶│  PPO Arbitration │
│  Simulated │           │   Inference    │            │  Policy          │
└────────────┘           └───────────────┘            └────────┬────────┘
                                                               │ γ ∈ [0,1]
┌────────────┐    w_t     ┌───────────────┐                    │
│  SAC Expert│──────────▶│  Blending      │◀───────────────────┘
│  (frozen)  │           │  (1-γ)h + γw   │
└────────────┘           └───────┬───────┘
                                 │ a_exec
                           ┌─────▼──────┐
                           │ Environment │
                           │ (2D / Isaac)│
                           └────────────┘
```

## Package Layout

```
brace_kinova/
├── configs/                     # YAML configuration files
│   ├── env.yaml                 # 2D lightweight env
│   ├── expert.yaml              # SAC expert (2D)
│   ├── belief.yaml              # Bayesian inference pretraining
│   ├── arbitration.yaml         # PPO arbitration (2D)
│   ├── isaac_env.yaml           # Isaac Sim env
│   ├── isaac_expert.yaml        # SAC expert (Isaac)
│   └── isaac_arbitration.yaml   # PPO arbitration (Isaac)
├── envs/
│   ├── reach_grasp_env.py       # Lightweight 2D Gymnasium env (no physics)
│   ├── isaac_env.py             # Isaac Lab Gymnasium env (Kinova Jaco2)
│   ├── isaac_config.py          # Dataclass configs for Isaac env
│   ├── scenarios.py             # Curriculum scenario definitions
│   └── wrappers.py              # Obs normalisation, goal masking
├── models/
│   ├── bayesian_inference.py    # Learnable Bayesian goal inference
│   ├── arbitration_policy.py    # PPO gamma-arbitration policy (SB3)
│   ├── expert_policy.py         # SAC wrapper + potential-field baseline
│   └── simulated_human.py       # Min-jerk + AR(1) noise human model
├── training/
│   ├── train_expert.py          # SAC expert training (2D env)
│   ├── train_belief.py          # Supervised belief pretraining
│   ├── train_arbitration.py     # PPO arbitration training (2D env)
│   ├── train_isaac_expert.py    # SAC expert training (Isaac Sim)
│   ├── train_isaac_arbitration.py # PPO arbitration training (Isaac Sim)
│   ├── rewards.py               # Expert and arbitration reward functions
│   ├── curriculum.py            # 5-stage curriculum manager
│   └── callbacks.py             # SB3 callbacks (curriculum, metrics, ckpt)
├── evaluation/
│   ├── evaluate.py              # Rollout and metric aggregation
│   └── visualize.py             # Matplotlib trajectory / gamma plots
├── ros_interface/
│   ├── brace_node.py            # ROS 1 inference node
│   ├── kinova_bridge.py         # Cartesian velocity → ros_kortex
│   └── dualsense_interface.py   # DualSense PS5 controller input
└── requirements.txt
```

## Environments

### 1. Lightweight 2D (`ReachGraspEnv`)

- **No physics engine** — simple kinematic integration.
- Fast iteration (~50k steps / second on CPU).
- Use for algorithm prototyping and baseline comparison.

### 2. Isaac Sim (`IsaacReachGraspEnv`)

- Full robot dynamics via **Isaac Lab** with the **Kinova Jaco2 (J2N6S300)**.
- Differential IK for Cartesian velocity control.
- Gravity compensation, joint limits, realistic inertia.
- Visual cuboid goals (green) and obstacles (red) on a tabletop.
- **Same observation/action spaces** as the 2D env — models transfer directly.

Both environments share the identical observation structure:

| Component               | Dim                   |
|--------------------------|-----------------------|
| EE position (XY)        | 2                     |
| EE velocity (XY)        | 2                     |
| Gripper state            | 1                     |
| Relative XY to objects   | 2 × n_objects         |
| Relative XY to obstacles | 2 × n_obstacles       |
| Min obstacle distance    | 1                     |
| Progress per object      | n_objects             |
| **Total**                | 5 + 3×n_obj + 2×n_obs |

Action space: `Box(3)` → `[vx, vy, gripper]` in [-1, 1].

## Training Pipeline

### Three-stage training (run in order):

#### 1. Expert (SAC)

Trains a goal-conditioned expert that sees the true target identity.

```bash
# 2D env
python -m brace_kinova.training.train_expert \
    --config brace_kinova/configs/expert.yaml

# Isaac Sim
isaaclab -p -m brace_kinova.training.train_isaac_expert \
    --config brace_kinova/configs/isaac_expert.yaml --headless
```

#### 2. Belief pretraining (supervised NLL)

Generates synthetic human trajectories and trains the Bayesian inference
module to predict the true goal.

```bash
python -m brace_kinova.training.train_belief \
    --config brace_kinova/configs/belief.yaml
```

> Note: Belief pretraining uses synthetic data with identical observation
> semantics to both envs, so a single run works for both backends.

#### 3. Arbitration (PPO + curriculum)

Trains the gamma-arbitration policy with 5-stage curriculum.

```bash
# 2D env
python -m brace_kinova.training.train_arbitration \
    --config brace_kinova/configs/arbitration.yaml

# Isaac Sim
isaaclab -p -m brace_kinova.training.train_isaac_arbitration \
    --config brace_kinova/configs/isaac_arbitration.yaml --headless
```

### Curriculum Stages

| Stage | Objects | Obstacles | Advancement Criteria            |
|-------|---------|-----------|----------------------------------|
| 1     | 1       | 0         | 80% success                      |
| 2     | 1       | 3         | 75% success, < 15% collisions   |
| 3     | 1       | 4         | 70% success                      |
| 4     | 3       | 3         | 65% success                      |
| 5     | 3       | 4         | Reward plateau (200-ep window)   |

## Isaac Sim Integration

The Isaac environment (`isaac_env.py`) bridges the BRACE training
pipeline with the Kinova Jaco2 in Isaac Lab:

- **Scene setup**: `setup_brace_scene()` creates the table, robot, and
  lighting using the same pattern as `KINOVA_CODEBASE/environments/`.
- **Controller**: Direct `DifferentialIKController` for Cartesian
  velocity → joint velocity, with held orientation and gravity comp.
- **Object management**: Visual cuboids spawned at init; teleported via
  USD Xform ops on each episode reset.
- **Planar constraint**: EE Z height is held constant; only XY velocity
  commands are exposed to the policy.

### Prerequisites

- **Isaac Sim 4.5+** and **Isaac Lab** installed.
- `isaaclab_assets` package available (ships with Isaac Lab).
- Run training scripts through the Isaac Lab launcher:
  `isaaclab -p -m brace_kinova.training.train_isaac_expert ...`

## Evaluation

```bash
python -m brace_kinova.evaluation.evaluate \
    --config brace_kinova/configs/arbitration.yaml \
    --n-episodes 100 --output results.json
```

## ROS 1 Deployment

```bash
rosrun brace_kinova brace_node.py \
    _config_path:=brace_kinova/configs/arbitration.yaml
```

Requires `rospy`, `ros_kortex`, and a connected Kinova Gen2/Gen3.

## Key Design Decisions

1. **Observation parity**: The Isaac env produces the *exact same*
   observation vector as the 2D env, enabling zero-shot model transfer.
2. **Distance-based detection**: Collision and grasp checks use XY
   distance thresholds (not PhysX contacts) for consistency with the
   2D env and deterministic behaviour.
3. **Separate training scripts**: Isaac Sim requires `AppLauncher`
   before any Isaac imports, so Isaac training scripts are standalone
   (`train_isaac_*.py`) rather than flags on the existing scripts.
4. **Curriculum reconfiguration**: `env_update_fn` now properly calls
   `set_scenario()` on running envs when the curriculum advances.
