# BRACE Kinova — Implementation Summary

## What Was Built

The entire BRACE (Bayesian Reinforcement Assistance with Context Encoding) codebase for Experiment 4 — a 2D planar reach-and-grasp task on the Kinova Gen3 arm. This codebase was built from the specifications in `PROMPT.md` and the patterns extracted from the OLD Jupyter notebooks.

---

## Project Structure (29 files created)

```
brace_kinova/
├── __init__.py                          # Package root
├── requirements.txt                     # Python dependencies
├── configs/
│   ├── env.yaml                         # Workspace bounds, object/obstacle params, episode settings
│   ├── expert.yaml                      # SAC expert training hyperparameters
│   ├── belief.yaml                      # Bayesian inference pretraining config
│   └── arbitration.yaml                 # PPO arbitration + curriculum + reward weights
├── envs/
│   ├── __init__.py
│   ├── reach_grasp_env.py               # Gymnasium env: ReachGraspEnv, ExpertReachGraspEnv, ArbitrationEnv
│   ├── wrappers.py                      # NormalizeObservation, GoalMaskedObservation, ClipAction
│   └── scenarios.py                     # 5 curriculum scenarios (basic → full complexity)
├── models/
│   ├── __init__.py
│   ├── bayesian_inference.py            # BayesianGoalInference (nn.Module, learnable β/wθ/wd)
│   ├── arbitration_policy.py            # GammaArbitrationPolicy (SB3 ActorCriticPolicy subclass)
│   ├── expert_policy.py                 # ExpertPolicy (SAC wrapper) + PotentialFieldExpert (fallback)
│   └── simulated_human.py              # SimulatedHuman (min-jerk + potential field + AR(1) pink noise)
├── training/
│   ├── __init__.py
│   ├── rewards.py                       # ExpertReward + ArbitrationReward (paper equations)
│   ├── curriculum.py                    # CurriculumManager (5 stages, auto-advancement)
│   ├── callbacks.py                     # CurriculumCallback, MetricsCallback, CheckpointCallback
│   ├── train_expert.py                  # SAC expert training (CLI script)
│   ├── train_belief.py                  # Bayesian belief pretraining (CLI script)
│   └── train_arbitration.py             # Full BRACE training with PPO + belief (CLI script)
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py                      # Evaluation: success rate, collisions, belief accuracy, baselines
│   └── visualize.py                     # Trajectory plots, belief entropy, gamma over time
└── ros_interface/
    ├── __init__.py
    ├── brace_node.py                    # ROS 1 node for real-time BRACE inference (~27 Hz)
    ├── kinova_bridge.py                 # 2D→3D velocity translation for ros_kortex
    └── dualsense_interface.py           # DualSense PS5 controller (pygame + ROS backends)
```

---

## Key Implementation Details

### Models

| Module | Description | Key Details |
|--------|-------------|-------------|
| `BayesianGoalInference` | Learnable Bayesian belief over N goals | `raw_beta=2.0`, `raw_w_theta=0.8`, `raw_w_dist=0.2` via `nn.Parameter` + `softplus`. Noisy-rational (Boltzmann) likelihood. EMA smoothing α=0.85. Temperature annealing. |
| `GammaArbitrationPolicy` | SB3-compatible PPO policy for scalar γ | 256×256 ReLU trunk → tanh actor head (1D) + value head. γ = 0.5*(a+1). Learned log_std clamped [-20, 2]. |
| `ExpertPolicy` | Frozen SAC model wrapper | Loads `expert_sac.zip`. Full goal observability. `PotentialFieldExpert` as analytical fallback. |
| `SimulatedHuman` | Noisy human operator model | Min-jerk trajectory + potential-field via-points + AR(1) pink noise (amplitude=3.2%, ar_coeff=0.5). |

### Environment

- **2D planar**: All motion in XY plane at fixed Z height. Observation ~21D, action 3D (vx, vy, gripper).
- **Gymnasium-compliant**: Registered as `BraceReachGrasp-v0` and `BraceExpertReachGrasp-v0`.
- **Goal ambiguity**: Multiple candidate objects, one hidden true target per episode.
- **Collision**: EE-obstacle contact terminates episode with -10 penalty.
- **Success**: Gripper closes around correct object within 0.04m threshold, +5 reward.

### Reward Functions (from paper)

**Expert reward** (dense, for SAC training):
```
R = 3.0*(d_{t-1} - d_t)/d_max - 0.8*||Δθ||² - 2.5*exp(-d_obs_min/d_safe)
```

**Arbitration reward** (belief-aware, for PPO training):
```
R = -10.0*1_collision + 2.5*γ*p_max*1_near - 1.5*γ*1_far
    + 3.0*p_max*(d_{t-1} - d_t) - 1.5*γ² + 2.0*log(p_true)
```

### Curriculum (5 stages)

| Stage | Objects | Obstacles | Min Episodes | Success Threshold |
|-------|---------|-----------|--------------|-------------------|
| 1. Basic reaching | 1 | 0 | 100 | 80% |
| 2. Collision avoidance | 1 | 3 | 200 | 75% (collision <15%) |
| 3. Challenging obstacles | 1 | 4 | 300 | 70% |
| 4. Goal ambiguity | 3 | 3 | 400 | 65% |
| 5. Full complexity | 3 | 4 | — | Reward plateau over 200 episodes |

### Training Hyperparameters

| Parameter | Expert (SAC) | Arbitration (PPO) | Belief Pretrain |
|-----------|-------------|-------------------|-----------------|
| Learning rate | 3e-4 | 3e-4 (cosine anneal) | 5e-3 |
| Batch size | 1024 | 1024 | 512 |
| Network | [256, 256, 256] | pi/vf=[256, 256] | — |
| Timesteps | 1,000,000 | 2,000,000 | 200 epochs |
| γ (discount) | 0.99 | 0.99 | — |
| Device | auto (CUDA if available) | auto (CUDA if available) | auto |

### ROS 1 Interface

- `brace_node.py`: Full BRACE inference node at ~27 Hz. Subscribes to `/joy`, `/object_positions`, `/obstacle_positions`, `/ee_state`. Publishes blended velocity to `/my_gen3/in/cartesian_velocity`.
- `kinova_bridge.py`: Converts 2D (vx, vy) → 3D (vx, vy, 0) for `ros_kortex`, handles gripper, enforces workspace bounds.
- `dualsense_interface.py`: Pygame backend (simulation) and ROS backend (real-world). 5 control modes matching the user study. Deadzone=0.1, L2/R2 for manual gamma.

---

## Training Commands (Run in Order)

All commands should be run from the repository root:
```
cd /home/kye/Desktop/Depo/Code/BRACE_KINOVA/BRACE_KINOVA
```

### Step 0: Install Dependencies

```bash
pip install -r brace_kinova/requirements.txt
```

### Step 1: Train Expert (SAC)

```bash
python -m brace_kinova.training.train_expert --config brace_kinova/configs/expert.yaml
```

- **Output**: `checkpoints/expert_sac.zip`, `checkpoints/expert_vecnormalize.pkl`
- **Duration**: ~1-2 hours on RTX 4090 (1M timesteps)
- **What it does**: Trains a SAC expert that knows the true goal and navigates optimally with obstacle avoidance. Uses the `ExpertReachGraspEnv` which appends the true goal one-hot to observations.

### Step 2: Pretrain Belief Module

```bash
python -m brace_kinova.training.train_belief --config brace_kinova/configs/belief.yaml
```

- **Output**: `checkpoints/bayesian_inference.pt`
- **Duration**: ~25-30 minutes on RTX 4090 (200 epochs, 5000 trajectories)
- **What it does**: Generates synthetic trajectories using the simulated human model, then optimizes the Bayesian inference module's learnable parameters (β, w_θ, w_d) via supervised NLL loss on true goal labels.

### Step 3: Train BRACE Arbitration (PPO + Curriculum)

```bash
python -m brace_kinova.training.train_arbitration --config brace_kinova/configs/arbitration.yaml
```

- **Output**: `checkpoints/arbitration_policy.zip`, `checkpoints/bayesian_inference_finetuned.pt`
- **Duration**: ~4-5 hours on RTX 4090 (2M timesteps with 5-stage curriculum)
- **What it does**: Jointly trains the PPO arbitration policy and fine-tunes the Bayesian belief module. Uses the pretrained expert (frozen) and simulated human. Progresses through 5 curriculum stages.

### Step 4: Evaluate

```bash
python -m brace_kinova.evaluation.evaluate --config brace_kinova/configs/arbitration.yaml --n-episodes 100 --output results.json
```

- **Output**: Printed comparison table + `results.json`
- **What it does**: Evaluates BRACE against baselines (human-only, expert-only, fixed-gamma). Reports success rate, time-to-grasp, collisions, belief accuracy at 25/50/75% path completion, and gamma statistics.

### Step 5: Deploy on Real Kinova (ROS 1)

```bash
# Requires ROS 1 + ros_kortex installed
rosrun brace_kinova brace_node.py _config_path:=brace_kinova/configs/arbitration.yaml
```

---

## Optional: GPU Device Selection

All training scripts accept a `--device` flag:

```bash
python -m brace_kinova.training.train_expert --config brace_kinova/configs/expert.yaml --device cuda
python -m brace_kinova.training.train_belief --config brace_kinova/configs/belief.yaml --device cuda
python -m brace_kinova.training.train_arbitration --config brace_kinova/configs/arbitration.yaml --device cuda
```

By default, `device: auto` in the YAML configs will use CUDA if an NVIDIA GPU is available.

## TensorBoard Monitoring

Training logs are written to `./logs/`. To monitor:

```bash
tensorboard --logdir=./logs
```

Then open `http://localhost:6006` in a browser.

---

## Architecture Patterns (from Old Code)

The following patterns were preserved from the OLD notebooks:

- **γ mapping**: `γ = 0.5 * (a + 1)` where `a ∈ [-1, 1]` from tanh actor output
- **Learnable Bayesian params**: `nn.Parameter(torch.tensor(init_val))` → `F.softplus(raw_param)` for positivity
- **Belief update**: `posterior ∝ prior × likelihood`, normalized, with EMA smoothing
- **Blending**: `a_exec = (1 - γ) * h_t + γ * w_t`
- **SB3 compatibility**: `GammaArbitrationPolicy` subclasses `ActorCriticPolicy` with tanh squashing
- **Expert freeze**: Expert SAC is loaded once and never updated during arbitration training
- **Vectorized training**: Multiple parallel environments via `DummyVecEnv`/`SubprocVecEnv`
