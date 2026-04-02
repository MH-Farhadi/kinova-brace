# BRACE Kinova — Cursor Agent Prompt

> **Purpose**: Feed this entire file to Cursor (Agent mode) to build a clean, modular Python codebase for training and deploying BRACE on a Kinova Gen3 arm (7-DOF) for a **2D planar** reach-and-grasp task (Experiment 4 in the ACM THRI paper revision). Training in Isaac Sim, deployment via ROS 1 with DualSense PS5 controller. This experiment is the primary response to reviewer demands for real-world robot validation (see Section 11 for full reviewer comments). **GPU target: NVIDIA RTX 4090 laptop.**

---

## 1. What BRACE Is

BRACE (**Bayesian Reinforcement Assistance with Context Encoding**) is a shared-autonomy framework with two jointly trained modules:

### 1.1 Bayesian Inference Module

Maintains a recursive belief distribution over N candidate goals. At each timestep, given human input `h_t` and state `s_t`, it updates:

```
P(g_i | X, H) ∝ P(g_i) · Π_{j=1..t} P(h_j | x_j, g_i)
```

The per-step likelihood is noisy-rational (Boltzmann):

```
P(h_j | x_j, g_i) ∝ exp(-β · cost(h_j | x_j, g_i))
```

Cost combines angular and distance deviation:

```
cost(h_j | x_j, g_i) = w_θ · θ_dev + w_d · d_dev

where:
  θ_dev = |arccos( (h_j · v_{x→g_i}) / (|h_j| · |v_{x→g_i}|) )|
  d_dev = |1 - |h_j| / h_opt|
```

Calibrated parameters (from paper): `β=10.0`, `w_θ=0.7`, `w_d=0.3`. Temporal EMA smoothing `α=0.85`. Uniform priors over goals.

The module has **learnable parameters** (`β`, `w_θ`, `w_d`) implemented via `nn.Parameter` with softplus activation (see old code patterns below). It can be:
- **Pretrained** via supervised NLL on trajectories with known goal labels.
- **Fine-tuned end-to-end** via REINFORCE gradients from the arbitration reward.

### 1.2 Arbitration Policy (PPO Actor-Critic)

Takes fused `[state_features, belief_vector]` as input (Feature Fusion Block), outputs scalar `γ ∈ [0,1]`. The executed action is:

```
a_t = (1 - γ_t) · h_t + γ_t · w_t
```

where `h_t` is the human input and `w_t` is the expert policy output.

**Architecture** (from paper + old code):
- Shared trunk: `input_dim → 256 → 256` (ReLU)
- Actor head: `256 → 1` (tanh), mapped from `[-1, 1]` to `[0, 1]` via `γ = 0.5 * (a + 1)`
- Critic head: `256 → 1` (linear, value estimate)
- Learned `log_std` parameter (single scalar, clamped `[-20, 2]`)
- Action distribution: `Normal(tanh(mu), softplus(log_std))`, then `tanh` on the sample

**PPO hyperparameters** (from paper):
- Learning rate: `3e-4` with cosine annealing
- Batch size: `1024`
- Epochs per batch: `4`
- Discount `γ`: `0.99`
- GAE `λ`: `0.95`
- Clip range: `0.2`
- Network: `pi=[256, 256]`, `vf=[256, 256]`, activation `ReLU`

### 1.3 Joint Training

Training jointly updates both modules. The mixed objective is:

```
L_total(θ, φ) = α · L_RL(θ, φ) + (1 - α) · L_supervised(φ)
```

where:
- `L_supervised(φ) = -E[log P_φ(g* | X, H)]` (NLL on true goal)
- `∇_φ L_RL ≈ E[ Σ_t R_t · ∇_φ log P_φ(b_t | X_{1:t}, H_{1:t}) ]` (REINFORCE)
- `α` is annealed during training

Stability mechanisms:
- Critic baseline: advantage `A(s_t) = R_t - V(s_t)` for variance reduction
- Gradient L2 scaling: if `||∇_φ L_RL||_2 > C_clip`, scale `g ← g · C_clip / ||g||_2`
- Temperature annealing: softmax temperature `τ` on belief, initially `τ ≥ 1.0`, decayed over training
- Confidence scaling: `α = min(1.0, p_max / c)` with `c = 0.8`
- Minibatch advantage normalization (zero mean, unit variance)

### 1.4 Curriculum (5 stages)

1. **Basic goal-directed behavior**: 100 episodes, >80% success rate
2. **Basic collision avoidance**: 200 episodes, >75% success, <15% collision rate
3. **Challenging obstacle configurations**: 300 episodes, >70% success
4. **Goal ambiguity** (2–3 potential targets): 400 episodes, >65% success
5. **Full complexity**: train until convergence (reward plateau for 200 episodes)

### 1.5 Reward Function

From the paper appendix (full form):

```
R = -w_coll · 1_collision
    + w_prox · γ · p_max · 1_near
    - w_far  · γ · 1_far
    + w_prog · p_max · (d_{t-1} - d_t)
    - w_auto · γ²
    + w_goal · log(p_true)
```

Weights: `w_coll=10.0`, `w_prox=2.5`, `w_far=1.5`, `w_prog=3.0`, `w_auto=1.5`, `w_goal=2.0`.

Indicators: `1_collision` (contact with obstacle), `1_near` (near predicted target), `1_far` (far from all targets).

### 1.6 Expert Policy Reward

```
R_expert(s, a) = 3.0 · (d_{t-1} - d_t) / d_max
               - 0.8 · ||Δθ||²
               - 2.5 · exp(-min_{o ∈ O} ||s - o|| / d_safe)
```

### 1.7 Simulated Human Model

- **Deterministic trajectory**: Minimum-jerk polynomial regression toward the true goal.
- **Via-points near obstacles**: Potential field approach (attraction to goal + repulsion from obstacles).
- **Stochastic noise**: AR(1) filter transforming white Gaussian noise into 1/f (pink) noise:
  ```
  y[n] = 0.5 · y[n-1] + 0.5 · x[n]
  ```
  Spectral slope: `-1.59`. Noise amplitude: `3.2%` (SD `2.7%`) of trajectory length.
- Sensitivity ranges tested: slope `[-1.2, -1.8]`, amplitude `[1.5%, 5.0%]`.

---

## 2. The Old Code (Reference Implementation)

All old code lives in `OLD/End-Effector-Test-Environment/`. It is implemented as Jupyter notebooks (NOT importable modules). Read these notebooks thoroughly to understand the patterns, then build the new code as proper Python modules.

### 2.1 Key Notebooks to Read

| Notebook | What It Contains |
|----------|-----------------|
| `BRACE/Reacher/brace integrated.ipynb` | **Full integrated BRACE** on Reacher-v4. Classes: `BayesianIntent` (8 goals on circle, learnable `raw_beta=2.0`, `raw_wang=0.8`), `GammaAC` (256×256 trunk, tanh actor, value head), `VecReacher` (32 parallel envs). Training: 8000 epochs, rollout_len=64, Adam γ@3e-4, Bayes@1e-3, discount 0.99. Policy input: `[obs_11D, belief_8D] = 19D`. |
| `BRACE/Reacher/expert.ipynb` | **SAC expert** for Reacher-v4. Hyperparams: lr=3e-4, buffer=400k, batch=1024, tau=0.005, γ=0.99, train_freq=64, gradient_steps=64, learning_starts=10k, 1M steps. |
| `BRACE/Reacher/belief pretraining.ipynb` | **Belief pretraining** using random actions. Adam lr=1e-2, batch 512, 200 epochs. Saves `bayes_reacher.pt`. |
| `BRACE/Reacher/surrogate pilot.ipynb` | **Simulated noisy human** for Reacher (expert + noise). |
| `BRACE/cursor control/end-to-end brace.ipynb` | **End-to-end BRACE** with `SimWorld` (vectorized cursor env). `BayesianIntent` (learnable β, w_th, w_dist), `GammaActorCritic` (256×256). Training: 6000 epochs, batch=64, unroll=64, lr_gamma=3e-4, lr_bayes=1e-3. Reward: `-0.01 + 0.5*(prev_dist - d_goal) + 2.0 if goal_reached - 4.0 if collision`. |
| `BRACE/cursor control/brace actor critic.ipynb` | **PPO arbitration** with SB3-compatible `GammaBeliefPolicy`. `BraceArbitrationEnv`: obs=`[x, y, d_goal_norm, d_obs_norm, belief_1..N]` (4+N dims), action=scalar `[-1,1]`. Reward: `-20*(γ - γ_ideal)²`, `-2` on collision. PPO: lr=3e-4, n_steps=1024, batch=1024, n_epochs=4, γ=0.995, GAE=0.97, clip=0.2, 600k timesteps. |
| `BRACE/cursor control/expert policy cursor.ipynb` | **SAC expert** on `BraceExpertEnv`. Obs: `[cursor_xy, goal_xy, d_goal_norm, d_obs_norm]` (6D). Action: 2D continuous `[-1,1]². Reward: step_penalty=-0.01, goal_bonus=+2.0, collision=-4.0, plus distance shaping. |
| `BRACE/cursor control/Bayesian Inference training.ipynb` | **Standalone belief pretraining** via supervised NLL on JSON trajectories. `BayesianGoalInference`: learnable raw_beta=2.0, raw_wth=0.8, raw_wdist=0.2 (softplus). Adam lr=5e-3, 20 epochs. Saves `bayes_goal_inference.pt`. |
| `BRACE/cursor control/assist controller.ipynb` | **Runtime controller**: `BayesFilter` + `GammaPolicy` (256×256→tanh scalar). obs_dim=18 = 10 state + 8 belief. Also has `SmallMLP` baselines for IDA/DQN. |
| `BRACE/cursor control/ablations.ipynb` | **Ablation grid**: reward variants, belief variants, curriculum on/off. PPO with MlpPolicy, 200k timesteps per cell. |
| `BRACE/Reach_to_grasp/plgrnd.ipynb` | **Early Fetch experiments**: `FetchPickAndPlaceDense-v4` with `gymnasium_robotics`. PPO: n_steps=2048, batch=32, n_epochs=6, γ=0.98, GAE=0.95, lr=3e-4, clip=0.25, ent_coef=0.05, 4 parallel envs, 1M steps, VecNormalize. |
| `PPO_SharedControl_MK8.ipynb` | **Primary gamma arbitration training**. `DemoArbitrationEnv`: 10D obs `[dot_pos, h_dir, goal_pos, w_dir, dist_ratio, obs_dist_ratio]`, scalar action `[-1,1]`, 8 goals, 5 obstacles. `GammaMlpPolicy` (SB3 ActorCriticPolicy subclass, tanh on raw mean). Reward: `-20*(γ - desired_γ)²`, `-2` on collision. Desired γ: 0.2 (near goal+obstacle), 0.3 (near goal), 0.4 (near obstacle), 0.8 (else). PPO: lr=3e-4, n_steps=1024, batch=1024, n_epochs=4, γ=0.99, GAE=0.95, clip=0.2, net_arch pi/vf=[256,256]. Saves `gamma_ppo_model.zip`. |
| `Controller_MK14.ipynb` | **Runtime/user-study controller**: Loads `gamma_ppo_model.zip`, Pygame visualization, optional serial force sensor input, pink noise via `scipy.signal.lfilter`. |

### 2.2 Common Patterns Across Old Code

- **γ mapping**: All pipelines use `γ = 0.5 * (a + 1)` with `a ∈ [-1, 1]` (scalar output from tanh).
- **Learnable Bayesian params**: `nn.Parameter(torch.tensor(initial_val))` → `F.softplus(raw_param)` to ensure positivity.
- **Belief update**: `belief = belief * likelihood; belief = belief / belief.sum()` (normalize).
- **Vectorized training**: Batch of parallel environments for speed (32 envs in Reacher, 64 batch in cursor).
- **Expert**: Always a frozen SAC or potential-field oracle, loaded separately.
- **SB3 compatibility**: Custom policies subclass `ActorCriticPolicy`, override `forward()` to add tanh squashing and return `(action, value, log_prob)`.

---

## 3. Paper Context — Experiment 3 (What Experiment 4 Simplifies)

Experiment 3 in the paper (the most complex) used:
- **Modified `FetchPickAndPlace-v3`**: 1 graspable cube, 3 visually similar target bins, obstacles constraining the path.
- At each episode start, one bin is randomly sampled as the hidden target → goal ambiguity.
- Obstacles create localized safety demands.
- Control via **Cartesian end-effector velocity**.
- Compared BRACE vs IDA vs DQN. Results: BRACE 86% success (vs 74% DQN, 68% IDA), fastest time (9.8s), fewest collisions (0.22).

---

## 4. Experiment 4 — What to Build (New)

This is a **simpler** version of Experiment 3 for a real-world Kinova deployment, deliberately constrained to a **2D planar workspace** to speed up training and improve model/process stability.

**Task**: Reach-and-grasp ONLY (NO pick-and-place, NO placing into bins).
- The Kinova arm reaches toward a target object while avoiding obstacles, then grasps it.
- All motion is constrained to a **2D planar workspace** (XY plane at a fixed height above the table). The end-effector moves only in the horizontal plane; the Z axis is locked to a constant height during reaching, and only changes for the final grasp descent. This reduces the action/observation space, speeds up training, and makes the model more stable.
- Multiple candidate goal objects (e.g., 3 objects on a table) create **goal ambiguity** — the system doesn't know which one the human wants.
- Obstacles along the path create **localized safety demands** requiring context-aware assistance.
- The human operator provides input via a **DualSense PS5 controller** (left analog stick for XY velocity, L2/R2 triggers for manual gamma adjustment in the manual-gamma condition).
- The expert provides optimal trajectories toward the inferred/true goal while avoiding obstacles.
- BRACE arbitrates between human and expert based on belief over goals and context (obstacle proximity, belief entropy).

**Robot**: Kinova Gen3 arm (2nd generation), 7-DOF. Although the arm has 7 joints, all experiments operate in a **2D planar XY workspace** to reduce complexity. Controlled via Cartesian end-effector velocity commands projected onto the XY plane.

**Simulation environment**: Training happens in **Isaac Sim** using the existing kinova-isaac codebase (see Section 10). The kinova-isaac codebase already has a `reach_to_grasp_VLA` environment with a top-down camera, Thorlabs table, and object spawning. **NOTE**: The current kinova-isaac setup uses a Kinova **Jaco2 J2N6S300** (6-DOF) robot model. This must be updated to use the **Kinova Gen3** (7-DOF) model, or the existing Jaco2 model can be used as a stand-in for training since we only need XY planar control (the specific arm kinematics matter less when projecting to 2D).

**Sim-to-Real connection**: A separate bridge between Isaac Sim and the real Kinova Gen3 robot already exists and has been validated with previous models trained in Isaac Sim. The real robot runs on **ROS 1** (not ROS 2) — the Kinova Gen3 only supports the ROS 1 `kortex_driver` stack.

**Deployment**: The trained model will be deployed via the existing **Isaac Sim ↔ ROS 1 bridge** for real-world experiments. Training happens in Isaac Sim; inference happens on the real robot through the existing sim-real connection. The ROS interface node must target **ROS 1** (rospy, not rclpy).

**Object Detection**: A separate YOLO-based model for object segmentation and detection has already been trained. It is not on this machine but will be integrated later for real-world perception. For now, focus on simulation training where ground-truth object positions are available from the simulator.

**Camera**: The kinova-isaac `reach_to_grasp_VLA` environment already has a **top-down camera** configured at position `(0.4, 0.0, 4.0)` pointing down at the table workspace center `(0.4, 0.0, 0.8)`, resolution `640×640`, FOV `65°`. This camera is sufficient for training purposes and matches the 2D planar workspace view.

**Purpose — Addressing Reviewer Concerns**: This experiment is critically important for the paper revision. Reviewers demanded real-world robot experiments to validate BRACE beyond simulation (see Section 11 for full reviewer comments). Experiment 4 directly addresses Reviewer 1 Point 5 ("simulated environments and relatively simple tasks limits the strength of the empirical validation") and the AE consensus that the paper needs stronger real-world evidence.

---

## 5. Project Structure to Create

Build all new code under the repo root (NOT inside `OLD/`):

```
brace_kinova/
├── __init__.py
├── envs/
│   ├── __init__.py
│   ├── reach_grasp_env.py          # Gymnasium env: reach-and-grasp with obstacles + goal ambiguity
│   ├── wrappers.py                 # Observation wrappers, normalization, flattening
│   └── scenarios.py                # Scenario configs: object positions, obstacle layouts, difficulty levels
├── models/
│   ├── __init__.py
│   ├── bayesian_inference.py       # BayesianGoalInference module (learnable β, w_θ, w_d)
│   ├── arbitration_policy.py       # GammaActorCritic (SB3-compatible PPO policy)
│   ├── expert_policy.py            # Expert policy wrapper (loads trained SAC)
│   └── simulated_human.py          # Simulated human: min-jerk + potential field + AR(1) pink noise
├── training/
│   ├── __init__.py
│   ├── train_expert.py             # Train SAC expert for reach-and-grasp
│   ├── train_belief.py             # Pretrain Bayesian inference module (supervised NLL)
│   ├── train_arbitration.py        # Full BRACE training: PPO + REINFORCE on belief, with curriculum
│   ├── curriculum.py               # Curriculum manager (5 stages with advancement criteria)
│   ├── rewards.py                  # BRACE reward function (all components)
│   └── callbacks.py                # SB3 callbacks: logging, checkpointing, curriculum advancement
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py                 # Evaluation: success rate, time, collisions, belief accuracy
│   └── visualize.py                # Trajectory viz, belief entropy plots, γ over time
├── ros_interface/
│   ├── __init__.py
│   ├── brace_node.py               # ROS 1 node: loads BRACE model, runs online inference
│   ├── kinova_bridge.py            # Maps BRACE 2D Cartesian velocity outputs to Kinova ros_kortex commands
│   └── dualsense_interface.py      # DualSense PS5 controller input handler (reads /joy, deadzone, mapping)
├── configs/
│   ├── expert.yaml                 # Expert SAC training hyperparameters
│   ├── belief.yaml                 # Belief pretraining config
│   ├── arbitration.yaml            # Arbitration PPO + curriculum config
│   └── env.yaml                    # Environment: workspace bounds, object/obstacle params
├── requirements.txt
└── README.md
```

---

## 6. Detailed Requirements

### 6.1 Environment — `reach_grasp_env.py`

Build a Gymnasium-compliant environment for the Kinova reach-and-grasp task. The environment must support **both** an Isaac Sim backend (for training in the existing kinova-isaac setup) and a lightweight standalone mode (for rapid prototyping/evaluation without Isaac Sim).

**Primary approach**: Build an Isaac Sim / Isaac Lab compatible environment that interfaces with the existing `kinova-isaac` codebase at `/home/kye/Desktop/Depo/Code/kinova-isaac`. The kinova-isaac setup already provides:
- A `reach_to_grasp_VLA` environment with scene design (Thorlabs table, robot, dome light, ground plane)
- Object spawning from YCB dataset or custom USD assets with configurable AABB bounds: min `[0.2, -0.3, 0.9]`, max `[0.60, 0.45, 1.05]`
- A top-down camera at `(0.4, 0.0, 4.0)` with 640×640 resolution, 65° FOV
- A `brace_v1` data collection profile with shared autonomy blending (`u_exec = (1-γ) * u_human + γ * u_auto`)
- Motion generation via scripted, RMPflow, cuRobo, or LULA planners
- Robot base height at `0.8m` above ground

**Fallback approach**: A lightweight 2D planar Gymnasium environment (no physics engine) similar to the cursor control environments in the OLD notebooks. This is useful for rapid iteration and debugging of the BRACE pipeline without Isaac Sim overhead.

**2D Planar Constraint**: All motion is in the **XY plane** at a fixed Z height. The environment projects everything to 2D:
- End-effector position → (x, y) only
- Velocities → (vx, vy) only  
- Object/obstacle positions → (x, y) only
- Actions → 2D Cartesian velocity (vx, vy) + gripper command

**Observation space** (continuous `Box`, 2D planar):
- End-effector position (2) — XY only
- End-effector velocity (2) — XY only
- Gripper state (width or binary open/close) (1)
- For each candidate object: relative XY position from EE (2 × N_objects)
- For each obstacle: relative XY position from EE (2 × N_obstacles)
- Distance to nearest obstacle (1)
- Normalized progress toward each object (N_objects)

Total: `5 + 2*N_objects + 2*N_obstacles + 1 + N_objects` (e.g., with 3 objects + 3 obstacles = 5+6+6+1+3 = 21D).

**Action space**: 2D Cartesian end-effector velocity + gripper command = `Box(-1, 1, shape=(3,))`. The third dimension (gripper) is binary: open during reaching, close when near target.

**Episode logic**:
- Reset: randomize object positions on the table (XY plane), randomize obstacle positions, sample one object as the hidden true target.
- Success: gripper closes around the correct object (within distance + gripper threshold). Reward `+goal_bonus`.
- Collision: EE or arm link contacts an obstacle. Reward `-collision_penalty`, episode ends.
- Timeout: max steps exceeded (e.g., 200 steps). Episode truncated.

**Reward for expert training** (dense, goal-aware, **2D distances**):
```
R_expert = 3.0 * (d_{t-1} - d_t) / d_max      # d = 2D Euclidean distance to goal in XY plane
         - 0.8 * ||Δθ||²                        # θ = heading angle change in XY plane
         - 2.5 * exp(-min_obstacle_dist / d_safe) # obstacle distances in 2D XY plane
```

**Reward for arbitration training** (BRACE-specific, see Section 1.5, **all distances in 2D XY**):
```
R = -10.0 * 1_collision
    + 2.5 * γ * p_max * 1_near
    - 1.5 * γ * 1_far
    + 3.0 * p_max * (d_{t-1} - d_t)
    - 1.5 * γ²
    + 2.0 * log(p_true)
```

**Scenarios** (`scenarios.py`): Define difficulty levels for the curriculum:
1. Single object, no obstacles (basic reaching)
2. Single object, obstacles present (collision avoidance)
3. Single object, challenging obstacle placement (tight gaps)
4. 2–3 objects, obstacles present (goal ambiguity)
5. 3 objects, obstacles, full complexity

Each scenario defines: object count, object position ranges, obstacle count, obstacle position ranges, and any other relevant parameters.

**Wrappers** (`wrappers.py`):
- `FlattenObservation` if the base env uses dict obs (like Fetch)
- `NormalizeObservation` / `VecNormalize` for training stability
- `GoalMaskedObservation` for the expert (removes goal ambiguity — expert sees which object is the true target)

### 6.2 Bayesian Inference Module — `bayesian_inference.py`

Port the pattern from the old notebooks into a proper `nn.Module`:

```python
class BayesianGoalInference(nn.Module):
    def __init__(self, n_goals: int, initial_beta: float = 2.0,
                 initial_w_theta: float = 0.8, initial_w_dist: float = 0.2):
        # Learnable parameters via nn.Parameter + softplus
        self.raw_beta = nn.Parameter(torch.tensor(initial_beta))
        self.raw_w_theta = nn.Parameter(torch.tensor(initial_w_theta))
        self.raw_w_dist = nn.Parameter(torch.tensor(initial_w_dist))

    def step_likelihood(self, human_action, ee_position, goal_positions):
        # Compute angular deviation between human action direction and direction to each goal
        # Compute distance deviation |1 - |h| / h_opt|
        # cost = w_θ * θ_dev + w_d * d_dev
        # likelihood = exp(-β * cost), stabilized via log-sum-exp
        ...

    def update_belief(self, prior, likelihood, tau=1.0):
        # posterior ∝ prior * likelihood
        # Apply temperature τ to the softmax normalization
        # Apply EMA smoothing (α=0.85)
        ...

    def forward(self, human_action, ee_position, goal_positions, prior_belief):
        # Returns: (updated_belief, neg_log_prob_for_reinforce)
        ...
```

Support:
- **Pretraining**: `train_belief.py` loads trajectories with known goal labels, optimizes NLL `L = -log P(g* | X, H)`.
- **End-to-end**: `forward()` returns `neg_log_prob` for REINFORCE gradient from the arbitration reward.
- **Temperature annealing**: `τ` parameter that starts high (≥1.0) and decays.
- **Batch operation**: All computations should handle batch dimensions `(batch, ...)`.

### 6.3 Arbitration Policy — `arbitration_policy.py`

SB3-compatible custom policy:

```python
class GammaArbitrationPolicy(ActorCriticPolicy):
    """
    PPO-compatible policy that outputs scalar γ ∈ [0, 1].
    Input: concatenation of [state_features, belief_vector].
    Architecture: shared 256×256 ReLU trunk → actor head (1, tanh) + critic head (1).
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule,
                         net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                         activation_fn=nn.ReLU, **kwargs)

    def forward(self, obs, deterministic=False):
        # Extract features, run through mlp_extractor
        # Actor: tanh(raw_mean) → Normal distribution with learned log_std
        # Critic: value estimate
        # Return (action, value, log_prob)
        # action is in [-1, 1], mapped to γ = 0.5*(action+1) OUTSIDE the policy
        ...
```

Follow the exact pattern from `GammaMlpPolicy` in `PPO_SharedControl_MK8.ipynb` and `GammaBeliefPolicy` in `brace actor critic.ipynb`.

### 6.4 Expert Policy — `train_expert.py` + `expert_policy.py`

**Training** (`train_expert.py`):
- Algorithm: SAC (from stable-baselines3)
- Environment: The reach-and-grasp env with **full goal observability** (expert knows the true target)
- Hyperparameters (from old Reacher expert code):
  - `learning_rate=3e-4`
  - `buffer_size=400_000`
  - `batch_size=1024`
  - `tau=0.005`
  - `gamma=0.99`
  - `train_freq=64`
  - `gradient_steps=64`
  - `learning_starts=10_000`
  - `net_arch=[256, 256, 256]`
  - Total timesteps: `1_000_000` (adjustable)
- Reward: `R_expert` as defined in Section 1.6
- Saves: `expert_sac.zip`

**Wrapper** (`expert_policy.py`):
- Loads the trained SAC model.
- Given full state (including true target), returns the optimal Cartesian velocity action.
- Frozen during arbitration training (no gradient updates).

### 6.5 Simulated Human — `simulated_human.py`

Port from the paper's appendix description and `Controller_MK14.ipynb` noise patterns:

```python
class SimulatedHuman:
    def __init__(self, noise_amplitude=0.032, ar_coeff=0.5, spectral_slope=-1.59):
        ...

    def plan_trajectory(self, current_pos, goal_pos, obstacle_positions):
        # Minimum-jerk trajectory toward goal
        # Add via-points using potential field near obstacles
        ...

    def get_action(self, current_pos, goal_pos, obstacle_positions):
        # Get deterministic direction from trajectory plan
        # Add AR(1) pink noise: y[n] = 0.5*y[n-1] + 0.5*x[n]
        # Scale noise by amplitude (3.2% of trajectory length)
        # Return noisy Cartesian velocity command
        ...
```

### 6.6 Arbitration Training — `train_arbitration.py`

The main training script implementing Algorithm 1 from the paper:

```
1. Load pretrained expert (frozen SAC)
2. Load pretrained (or initialize) Bayesian inference module
3. Initialize arbitration policy (PPO with GammaArbitrationPolicy)
4. For each curriculum stage:
   a. Configure environment difficulty via scenarios.py
   b. For each episode:
      - Reset env, initialize uniform belief b_0
      - For each step t:
        · Observe state s_t
        · Simulated human produces h_t (noisy action toward true goal)
        · Update belief: b_t = BayesianGoalInference(h_t, s_t, goals, b_{t-1})
        · Fuse observation: obs_fused = concat([state_features, b_t])
        · Compute γ_t, V(s_t) from arbitration policy
        · Blend: a_t = (1 - γ_t) * h_t + γ_t * w_t (expert action)
        · Execute a_t, observe r_t, s_{t+1}
        · Store transition
      - Compute advantages (GAE) and returns
      - PPO update on arbitration policy parameters θ
      - Compute L_total = α·L_RL + (1-α)·L_supervised on inference module parameters φ
        · L_supervised = -log P_φ(g* | X, H)
        · L_RL via REINFORCE with advantage baseline
      - Apply gradient clipping on φ (L2 norm cap)
      - Apply temperature annealing on belief τ
      - Apply confidence scaling: α_update = min(1.0, p_max / 0.8)
   c. Check curriculum advancement criteria
   d. Log metrics to TensorBoard
5. Save: arbitration_policy.zip, bayesian_inference.pt
```

### 6.7 Curriculum Manager — `curriculum.py`

```python
class CurriculumManager:
    """Manages 5-stage curriculum with automatic advancement."""

    stages = [
        Stage(name="basic_reaching", n_objects=1, n_obstacles=0,
              min_episodes=100, success_threshold=0.80),
        Stage(name="collision_avoidance", n_objects=1, n_obstacles=3,
              min_episodes=200, success_threshold=0.75, max_collision_rate=0.15),
        Stage(name="challenging_obstacles", n_objects=1, n_obstacles=4,
              min_episodes=300, success_threshold=0.70),
        Stage(name="goal_ambiguity", n_objects=3, n_obstacles=3,
              min_episodes=400, success_threshold=0.65),
        Stage(name="full_complexity", n_objects=3, n_obstacles=4,
              plateau_window=200),
    ]

    def should_advance(self, metrics) -> bool:
        ...
```

### 6.8 Rewards — `rewards.py`

Implement both reward functions as callable classes:

```python
class ExpertReward:
    """Dense reward for expert (SAC) training. Goal-aware."""
    # R = 3.0*(d_{t-1}-d_t)/d_max - 0.8*||Δθ||² - 2.5*exp(-d_obs_min/d_safe)

class ArbitrationReward:
    """BRACE reward for arbitration (PPO) training. Belief-aware."""
    # R = -w_coll*1_coll + w_prox*γ*p_max*1_near - w_far*γ*1_far
    #     + w_prog*p_max*(d_{t-1}-d_t) - w_auto*γ² + w_goal*log(p_true)
    # Weights: w_coll=10, w_prox=2.5, w_far=1.5, w_prog=3.0, w_auto=1.5, w_goal=2.0
```

### 6.9 Evaluation — `evaluate.py`

Metrics to compute (matching the paper's Experiment 3 table):
- **Success rate** (%): gripper grasps the correct object
- **Time to grasp** (seconds or steps)
- **Collision count** per episode
- **Belief accuracy**: how often the highest-probability goal matches the true goal, measured at 25%, 50%, 75% of path completion
- **γ statistics**: mean γ, γ in constrained vs. unconstrained regions
- **Belief entropy** over time

Compare BRACE against baselines: (1) no assistance (human only), (2) full autonomy (expert only, random goal), (3) fixed γ, (4) IDA-style baseline, (5) BRACE.

### 6.10 ROS 1 Interface — `ros_interface/`

**IMPORTANT**: The Kinova Gen3 (2nd generation) robot uses **ROS 1** (Melodic/Noetic) with the `ros_kortex` driver package. All ROS nodes must use **rospy**, NOT rclpy (ROS 2). A separate bridge between Isaac Sim and the real robot already exists and has been validated.

**`brace_node.py`**: A ROS 1 node for real-time BRACE inference:

```python
import rospy
from geometry_msgs.msg import Twist, TwistStamped, PoseArray, PoseStamped
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Joy

class BraceNode:
    def __init__(self):
        rospy.init_node('brace_node')
        # Load trained models: expert_sac.zip, bayesian_inference.pt, arbitration_policy.zip
        # Subscribe to:
        #   /joy (sensor_msgs/Joy) - DualSense PS5 controller input
        #   /object_positions (geometry_msgs/PoseArray) - from YOLO detection pipeline
        #   /obstacle_positions (geometry_msgs/PoseArray) - detected obstacle poses
        #   /ee_state (geometry_msgs/PoseStamped) - current EE pose from kortex_driver
        # Publish to:
        #   /my_gen3/in/cartesian_velocity (kortex_driver_msgs/TwistCommand) - Cartesian velocity
        #   /brace/belief (std_msgs/Float32MultiArray) - current belief distribution (for viz)
        #   /brace/gamma (std_msgs/Float32) - current assistance level (for viz)
        ...

    def joy_callback(self, msg):
        # Parse DualSense PS5 controller input:
        #   Left stick X (axes[0]) → vx, Left stick Y (axes[1]) → vy
        #   L2 trigger (axes[4]) → decrease gamma (manual mode)
        #   R2 trigger (axes[5]) → increase gamma (manual mode)
        #   Square button → reset position
        # Apply deadzone filtering (0.1 threshold)
        ...

    def inference_callback(self, timer_event):
        # 1. Get current state from subscriptions (project to 2D XY plane)
        # 2. Get human input from DualSense controller (XY velocity only)
        # 3. Get expert action (from loaded SAC, given state + highest-belief goal)
        # 4. Update belief
        # 5. Compute γ from arbitration policy
        # 6. Blend: cmd = (1-γ)*human + γ*expert (in 2D XY plane)
        # 7. Publish blended Cartesian velocity command to Kinova
        ...
```

Target inference rate: ~27 Hz (36ms per cycle, matching paper benchmarks). The Kinova Gen3 arm operates at 20 Hz control frequency, so BRACE at 27 Hz will not be a bottleneck.

**`kinova_bridge.py`**: Translates BRACE's 2D Cartesian velocity output to the `ros_kortex` driver interface. The Kinova Gen3 `ros_kortex` package exposes Cartesian velocity control via action servers and topic-based twist commands. The bridge must:
- Convert 2D (vx, vy) to 3D Cartesian velocity (vx, vy, 0) at fixed Z height
- Handle gripper open/close via the `ros_kortex` gripper action server
- Enforce workspace bounds and safety limits
- Interface with the existing Isaac Sim ↔ ROS 1 bridge for sim-to-real transfer

**`dualsense_interface.py`**: Dedicated DualSense PS5 controller handler (see Section 12 for details). Reads from `/joy` topic (published by `joy_node` ROS package) and provides clean velocity commands with deadzone filtering.

### 6.11 Configuration Files — `configs/`

All hyperparameters must be in YAML, not hardcoded. Example `arbitration.yaml`:

```yaml
ppo:
  learning_rate: 3.0e-4
  lr_schedule: cosine
  n_steps: 1024
  batch_size: 1024
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  net_arch:
    pi: [256, 256]
    vf: [256, 256]
  activation: relu

bayesian:
  initial_beta: 2.0
  initial_w_theta: 0.8
  initial_w_dist: 0.2
  ema_alpha: 0.85
  initial_temperature: 1.0
  temperature_decay: 0.999
  confidence_threshold: 0.8
  lr: 1.0e-3
  gradient_clip_norm: 1.0

reward:
  w_collision: 10.0
  w_proximity: 2.5
  w_far: 1.5
  w_progress: 3.0
  w_autonomy: 1.5
  w_goal: 2.0

training:
  total_timesteps: 2_000_000
  alpha_initial: 0.3
  alpha_final: 0.9
  alpha_anneal_steps: 500_000
  log_dir: ./logs/arbitration
  save_dir: ./checkpoints
  save_freq: 50_000
  eval_freq: 10_000
  eval_episodes: 50

curriculum:
  stages:
    - name: basic_reaching
      n_objects: 1
      n_obstacles: 0
      min_episodes: 100
      success_threshold: 0.80
    - name: collision_avoidance
      n_objects: 1
      n_obstacles: 3
      min_episodes: 200
      success_threshold: 0.75
      max_collision_rate: 0.15
    - name: challenging_obstacles
      n_objects: 1
      n_obstacles: 4
      min_episodes: 300
      success_threshold: 0.70
    - name: goal_ambiguity
      n_objects: 3
      n_obstacles: 3
      min_episodes: 400
      success_threshold: 0.65
    - name: full_complexity
      n_objects: 3
      n_obstacles: 4
      plateau_window: 200
```

### 6.12 Dependencies — `requirements.txt`

```
gymnasium
gymnasium-robotics
stable-baselines3
torch
numpy
scipy
pyyaml
tensorboard
matplotlib
pygame
```

**GPU Training**: All training scripts MUST support GPU acceleration. The target hardware is an **NVIDIA RTX 4090 laptop GPU**. Ensure:
- All PyTorch models use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Training configs include a `device` parameter (default: `"auto"` which selects GPU if available)
- Stable-Baselines3 PPO/SAC use `device="cuda"` when available
- Batch sizes are tuned for the RTX 4090's 16GB VRAM
- Mixed precision training (torch.cuda.amp) should be an option for faster training
- The RTX 4090 should reduce the ~7 hour training time (reported on RTX 3080) significantly

**Isaac Sim / Isaac Lab dependencies** (installed via Isaac Sim environment, NOT pip):
```
# isaacsim (NVIDIA Isaac Sim)
# isaaclab (Isaac Lab framework)
# isaaclab_assets (robot asset configs)
# omni.isaac (Omniverse Isaac extensions)
```

The kinova-isaac codebase (`/home/kye/Desktop/Depo/Code/kinova-isaac`) has its own `pyproject.toml` with:
- `requires-python >= 3.11`
- Packages: `controllers`, `data_collection`, `environments`, `kinova`, `motion_generation`, `utilities`, `copilot_demo`
- No pinned runtime deps — those come from the Isaac Sim environment

Optional (for ROS 1 deployment, not in pip):
```
# rospy (installed via ROS workspace, not pip)
# geometry_msgs, std_msgs, sensor_msgs (ROS 1 message packages)
# ros_kortex (Kinova Gen3 ROS 1 driver)
# joy (ROS joystick driver for DualSense)
```

---

## 7. Implementation Guidelines

1. **Read the old notebooks thoroughly** before writing new code. Extract working patterns — don't reinvent what already works.
2. **Modular `.py` files**, NOT notebooks. Every class should be importable.
3. **Type hints** on all function signatures.
4. **Docstrings** on all public classes and functions.
5. **Gymnasium-compliant**: Register the environment with `gymnasium.register()`.
6. **SB3-compatible**: The arbitration policy must work seamlessly with `stable_baselines3.PPO`.
7. **CLI-runnable**: All training scripts accept config file path as command-line argument:
   ```bash
   python -m brace_kinova.training.train_expert --config configs/expert.yaml
   python -m brace_kinova.training.train_belief --config configs/belief.yaml
   python -m brace_kinova.training.train_arbitration --config configs/arbitration.yaml
   ```
8. **Standardized save/load**: Expert → `expert_sac.zip`, Belief → `bayesian_inference.pt`, Arbitration → `arbitration_policy.zip`, VecNormalize stats → `vecnormalize.pkl`.
9. **TensorBoard logging**: All training scripts log to TensorBoard.
10. **Reproducible**: Set seeds everywhere; configs include seed parameter.
11. **ROS 1 interface is optional**: It should import from `brace_kinova.models` but not require ROS to be installed for training.
12. **GPU training by default**: All training scripts must auto-detect and use GPU (NVIDIA RTX 4090 laptop). Use `torch.device("cuda")` when available. Include `--device` CLI argument.
13. **2D planar**: All environment observations and actions are 2D (XY plane). No Z-axis movement during reaching. Gripper descent is a separate terminal action.
14. **Isaac Sim integration**: The training environment should interface with the kinova-isaac codebase at `/home/kye/Desktop/Depo/Code/kinova-isaac`. Use the existing `reach_to_grasp_VLA` scene design and top-down camera configuration.

---

## 8. Training Pipeline (Execution Order)

```
Step 1: Train Expert
   python -m brace_kinova.training.train_expert --config configs/expert.yaml
   → Produces: checkpoints/expert_sac.zip

Step 2: Pretrain Belief Module
   python -m brace_kinova.training.train_belief --config configs/belief.yaml
   → Produces: checkpoints/bayesian_inference.pt

Step 3: Train BRACE Arbitration (with curriculum)
   python -m brace_kinova.training.train_arbitration --config configs/arbitration.yaml
   → Produces: checkpoints/arbitration_policy.zip, checkpoints/bayesian_inference_finetuned.pt

Step 4: Evaluate
   python -m brace_kinova.evaluation.evaluate --config configs/arbitration.yaml
   → Produces: evaluation metrics, comparison tables, plots

Step 5: Deploy (ROS 1 via Isaac Sim bridge)
   rosrun brace_kinova brace_node.py _config_path:=configs/arbitration.yaml
```

---

## 9. Additional Notes

### 9.1 Training Hardware & Timing
- The paper mentions **~7 hours** GPU training time for full curriculum on RTX 3080. With the **RTX 4090 laptop GPU**, this should be significantly faster (estimated ~4-5 hours). The 2D planar constraint further reduces training time since observation/action spaces are smaller.
- Bayesian pretrain takes **~45 minutes** on RTX 3080; expect ~25-30 minutes on RTX 4090.
- Inference latency target: **36ms** per cycle (27 Hz) on desktop GPU, **49ms** on laptop CPU.
- The paper reports Bayesian inference takes ~3ms and policy forward pass ~12ms — the rest is environment stepping.
- The belief module's computational cost scales linearly with number of goals.

### 9.2 Simulated Human Validation
- For the simulated human during training, noise amplitude of 3.2% with AR(1) coefficients [0.5, 0.5] has been validated against real human data (798 trajectories, 23 participants from EEG decoder cursor control study).
- The simulated human model uses minimum-jerk polynomial regression for deterministic trajectory + potential field via-points near obstacles + AR(1) pink noise filter.
- Cross-modal validation showed comparable normalized jerk statistics (RMSE < 0.14) between EEG-based and joystick/force-sensor trajectories.
- Sensitivity analysis: BRACE maintained above 92% of optimal performance across spectral slope [-1.2, -1.8] and amplitude [1.5%, 5.0%].

### 9.3 2D Planar Workspace Rationale
- The 2D constraint dramatically simplifies the problem: observation space drops from ~29D to ~21D, action space from 4D to 3D.
- This matches the paper's existing cursor control experiment (Experiment 1) which was also 2D planar and produced the strongest results.
- The Kinova Gen3 arm's 7-DOF redundancy makes it trivial to constrain the EE to a fixed-height XY plane — the inverse kinematics have many solutions at any given XY point.
- The top-down camera in the kinova-isaac environment provides a natural 2D view of the workspace, matching the observation space.

### 9.4 DualSense PS5 Controller
- The DualSense controller is already validated in the paper's user study (N=12 participants, Experiment 1: Planar Cursor Control).
- The paper explicitly states: "participants performed a goal-directed cursor control task using a DualSense controller" (Section 4.1).
- The controller was also used in the force-sensor comparison study (Appendix G, Table 9) showing input-modality agnosticism.

### 9.5 Object Detection (Future Integration)
- A separate YOLO-based model for object segmentation and detection has been trained.
- For simulation training, ground-truth object positions come from Isaac Sim directly.
- For real-world deployment, the YOLO model will provide object positions via a ROS 1 topic.
- The top-down camera in kinova-isaac is suitable for both training (ground truth) and inference (YOLO).

---

## 10. kinova-isaac Codebase Reference

The existing kinova-isaac codebase at `/home/kye/Desktop/Depo/Code/kinova-isaac` provides the simulation environment and robot configuration. This section documents everything relevant for building the BRACE training pipeline.

### 10.1 Directory Structure

```
kinova-isaac/
├── environments/
│   ├── reach_to_grasp/          # Base reach-and-grasp env (no camera)
│   │   ├── config.py            # SceneConfig, CameraConfig (viewport only)
│   │   ├── demo.py              # Standalone demo with YCB objects
│   │   └── utils.py             # design_scene(): ground plane, dome light, table, robot
│   ├── reach_to_grasp_VLA/      # VLA-extended env (with top-down camera)
│   │   ├── config.py            # + TopDownCameraConfig: pos=(0.4,0,4.0), target=(0.4,0,0.8), 640x640, 65° FOV
│   │   ├── demo_VLA.py          # Demo with top-down camera creation
│   │   └── utils.py             # Same design_scene() as base
│   ├── utils/
│   │   ├── object_loader.py     # USD/YCB object loading
│   │   ├── physix.py            # Physics config utilities
│   │   └── camera/
│   │       └── topdown.py       # TopDownCamera: creates USD Camera prim, sets position/FOV
│   └── README.md
├── data_collection/
│   ├── profiles/
│   │   ├── brace_v1.py          # BRACE data collection profile (shared autonomy blend)
│   │   ├── vla_v0.py / vla_v1.py / ticks_v0.py
│   │   └── registry.py          # Profile registry: ticks_v0, vla_v0, vla_v1, brace_v1
│   ├── core/
│   │   ├── input_mux.py         # CommandMuxInputProvider, SampleAndHoldInputProvider, SharedAutonomyBlendInputProvider
│   │   ├── schemas.py           # Pose, DetectedObject, to_json
│   │   ├── objects.py           # ObjectsTracker (PhysX rigid-body views or USD Xform)
│   │   └── logger.py            # SessionLogWriter (per-episode tick-level JSON logs)
│   ├── engine/
│   │   ├── episode_runner.py    # Episode lifecycle management
│   │   └── waypoints.py         # Waypoint definitions
│   ├── config.py                # RunConfig dataclasses (episode, task, planner, objects, logging)
│   ├── collect_data.py          # Main entry: python -m data_collection.collect_data --profile ...
│   └── README.md
├── motion_generation/
│   ├── mogen.py                 # MotionGenerationAgent: label→prim, GraspPoseProvider, world↔base transforms
│   ├── planners/
│   │   ├── factory.py           # create_planner()
│   │   ├── base.py              # PlannerContext(base_frame, ee_link, urdf_path, config_dir)
│   │   ├── scripted.py          # Scripted planner (simple waypoints)
│   │   ├── rmpflow.py           # RMPflow planner
│   │   ├── lula.py              # LULA planner
│   │   └── curobo_v2.py         # cuRobo V2 planner
│   ├── grasp_estimation/
│   │   ├── obb.py               # Oriented bounding box grasp estimation
│   │   └── replicator.py        # Replicator-based grasp estimation
│   └── README.md
├── copilot_demo/
│   └── copilot_demo/
│       ├── demo_isaacsim.py     # Full demo: reach_to_grasp_VLA scene + planners + controller
│       ├── backends.py          # OracleBackend, HFBackend
│       ├── executor.py          # ActionExecutor
│       └── extractor.py         # InputExtractor
├── scripts/
│   └── cli.py                   # CLI argument definitions, default EE link: j2n6s300_end_effector
├── grasp-vla/
│   ├── rollout_xvla_isaac.py    # VLA rollout in Isaac Sim
│   └── BRACE/BRACE_ACM_THRI-2.pdf
└── pyproject.toml               # name=kinova-isaac, requires-python>=3.11
```

### 10.2 Robot Configuration

**IMPORTANT**: The kinova-isaac codebase currently uses **Kinova Jaco2 J2N6S300** (6-DOF), NOT the Gen3 (7-DOF). Key details:

- **Asset**: `isaaclab_assets.KINOVA_JACO2_N6S300_CFG` as `BASE_ROBOT_CFG`
- **Joints**: `j2n6s300_joint_1` through `j2n6s300_joint_6`, plus finger joints (`j2n6s300_joint_finger_*`, `j2n6s300_joint_finger_tip_*`)
- **Joint limits** (approximate, in radians):
  - Joint 2: `[0.820, 5.463]`
  - Joint 3: `[0.332, 5.952]`
  - Finger joints: up to ~1.51 / 2.0 rad
- **Prim path**: `/World/Origin1/Robot`
- **Robot base height**: `0.8m` above ground
- **Default EE link**: `j2n6s300_end_effector`
- **URDF** (for cuRobo): `motion_generation/planners/planners_config/cuRobo/kinovaJacoJ2N6S300.urdf`

**For the 2D planar BRACE experiment**: Since we only need XY planar motion, the Jaco2 model can serve as a reasonable training stand-in. The sim-to-real transfer targets the Gen3, but the 2D constraint means the specific arm kinematics are less critical — we primarily need accurate XY end-effector positioning and collision geometry.

### 10.3 Scene Configuration

The `design_scene()` function creates:
- **Ground plane**: `GroundPlaneCfg` at `/World/defaultGroundPlane`
- **Dome light**: intensity `2000`, color `(0.75, 0.75, 0.75)` at `/World/Light`
- **Origin**: `/World/Origin1` with translation from `define_origins` (grid layout, default single origin)
- **Table**: Thorlabs table USD from Nucleus, scale `(1.5, 2.0, 1.0)`, translation `(0, 0, 0.8)` (table surface at ~0.8m)
- **Robot**: Articulation from `KINOVA_JACO2_N6S300_CFG` with configurable default joint positions

**Object spawn bounds** (in parent frame): min `[0.2, -0.3, 0.9]`, max `[0.60, 0.45, 1.05]`
**Controller workspace**: min `(0.20, -0.45, 0.01)`, max `(0.6, 0.45, 0.35)` (base frame)

### 10.4 Top-Down Camera

The `reach_to_grasp_VLA` config defines a `TopDownCameraConfig`:
- **Prim path**: `/World/Origin1/TopDownCamera`
- **Position**: `(0.4, 0.0, 4.0)` — directly above the workspace center
- **Target**: `(0.4, 0.0, 0.8)` — pointing down at table surface
- **Resolution**: `640 × 640`
- **FOV**: `65.0°`

Implementation in `environments/utils/camera/topdown.py` creates a USD Camera prim, sets translate + zero RotateXYZ, and maps FOV to focal length.

### 10.5 BRACE Data Collection Profile (`brace_v1`)

The `brace_v1` profile implements shared autonomy data collection:

- **Environment**: `reach_to_grasp_VLA` by default
- **Expert**: `_BraceExpert` — scripted stages: `init_open → reach → approach → grasp → lift → done` with Cartesian deltas + gripper in **base frame** (7D: translation + gripper)
- **Blending**: `SharedAutonomyBlendInputProvider`: `u_exec = (1-γ) * u_human + γ * u_auto`
- **Gamma modes**: `--gamma-mode fixed|stage` with per-stage CLI flags: `--gamma-reach`, `--gamma-approach`, `--gamma-grasp`, `--gamma-lift`
- **Human input**: `SampleAndHoldInputProvider` + `Se3KeyboardInput` if GUI and `--control keyboard`; otherwise idle. **No gamepad support yet** — needs to be added for DualSense.
- **Camera logging**: `--enable_cameras` creates top-down camera + `isaaclab.sensors.Camera`, logs RGB images under `session_brace_*/episode_*/images/`
- **Session logs**: `SessionLogWriter` per episode under `logs/data_collection/session_brace_<timestamp>/episode_XXXX/`; ticks include `shared_autonomy` (human/auto/exec/gamma) and `task` (goal id/label, stage)

### 10.6 Input Handling (`core/input_mux.py`)

Key classes:
- **`CommandMuxInputProvider`**: Routes commands from multiple sources
- **`SampleAndHoldInputProvider`**: Samples at fixed rate, holds between samples
- **`SharedAutonomyBlendInputProvider`**: Implements `u_exec = (1-γ) * u_human + γ * u_auto` — this is the core BRACE blending

**Current limitation**: Only keyboard input (`Se3KeyboardInput`) is supported. DualSense/gamepad input needs to be added as a new `InputProvider` subclass.

---

## 11. Reviewer Required Revisions (ACM THRI — THRI-2025-0214)

The paper received a **Major Revision** decision from ACM Transactions on Human-Robot Interaction. **Deadline: May 30, 2026**. Three reviewers provided feedback. This section summarizes ALL reviewer concerns and maps them to required implementation/experimental work. The new Experiment 4 (Kinova real-world deployment) is the primary response to several critical concerns.

### 11.1 Associate Editor Summary (Consensus)

The AE identified 6 major weaknesses:

1. **Motivation for end-to-end formulation not established**: Paper doesn't provide clear empirical or theoretical evidence showing why modular approaches fail. Need concrete examples of where sequential pipelines break down.

2. **Methodological ambiguities**: Unclear how belief representations are incorporated into action selection, how training data is collected with human-system interaction, and how components integrate. Figures don't clearly illustrate data flow.

3. **Theory-implementation mismatch**: Paper presents utility-based analysis but trains with PPO + handcrafted reward. Connection between theoretical claims and learned policy is unclear.

4. **Limited experimental design**: Human study is limited (single 2D cursor task, N=12), user agency/trust not directly measured, statistical analysis missing assumption checks (normality, sphericity for ANOVA), inconsistent baselines across experiments.

5. **Generalizability concerns**: Reliance on simulated environments and simple tasks limits empirical validation. **This is directly addressed by the new Kinova real-world experiment.**

6. **Insufficient literature positioning**: Related work is not comprehensive enough, heavy self-citations, limited engagement with uncertainty-aware assistance literature.

### 11.2 Reviewer 1 (Recommendation: Reject)

**Critical points requiring action:**

- **Introduction**: "The core challenge" sentence is underspecified — needs concrete examples of where/how shared autonomy fails in practice.
- **Research gap**: No concrete evidence that modular designs fail; assumption that end-to-end is better is unjustified.
- **Human factors missing**: Introduction is control/optimization-centric; no discussion of user agency, cognitive load, or the user study.
- **Citation numbering**: First citation starts at [19] — fix ordering.
- **Related work**: Too many self-citations (many are arXiv preprints); need broader shared autonomy literature coverage (information-theoretic assistance, entropy-regularized arbitration, risk-sensitive blending).
- **Methodology clarity**: Fig. 1 and Fig. 2 don't clearly show data flow between variables and modules.
- **Utility function**: Eq. 9 is a heuristic reward, not a principled utility over outcomes. Paper calls it "utility" but it behaves like handcrafted reward.
- **PPO vs utility**: Section 3.2 presents utility as the optimized objective, but actual training uses PPO on a separate handcrafted reward. No formal connection established.
- **Appendix A**: Assumptions (expert efficiency, concavity) are chosen to make the conclusion work, not derived from problem structure.

**Results criticisms:**
- **NASA-TLX**: Paper uses non-standard items (Satisfaction, Ease, etc. instead of Mental Demand, Physical Demand, etc.). **Must switch to standard NASA-TLX** (see TLX Scale PDF).
- **Human study scope**: Why only for Experiment 1 (cursor)? Need human studies for other experiments too. N=12 is too small; standard is ≥30.
- **User agency**: Claimed but not measured. Recommend checking "The Sense of Agency in Assistive Robotics Using Shared Autonomy" by Collier et al. (2025, HRI).
- **Statistics**: ANOVA needs normality checks (Shapiro-Wilk), sphericity checks (Mauchly's test + Greenhouse-Geisser correction), and post-hoc pairwise comparisons with multiple-comparison corrections.
- **All experiments are simulated**: "not convinced that user-study results obtained in the absence of a real-robot deployment are sufficiently informative."
- **Inconsistent baselines**: 4 baselines for cursor, 3 for Reacher, 2 for pick-and-place — need consistent baselines across all experiments.

### 11.3 Reviewer 2 (Recommendation: Minor Revision)

**Positive feedback**: "meaningful, well-motivated contribution", "robust experimental evaluation"

**Suggestions requiring action:**
- Add dedicated **formalization/problem-definition section** early in the paper.
- Clarify **transparency cues** — were they controlled across conditions? BRACE uniquely shows inferred goal + assistance level; baselines may not have comparable feedback. This confounds transparency vs. performance.
- **Context definition**: Method name emphasizes "context encoding" but context isn't introduced as a distinct component. Need explicit definition + ablation isolating context-related inputs.
- **Trust claims**: Soften or qualify — trust is not directly measured. Reported metrics (confidence, satisfaction, NASA-TLX) are related but distinct constructs.
- **Notation**: Introduce notation before use; improve section transitions.
- **Discrete goal assumption**: Acknowledge limitation of predefined discrete goal set; discuss extensions to open-ended settings.
- **Reproducibility**: Include supplementary materials, code repositories.

### 11.4 Reviewer 3 (Recommendation: Major Revision)

**Methodology questions:**
- How are "raw goal probabilities" used to compute executed expert action? Is it belief-weighted mixture of goal-conditioned expert actions or selecting a single goal's expert?
- How is training data collected for arbitration learning? Does it reflect humans adapting their input based on assistance (closed-loop coupling)?
- Does the simulated human incorporate/expect assistance, or is it a fixed model? Discuss implications for sim-to-human validity.

**Experiment questions:**
- Section 4.1: How is environment randomized across 24 trials? Are goals fixed while obstacles vary, or both change?
- Section 4.1.2: Interface displays inferred goal — was this feedback present in training data? If not, train-test mismatch.
- Section 4.3: Why does IDA trigger multiple interventions early in episodes?

**Figure improvements needed:**
- Fig. 4: Unclear start/goal positions. Add description of gray circles and labeled yellow circles.
- Fig. 8(b)(c): Enlarge x-axis/y-axis labels.
- Fig. 8(d)(e)(f): Colors don't match goals; unclear which circle is graspable object; no visible trajectory toward object then bins.

**Typos**: Page 2 Line 14 ("I" → "i"), Page 11 Line 41 ("An a priori"), Page 12 Line 42 (repeating sentence), Page 13 Lines 8-9 (clarify 0.28/0.74 are gamma values).

### 11.5 How Experiment 4 Addresses Reviewer Concerns

The new Kinova real-world experiment directly addresses these critical reviewer points:

| Reviewer Concern | How Experiment 4 Addresses It |
|---|---|
| R1: "All experiments are simulated" | Real-world Kinova Gen3 deployment with physical objects and obstacles |
| R1: "Relatively simple tasks" | Real robotic arm reaching/grasping is substantially more complex than 2D cursor |
| AE: "Generalizability concerns" | Demonstrates BRACE transfers from simulation to real hardware |
| R1: "Absence of real-robot deployment" | Direct real-robot deployment with DualSense controller |
| R2: "Cursor task limits generalization" | Extends human-in-the-loop validation to physical manipulation |
| R3: "Sim-to-human validity" | Real human operators with physical robot, validating simulated human model |
| R1: "N=12 too small" | Plan for larger sample size (≥30) in new experiments |
| R1: "Inconsistent baselines" | Use consistent baselines across ALL experiments including Experiment 4 |

**Additional paper changes needed** (not implementation, but important context):
- Rewrite introduction with human factors perspective (agency, cognitive load)
- Add formal problem definition section
- Fix citation numbering
- Expand related work with broader shared autonomy literature
- Switch to standard NASA-TLX items
- Add normality/sphericity checks to statistical analysis
- Clarify transparency cues across conditions
- Soften trust claims
- Fix all figures per R3 feedback
- Acknowledge discrete goal limitation
- Release code repository for reproducibility

---

## 12. DualSense PS5 Controller Integration

The DualSense wireless controller is the primary human input device for both simulation user studies and real-world deployment. This section documents the existing implementation patterns from the OLD notebooks (Controller_MK13/MK14) and specifies how to adapt them for the new codebase.

### 12.1 Existing Implementation (from Controller_MK14.ipynb)

The OLD code uses **pygame** for DualSense input. Key patterns:

**Initialization:**
```python
import pygame
pygame.init()
pygame.joystick.init()

joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Joystick initialized:", joystick.get_name())  # "DualSense Wireless Controller"

AXIS_L2 = 4  # Left trigger axis index
AXIS_R2 = 5  # Right trigger axis index
```

**Input reading (per-frame in game loop):**
```python
dx, dy = 0.0, 0.0
if joystick:
    axis_0 = joystick.get_axis(0)  # Left stick X
    axis_1 = joystick.get_axis(1)  # Left stick Y
    deadzone = 0.1
    if abs(axis_0) > deadzone or abs(axis_1) > deadzone:
        dx = axis_0
        dy = axis_1
    else:
        dx, dy = 0.0, 0.0

    # L2/R2 for manual gamma adjustment
    l2_val = joystick.get_axis(AXIS_L2)
    r2_val = joystick.get_axis(AXIS_R2)
    if gamma_mode == "manual":
        if l2_val > 0.1:
            gamma = max(0.0, gamma - 0.003)
        if r2_val > 0.1:
            gamma = min(1.0, gamma + 0.003)

# Apply deadzone and scale
if abs(dx) < 0.1 and abs(dy) < 0.1:
    dx, dy = 0.0, 0.0
dx *= MAX_SPEED
dy *= MAX_SPEED
human_input = [dx, dy]
```

**Control modes (from MK14, 5 conditions matching the user study):**
```python
gamma_modes = [0.0, 0.5, 1.0, "manual", "ai"]
# 0.0  = Mode 1: No assistance (human only, γ=0)
# 0.5  = Mode 2: IDA (binary intervention)
# 1.0  = Mode 3: Reddy's DQN (continuous blending with base γ=0.25)
# "manual" = Mode 4: User-controlled gamma (L2/R2 triggers)
# "ai"  = Mode 5: BRACE (AI-predicted gamma from PPO model)
```

**Button mappings:**
- Left analog stick (axes 0, 1): XY movement direction
- L2 trigger (axis 4): Decrease gamma (manual mode only)
- R2 trigger (axis 5): Increase gamma (manual mode only)
- Square button: Reset position
- Keyboard fallback: Arrow keys for movement

### 12.2 GammaPredictor (from MK14)

The MK14 notebook loads a pre-trained PPO model for gamma prediction:

```python
class GammaPredictor:
    def __init__(self, model_path="gamma_ppo_model.zip"):
        self.model = PPO.load(model_path)
        self.max_dist = np.sqrt(GAME_AREA_SIZE[0]**2 + GAME_AREA_SIZE[1]**2)

    def prepare_observation(self, dot_pos, target_pos, human_input):
        # 10D observation vector:
        # [dot_pos(2), human_dir(2), target_pos(2), perfect_dir(2), normalized_dist(1), obs_dist_ratio(1)]
        dot_pos = np.array(dot_pos, dtype=np.float32)
        target_pos = np.array(target_pos, dtype=np.float32)
        to_target = target_pos - dot_pos
        dist = np.linalg.norm(to_target)
        perfect_dir = to_target / dist if dist > 0 else np.zeros(2)
        h_mag = np.linalg.norm(human_input)
        human_dir = human_input / h_mag if h_mag > 0 else np.zeros(2)
        obs = np.concatenate([dot_pos, human_dir, target_pos, perfect_dir,
                              [dist / self.max_dist], [obs_dist_ratio]])
        return obs.astype(np.float32)

    def predict_gamma(self, dot_pos, target_pos, human_input):
        obs = self.prepare_observation(dot_pos, target_pos, human_input)
        action, _ = self.model.predict(obs[np.newaxis, :], deterministic=True)
        return float(action[0])
```

### 12.3 Noise Generation (from MK14)

Pink noise via scipy for simulated human imperfection:

```python
from scipy.signal import lfilter

def pink_noise():
    """Pink noise (1/f) via AR(1) filter."""
    a = [1, -0.95]
    return lfilter([1], a, np.random.normal(0, 290, size=1))[0]

NOISE_FUNCTION = pink_noise
NOISE_MAGNITUDE = 2.5
```

### 12.4 Expert Direction Computation (from MK14)

Potential-field-based obstacle avoidance for the expert policy:

```python
def compute_perfect_direction(dot_pos, goal_pos, obstacles):
    """Goal-directed + obstacle repulsion (potential field)."""
    gx = goal_pos[0] - dot_pos[0]
    gy = goal_pos[1] - dot_pos[1]
    goal_dist = math.hypot(gx, gy)
    goal_dir = [gx / goal_dist, gy / goal_dist] if goal_dist > 1e-6 else [0, 0]

    repulse_x, repulse_y = 0.0, 0.0
    repulsion_radius = 27 * SCALING_FACTOR
    repulsion_gain = 30000.0

    for obs in obstacles:
        dx = dot_pos[0] - obs[0]
        dy = dot_pos[1] - obs[1]
        dist_o = math.hypot(dx, dy)
        if dist_o < repulsion_radius and dist_o > 1e-6:
            push_dir = [dx / dist_o, dy / dist_o]
            strength = repulsion_gain / (dist_o ** 2)
            repulse_x += push_dir[0] * strength
            repulse_y += push_dir[1] * strength

    # Combine goal attraction + obstacle repulsion
    combined = [goal_dir[0] + repulse_x, goal_dir[1] + repulse_y]
    mag = math.hypot(combined[0], combined[1])
    return [combined[0] / mag, combined[1] / mag] if mag > 1e-6 else goal_dir
```

### 12.5 New DualSense Implementation for Kinova

For the new codebase, implement DualSense support at two levels:

**A. Pygame-based (for simulation/user studies):**
- Wrap the MK14 patterns into a clean `DualSenseInput` class
- Support the same 5 control modes as the user study
- Add configurable deadzone, sensitivity, and axis mapping
- Support hot-plugging (detect controller connect/disconnect)

**B. ROS 1-based (for real-world Kinova deployment):**
- Use the ROS `joy` package (`joy_node`) to publish DualSense input as `sensor_msgs/Joy`
- Create a `DualSenseInterface` node that subscribes to `/joy` and publishes clean velocity commands
- Map left stick to XY Cartesian velocity, L2/R2 to gamma adjustment
- Apply deadzone filtering and velocity scaling matching the simulation

**DualSense PS5 axis/button mapping (SDL2/pygame):**
| Index | Input | Use |
|-------|-------|-----|
| Axis 0 | Left stick X | End-effector X velocity |
| Axis 1 | Left stick Y | End-effector Y velocity |
| Axis 2 | Right stick X | (unused / future: camera pan) |
| Axis 3 | Right stick Y | (unused / future: camera tilt) |
| Axis 4 | L2 trigger | Decrease gamma (manual mode) |
| Axis 5 | R2 trigger | Increase gamma (manual mode) |
| Button 0 | Square | Reset position |
| Button 1 | Cross | Confirm / Start trial |
| Button 2 | Circle | Cancel / Skip |
| Button 3 | Triangle | Toggle mode |

---

## 13. Paper Context — Key Results to Reproduce/Extend

This section documents the paper's key quantitative results that the new experiment must match or exceed, providing baseline targets for Experiment 4.

### 13.1 Experiment 1 Results (2D Cursor Control, N=12 human participants)

| Condition | Success (%) | Time (s) | Path Efficiency | Throughput |
|-----------|------------|----------|-----------------|------------|
| No Assist (γ=0) | 72.1 ± 3.2 | 8.44 ± 0.31 | 0.43 ± 0.05 | 1.14 ± 0.10 |
| DQN | 89.8 ± 2.5 | 5.62 ± 0.26 | 0.75 ± 0.05 | 1.20 ± 0.10 |
| IDA | 92.5 ± 2.4 | 5.07 ± 0.27 | 0.63 ± 0.06 | 1.19 ± 0.10 |
| Manual γ | 86.8 ± 1.9 | 4.50 ± 0.23 | 0.82 ± 0.06 | 1.25 ± 0.09 |
| **BRACE** | **98.3 ± 1.7** | **3.30 ± 0.22** | **0.89 ± 0.06** | **1.27 ± 0.08** |

Key γ dynamics: starts at 0.28 ± 0.12 during trajectory initiation (high uncertainty), increases to 0.74 ± 0.09 as goal confidence improves.

### 13.2 Experiment 2 Results (Reacher-2D, simulated)

- BRACE: 4.8 goals/minute (variance σ²=0.46)
- IDA: 3.7 goals/minute (variance σ²=0.89)
- Noisy pilot: 2.4 goals/minute
- BRACE advantage: 29.7% over IDA, 106% over pilot-only

### 13.3 Experiment 3 Results (Fetch Pick-and-Place, simulated)

| Method | Success (%) | Time (s) | Collisions | Placement Error (cm) |
|--------|------------|----------|------------|---------------------|
| IDA | 68 ± 2.9 | 14.7 ± 0.8 | 0.58 ± 0.07 | 2.50 ± 0.20 |
| DQN | 74 ± 2.6 | 12.3 ± 0.7 | 0.41 ± 0.06 | 1.90 ± 0.16 |
| **BRACE** | **86 ± 2.2** | **9.8 ± 0.6** | **0.22 ± 0.04** | **1.30 ± 0.12** |

### 13.4 Ablation Results (from paper)

| Variant | Success (%) | Time (s) | Path Efficiency |
|---------|------------|----------|-----------------|
| Frozen pretrained belief | 81.5 | 4.13 | 0.64 |
| Joint training from scratch | 90.7 | 3.49 | 0.71 |
| Warm-start (5 ep) | 93.4 | 3.25 | 0.75 |
| Warm-start (15 ep) | 94.6 | 3.12 | 0.76 |
| Warm-start (30 ep) | 94.9 | 3.08 | 0.77 |
| Uniform prior (no belief) | 87.2 ± 3.0 | 3.96 ± 0.32 | 0.62 ± 0.07 |
| Full belief conditioning | 94.6 ± 2.3 | 3.12 ± 0.27 | 0.76 ± 0.05 |
| Without curriculum | 90.2 ± 2.7 | 3.58 ± 0.30 | 0.68 ± 0.06 |
| With curriculum | 94.6 ± 2.3 | 3.12 ± 0.27 | 0.76 ± 0.05 |

### 13.5 Expert Robustness Results

| Expert Policy | Expert Performance | BRACE Performance | Delta |
|---------------|-------------------|-------------------|-------|
| Full | 100% | 98.2% | -1.8% |
| Horizon-Limited | 82.4% | 94.1% | +11.7% |
| Delayed | 73.6% | 92.8% | +19.2% |
| Random-Perturbed | 63.1% | 90.5% | +27.4% |

### 13.6 Input Modality Results (DualSense vs Force-based)

| Interface | Success Rate (%) | Completion Time (s) | Path Efficiency |
|-----------|-----------------|--------------------:|-----------------|
| DualSense | 98.1 ± 1.8 | 3.32 ± 0.23 | 0.88 ± 0.06 |
| Force-based | 97.6 ± 2.1 | 3.41 ± 0.29 | 0.85 ± 0.07 |

No significant differences (all p > 0.7), confirming input-modality agnosticism.

### 13.7 Computational Benchmarks

| Metric | RTX 3080 Desktop | i5 Laptop CPU |
|--------|-----------------|---------------|
| End-to-end latency | 36 ms (27 Hz) | 49 ms (20.4 Hz) |
| Bayesian inference | ~3 ms | — |
| Policy forward pass | ~12 ms | — |
| Full curriculum training | ~7 hours | — |
| Belief pretraining | ~45 minutes | — |

### 13.8 Uncertainty-Level Performance (Maze Environment)

| Uncertainty Level | Success Improvement (%) | Time Reduction (%) | Path Efficiency Improvement (%) |
|-------------------|------------------------|-------------------|---------------------------------|
| Low (H < 0.5) | 1.1 ± 0.4 | 3.2 ± 0.7 | 2.3 ± 0.6 |
| Medium (0.5 ≤ H ≤ 1.0) | 2.4 ± 0.5 | 8.7 ± 1.1 | 5.8 ± 0.9 |
| High (H > 1.0) | 4.3 ± 0.6 | 16.2 ± 1.8 | 9.1 ± 1.2 |
| Multi-target | 13.1 ± 1.2 | 24.5 ± 2.3 | 18.6 ± 1.7 |
