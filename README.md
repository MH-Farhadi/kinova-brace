# BRACE Kinova

BRACE (**Bayesian Reinforcement Assistance with Context Encoding**) for a Kinova reach-and-grasp shared-autonomy task in a **2D planar workspace**.

This repository now includes a full modular Python implementation under `brace_kinova/`:

- `envs/`: 2D Gymnasium reach-grasp env, scenarios, wrappers
- `models/`: Bayesian inference, PPO arbitration policy, expert wrapper, simulated human
- `training/`: expert training (SAC), belief pretraining, arbitration training (PPO + curriculum), rewards, callbacks
- `evaluation/`: baseline + BRACE evaluation and plotting utilities
- `ros_interface/`: ROS 1 inference node, Kinova bridge, DualSense interface
- `configs/`: YAML configs for environment, expert, belief, and arbitration

## Implemented Features

- Bayesian belief update with learnable `beta`, `w_theta`, `w_dist` using `nn.Parameter` + `softplus`
- PPO-compatible scalar arbitration policy with `gamma = 0.5 * (action + 1)`
- 5-stage curriculum progression (basic reaching to full complexity)
- Simulated human model with minimum-jerk direction + potential-field avoidance + AR(1) pink noise
- Expert training pipeline (SAC), belief pretraining, and arbitration training pipeline
- Evaluation pipeline with success, time, collision, belief-accuracy, and gamma metrics
- ROS 1 deployment interface (`rospy`) for Kinova control path
- GPU auto-detection (`cuda` when available)

## Project Layout

```text
brace_kinova/
├── __init__.py
├── requirements.txt
├── configs/
│   ├── env.yaml
│   ├── expert.yaml
│   ├── belief.yaml
│   └── arbitration.yaml
├── envs/
├── models/
├── training/
├── evaluation/
└── ros_interface/
```

## Setup

From repo root:

```bash
pip install -r brace_kinova/requirements.txt
```

## Run Commands (Training + Evaluation)

Run these in order from repo root:

```bash
# 1) Train expert (SAC)
python -m brace_kinova.training.train_expert --config brace_kinova/configs/expert.yaml

# 2) Pretrain Bayesian belief module
python -m brace_kinova.training.train_belief --config brace_kinova/configs/belief.yaml

# 3) Train BRACE arbitration (PPO + curriculum)
python -m brace_kinova.training.train_arbitration --config brace_kinova/configs/arbitration.yaml

# 4) Evaluate BRACE and baselines
python -m brace_kinova.evaluation.evaluate --config brace_kinova/configs/arbitration.yaml --n-episodes 100 --output results.json
```

Optional explicit GPU:

```bash
python -m brace_kinova.training.train_expert --config brace_kinova/configs/expert.yaml --device cuda
python -m brace_kinova.training.train_belief --config brace_kinova/configs/belief.yaml --device cuda
python -m brace_kinova.training.train_arbitration --config brace_kinova/configs/arbitration.yaml --device cuda
```

TensorBoard monitoring:

```bash
tensorboard --logdir=./logs
```

## ROS 1 Deployment (Optional)

Requires ROS 1 + Kinova `ros_kortex` setup:

```bash
rosrun brace_kinova brace_node.py _config_path:=brace_kinova/configs/arbitration.yaml
```

## Notes

- A longer implementation breakdown is in `IMPLEMENTATION_SUMMARY.md`.
- This implementation follows the architecture and hyperparameter guidance from `PROMPT.md`.