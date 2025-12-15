# kinova-isaac

Kinova Jaco2 (J2N6S300) **simulation demos, controllers, and motion-generation** built on top of `IsaacLab/` (and optionally cuRobo).

This folder is intended to be run using **Isaac Lab’s launcher** so you get the correct Isaac Sim Python/runtime:

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/demo.py --device cuda
```

## What’s inside

- `demo.py`: Main interactive demo (cartesian velocity jog) + optional object spawning + logging.
- `controllers/`: Modular cartesian velocity jog controller + keyboard input + safety/modes.
- `motion_generation/`: Grasp-loop demo + planner adapters (`scripted`, `rmpflow`, `curobo`, `lula`) + grasp pose providers (`obb`, `replicator`).
- `data_collection/`: Episode runner that logs trajectories/objects for repeated grasp attempts.
- `environments/`: Scene construction (`reach_to_grasp`) and object loading utilities.
- `assist/`: Logging utilities and “assist” orchestration components (currently used mainly for logging/tracking).

## Prerequisites

- **Isaac Sim + Isaac Lab**: `kinova-isaac` imports `isaaclab.*` and expects to run under an Isaac Sim runtime.
- **GPU**: Most workflows are meant for `--device cuda` (but `--device cpu` can work for basic checks).
- **Nucleus assets (optional)**: Some demos spawn YCB objects from Isaac Nucleus. If you don’t have access, use `--no-objects` where supported.

## Run demos

### 1) Main demo: cartesian jog + optional object spawning (`kinova-isaac/demo.py`)

GUI (keyboard teleop):

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/demo.py --device cuda
```

Headless (no keyboard control; useful for smoke tests):

```bash
./IsaacLab/isaaclab.sh -p kinova-isaac/demo.py --headless --device cuda
```

Useful flags (see `kinova-isaac/scripts/cli.py`):

- `--no-objects`: disable object spawning
- `--num-objects N`: number of objects
- `--spawn-min x y z` / `--spawn-max x y z`: spawn bounds
- `--speed`: jog linear speed
- `--print-ee`: print end-effector position

### 2) Controller-only demo (`kinova-isaac/controllers/demo.py`)

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/controllers/demo.py --device cuda
```

### 3) Environment smoke demo (`kinova-isaac/environments/reach_to_grasp/demo.py`)

This runs a simple loop that resets and perturbs joint targets.

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/environments/reach_to_grasp/demo.py --device cuda
```

### 4) Motion generation grasp loop (`kinova-isaac/motion_generation/demo.py`)

Scripted planner (default) + OBB grasp provider:

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/motion_generation/demo.py --device cuda --planner scripted --grasp obb --num-episodes 3
```

Headless:

```bash
./IsaacLab/isaaclab.sh -p kinova-isaac/motion_generation/demo.py --headless --device cuda --planner scripted --num-episodes 3
```

Key options (see `kinova-isaac/motion_generation/cli.py`):

- `--planner`: `scripted | rmpflow | curobo | lula`
- `--grasp`: `obb | replicator`
- `--num-episodes`, `--num-objects`
- `--spawn-min/--spawn-max`, `--min-distance`, `--scale-min/--scale-max`
- `--pregrasp`, `--lift`, `--tolerance`

### 5) Data collection (`kinova-isaac/data_collection/demo.py`)

Runs multiple episodes and logs results under `--logs-root` (default `logs/assist`):

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/data_collection/demo.py --headless --device cuda --planner scripted --num-episodes 10 --logs-root kinova-isaac/logs/assist
```

## Logs

- `kinova-isaac/logs/assist/`: JSONL logs and metadata emitted by the demos/data-collection scripts.

## Notes / gotchas

- **Keyboard teleop requires GUI**: run without `--headless` to use keyboard input.
- **Object spawning depends on asset availability**: if spawning fails, rerun with `--no-objects` to isolate issues.

