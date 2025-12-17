# `data_collection/`

Episode-based data collection utilities built on the same Kinova + Isaac Lab scene as the demos.

This package is structured to support **multiple task-specific data-collection pipelines** while reusing shared tooling.

## Recommended layout

- `data_collection/core/`: task-agnostic utilities (logging, schemas, trackers, mux inputs)
- `data_collection/engine/`: reusable episode/rollout helpers (runners, waypoint generators, termination helpers)
- `data_collection/samplers/`: reusable estimators/samplers (target selection, grasp pose estimation helpers)
- `data_collection/tasks/<task_name>/`: task-specific pipelines (their own CLI + run loop)

## Run

### Default task (reach-to-grasp)

Runs multiple episodes and writes logs under `--logs-root` (default: `logs/data_collection`):

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/data_collection/demo.py --headless --device cuda --planner scripted --num-episodes 10 --logs-root kinova-isaac/logs/data_collection
```

### Task runner (recommended)

- Default task:

```bash
cd <repo-root>
python -m data_collection.cli --num-episodes 10 --planner scripted --logs-root kinova-isaac/logs/data_collection
```

- Explicit task selection:

```bash
cd <repo-root>
python -m data_collection.cli reach_to_grasp --num-episodes 10 --planner scripted --logs-root kinova-isaac/logs/data_collection
```

## Notes

- Uses the same object spawning logic as the demos. If asset spawning is failing, try `--no-objects`.
- Planner choices in this script are currently `scripted | rmpflow | curobo`.


