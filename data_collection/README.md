# `data_collection/`

Episode-based data collection utilities built on the same Kinova + Isaac Lab scene as the demos.

This package is structured to support **multiple environments** + **multiple data schemas** (profiles) while reusing shared tooling.

## Recommended layout

- `data_collection/core/`: task-agnostic utilities (logging, schemas, trackers, mux inputs)
- `data_collection/engine/`: reusable episode/rollout helpers (runners, waypoint generators, termination helpers)
- `data_collection/samplers/`: reusable estimators/samplers (target selection, grasp pose estimation helpers)
- `data_collection/envs/`: registry mapping env names → `environments/<env_name>`
- `data_collection/profiles/`: “what to record” (data structure) implementations + registry

## Run

### `collect_data` (recommended: env + profile)

This entrypoint lets you choose:
- **`--env`**: which environment to load from `kinova-isaac/environments/*`
- **`--profile`**: what data structure to record (ticks-only today; add more profiles over time)

#### Quick start (ticks-only, 30s)

```bash
cd <repo-root>/kinova-isaac
python -m data_collection.collect_data --env reach_to_grasp --profile ticks_v0 --duration-s 30 --log-rate-hz 10 --logs-root logs/data_collection
```

#### Headless + GPU example

```bash
cd <repo-root>/kinova-isaac
python -m data_collection.collect_data --env reach_to_grasp --profile ticks_v0 --headless --device cuda:0 --duration-s 60 --log-rate-hz 10 --logs-root logs/data_collection
```

#### Interactive keyboard control (GUI)

```bash
cd <repo-root>/kinova-isaac
python -m data_collection.collect_data --env reach_to_grasp --profile ticks_v0 --control keyboard --duration-s 60 --logs-root logs/data_collection
```

### Important arguments

- **Selection**
  - `--env`: `reach_to_grasp` | `reach_to_grasp_VLA` (add more in `data_collection/envs/registry.py`)
  - `--profile`: currently `ticks_v0` (add more in `data_collection/profiles/registry.py`)
- **Duration / rate**
  - `--duration-s`: how long to run (seconds)
  - `--log-rate-hz`: tick logging frequency
- **Logging**
  - `--logs-root`: where sessions are written (each run creates a new session folder)
- **Control**
  - `--control`: `keyboard` (GUI) or `idle`
- **Object spawning / controller**
  - Shared flags come from `scripts/cli.py` (e.g. `--num-objects`, `--spawn-min`, `--spawn-max`, `--min-distance`, `--no-objects`, `--speed`, `--rot-speed`, `--ee_link`, …)
- **Isaac/Kit**
  - AppLauncher flags come from IsaacLab (`--headless`, `--device`, etc.)

### Legacy scripts (still available)

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/data_collection/demo.py --headless --device cuda --planner scripted --num-episodes 10 --logs-root kinova-isaac/logs/data_collection
```

## Notes

- Uses the same object spawning logic as the demos. If asset spawning is failing, try `--no-objects`.
- Planner choices in this script are currently `scripted | rmpflow | curobo`.


