# `motion_generation/`

Motion generation utilities and demos for Kinova Jaco2 (J2N6S300):

- **Demo**: grasp-loop episode runner (`demo.py`)
- **CLI**: consolidated flags (`cli.py`)
- **Planners**: scripted + adapters for other planners (`planners/`)
- **Grasp estimation**: OBB and Replicator-based providers (`grasp_estimation/`)

## Run: grasp loop demo

Use Isaac Lab’s launcher so the correct Isaac Sim Python/runtime is used:

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/motion_generation/demo.py --device cuda --planner scripted --grasp obb --num-episodes 3
```

Headless:

```bash
./IsaacLab/isaaclab.sh -p kinova-isaac/motion_generation/demo.py --headless --device cuda --planner scripted --num-episodes 3
```

## Common options

See `cli.py` for the full list. Frequently used:

- `--planner`: `scripted | rmpflow | curobo | lula`
- `--grasp`: `obb | replicator`
- `--num-episodes`, `--num-objects`
- `--spawn-min x y z` / `--spawn-max x y z`
- `--no-objects`: skip object spawning (good for debugging asset/network issues)


