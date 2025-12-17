# `data_collection/`

Episode-based data collection utilities built on the same Kinova + Isaac Lab scene as the demos.

## Run

Runs multiple episodes and writes logs under `--logs-root` (default: `logs/assist`):

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/data_collection/demo.py --headless --device cuda --planner scripted --num-episodes 10 --logs-root kinova-isaac/logs/assist
```

## Notes

- Uses the same object spawning logic as the demos. If asset spawning is failing, try `--no-objects`.
- Planner choices in this script are currently `scripted | rmpflow | curobo`.


