# `controllers/`

Modular cartesian velocity jog controller and input providers used by the Kinova demos.

## Run

This code is meant to run under Isaac Lab / Isaac Sim. Use the Isaac Lab launcher:

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/controllers/demo.py --device cuda
```

## Notes

- **Keyboard input**: requires running **without** `--headless`.
- **Modes**: the demo prints mode keys on startup (translate/rotate/gripper).


