# `environments/`

Scene setup and object loading utilities used by the Kinova demos:

- `reach_to_grasp/`: scene config + `design_scene(...)` helper
- `object_loader.py`: spawns objects (typically YCB assets)
- `physix.py`: physics configuration helpers

## Run: reach-to-grasp environment demo

```bash
cd <repo-root>
./IsaacLab/isaaclab.sh -p kinova-isaac/environments/reach_to_grasp/demo.py --device cuda
```


