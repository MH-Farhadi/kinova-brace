"""Assistive control pipeline (v0) for Kinova Isaac simulation.

This package exposes:
- config: hyperparameters and tunable thresholds
- schemas: compact data records for logging and LM inputs
- window: rolling context builder at fixed assist rate
- objects: lightweight object tracker abstraction
- policy: heuristic suggestion engine (LM stub)
- dialogue: suggestion state machine and cooldowns
- actions: minimal align-and-grasp execution primitives
- input_mux: command multiplexer to inject actions
- orchestrator: ties components together at 10 Hz
- logger: structured JSONL logging utilities

Note: Submodules are intentionally not imported at package import time to
avoid bringing in heavy sim dependencies during lightweight tests.
Import submodules explicitly, e.g. `from assist.config import AssistConfig`.
"""

# Intentionally no eager submodule imports here



