# `assist/`

“Assist” utilities used by the Kinova demos for:

- logging (`logger.py`)
- object tracking (`objects.py`)
- input multiplexing / orchestration (`input_mux.py`, `orchestrator.py`)

This code is currently used primarily by `kinova-isaac/demo.py` and `data_collection/demo.py` to emit structured logs.

## Logs

Default log root is `kinova-isaac/logs/assist/` (configurable in some scripts via `--logs-root`).


