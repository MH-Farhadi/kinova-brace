"""Reusable building blocks for data collection across tasks/projects.

IMPORTANT:
This package may be imported by CLIs *before* Isaac/Kit is started.
Avoid importing modules that require `omni.*` at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CommandMuxInputProvider",
    "ObjectsTracker",
    "SessionLogWriter",
    "TickLoggingConfig",
]

_EXPORTS = {
    "CommandMuxInputProvider": ("data_collection.core.input_mux", "CommandMuxInputProvider"),
    "ObjectsTracker": ("data_collection.core.objects", "ObjectsTracker"),
    "SessionLogWriter": ("data_collection.core.logger", "SessionLogWriter"),
    "TickLoggingConfig": ("data_collection.core.logger", "TickLoggingConfig"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(name)
    mod_name, attr = spec
    mod = import_module(mod_name)
    val = getattr(mod, attr)
    globals()[name] = val  # cache
    return val


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(list(globals().keys()) + list(__all__)))


