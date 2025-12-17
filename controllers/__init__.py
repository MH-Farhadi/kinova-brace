"""Kinova Isaac robot controllers.

IMPORTANT:
This package is imported by data-collection tooling *before* Isaac/Kit is started.
Avoid importing modules that require `omni.*` at import time.

We keep the public API via lazy imports (PEP 562) so existing code like:
`from controllers import CartesianVelocityJogController`
continues to work when running under Isaac/Kit.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    # Controllers
    "CartesianVelocityJogConfig",
    "CartesianVelocityJogController",
    # Input
    "Se3KeyboardInput",
    # Modes
    "ControlMode",
    "ModeManager",
    # Base
    "ArmController",
    "ArmControllerConfig",
    "InputProvider",
]


_EXPORTS = {
    # Controllers
    "CartesianVelocityJogConfig": ("controllers.cartesian_velocity", "CartesianVelocityJogConfig"),
    "CartesianVelocityJogController": ("controllers.cartesian_velocity", "CartesianVelocityJogController"),
    # Input
    "Se3KeyboardInput": ("controllers.input.keyboard", "Se3KeyboardInput"),
    # Modes
    "ControlMode": ("controllers.modes", "ControlMode"),
    "ModeManager": ("controllers.modes", "ModeManager"),
    # Base
    "ArmController": ("controllers.base", "ArmController"),
    "ArmControllerConfig": ("controllers.base", "ArmControllerConfig"),
    "InputProvider": ("controllers.base", "InputProvider"),
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
