from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PhysicsConfig:
    """Centralized physics configuration for simulation and spawned rigid objects.

    Use this to avoid scattering physics flags across CLI and code paths.
    """

    # Simulation settings
    device: str = "cuda:0"
    dt: float = 1.0 / 80.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Rigid object defaults
    enable_physics: bool = True
    density: float | None = 800.0
    mass_kg: float | None = None
    contact_offset: float = 0.003
    rest_offset: float = 0.0

    # Spawn placement conveniences
    snap_z_to: float | None = None
    z_clearance: float = 0.02
    orientation_euler_deg: Tuple[float, float, float] | None = (90, 0, 0)

    # Visual fallback
    apply_preview_surface: bool = False
    preview_surface_diffuse: Tuple[float, float, float] = (0.7, 0.7, 0.7)


def apply_to_simulation_cfg(sim_cfg, phys: PhysicsConfig) -> None:
    """Mutate an isaaclab.sim.SimulationCfg with values from PhysicsConfig."""
    sim_cfg.dt = float(phys.dt)
    sim_cfg.gravity = tuple(phys.gravity)
    # device is handled at SimulationCfg creation in demo.py


def object_loader_kwargs_from_physix(phys: PhysicsConfig) -> dict:
    """Translate PhysicsConfig into ObjectLoaderConfig kwargs."""
    return {
        "enable_physics": bool(phys.enable_physics),
        "density": None if phys.density is None else float(phys.density),
        "mass_kg": None if phys.mass_kg is None else float(phys.mass_kg),
        "contact_offset": float(phys.contact_offset),
        "rest_offset": float(phys.rest_offset),
        "snap_z_to": None if phys.snap_z_to is None else float(phys.snap_z_to),
        "z_clearance": float(phys.z_clearance),
        "orientation_euler_deg": None if phys.orientation_euler_deg is None else tuple(phys.orientation_euler_deg),
        "apply_preview_surface": bool(phys.apply_preview_surface),
        "preview_surface_diffuse": tuple(phys.preview_surface_diffuse),
    }


