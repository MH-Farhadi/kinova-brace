from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PhysicsConfig:
    """Centralized physics configuration for simulation and spawned rigid objects."""

    # Simulation settings
    device: str = "cuda:0"
    dt: float = 1.0 / 240.0
    sub_steps: int = 4
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Rigid object defaults
    enable_physics: bool = True
    density: float | None = 600.0
    mass_kg: float | None = None
    contact_offset: float = 0.005
    rest_offset: float = 0.0015
    # Object friction/material defaults
    object_static_friction: float = 3.0
    object_dynamic_friction: float = 3.0
    object_restitution: float = 0.0
    object_friction_combine_mode: str = "max"
    object_torsional_patch_radius: float = 0.02
    object_min_torsional_patch_radius: float = 0.01

    # Spawn placement
    snap_z_to: float | None = 0.82  # Table surface height
    z_clearance: float = 0.005
    orientation_euler_deg: Tuple[float, float, float] | None = (-90, 0, 0)

    # Visual fallback
    apply_preview_surface: bool = False
    preview_surface_diffuse: Tuple[float, float, float] = (0.7, 0.7, 0.7)


def apply_to_simulation_cfg(sim_cfg, phys: PhysicsConfig) -> None:
    """Apply physics settings to simulation configuration."""
    sim_cfg.dt = float(phys.dt)
    sim_cfg.sub_steps = int(phys.sub_steps)
    sim_cfg.gravity = tuple(phys.gravity)


def object_loader_kwargs_from_physix(phys: PhysicsConfig) -> dict:
    """Convert PhysicsConfig to ObjectLoaderConfig kwargs."""
    return {
        "enable_physics": bool(phys.enable_physics),
        "density": None if phys.density is None else float(phys.density),
        "mass_kg": None if phys.mass_kg is None else float(phys.mass_kg),
        "contact_offset": float(phys.contact_offset),
        "rest_offset": float(phys.rest_offset),
        "static_friction": float(phys.object_static_friction),
        "dynamic_friction": float(phys.object_dynamic_friction),
        "restitution": float(phys.object_restitution),
        "friction_combine_mode": str(phys.object_friction_combine_mode),
        "torsional_patch_radius": float(phys.object_torsional_patch_radius),
        "min_torsional_patch_radius": float(phys.object_min_torsional_patch_radius),
        "snap_z_to": None if phys.snap_z_to is None else float(phys.snap_z_to),
        "z_clearance": float(phys.z_clearance),
        "orientation_euler_deg": None if phys.orientation_euler_deg is None else tuple(phys.orientation_euler_deg),
        "apply_preview_surface": bool(phys.apply_preview_surface),
        "preview_surface_diffuse": tuple(phys.preview_surface_diffuse),
    }