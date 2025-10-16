from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

import importlib
import random

# Omniverse client for listing Nucleus directories (available after AppLauncher)
import omni.client  # type: ignore

# Isaac Lab utilities to resolve Nucleus asset roots
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # type: ignore


SUPPORTED_EXTENSIONS = (".usd",)


@dataclass
class PlannedPlacement:
    """Container for planned object placements."""

    prim_paths: List[str]
    positions: List[Tuple[float, float, float]]


@dataclass
class SpawnBounds:
    min_xyz: Tuple[float, float, float]
    max_xyz: Tuple[float, float, float]


@dataclass
class ObjectLoaderConfig:
    """Configuration for spawning objects from Nucleus.

    - nucleus_dirs: Nucleus directories to sample USDs from (defaults to YCB props)
    - bounds: Axis-aligned bounds for random placement (in meters, relative to parent prim)
    - min_distance: Minimum separation between objects (meters)
    - uniform_scale_range: Optional (min,max) uniform scale applied at spawn
    - apply_preview_surface: Whether to apply a simple preview surface material
    - preview_surface_diffuse: RGB color if preview surface is applied
    - snap_z_to: Optional Z plane to snap objects to before adding z_clearance
    - z_clearance: Additional Z offset above snap plane
    - mass_kg: If provided, sets object mass; otherwise density is used
    - density: If mass_kg is None, density (kg/m^3) to assign
    """

    nucleus_dirs: List[str]
    bounds: SpawnBounds
    min_distance: float = 0.08
    uniform_scale_range: Optional[Tuple[float, float]] = None
    apply_preview_surface: bool = False
    preview_surface_diffuse: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    snap_z_to: Optional[float] = None
    z_clearance: float = 0.02
    mass_kg: Optional[float] = None
    density: float = 500.0


class ObjectLoader:
    """Nucleus-only object loader that discovers USD assets and spawns them dynamically."""

    def __init__(self, cfg: ObjectLoaderConfig) -> None:
        self.cfg = cfg
        # Default to YCB props if nothing provided
        if not self.cfg.nucleus_dirs:
            self.cfg.nucleus_dirs = [f"{ISAAC_NUCLEUS_DIR}/Props/YCB"]

    def _list_usd_files_recursive(self, nucleus_dir: str, max_depth: int = 4, limit: Optional[int] = None) -> List[str]:
        """Recursively list USD files within a Nucleus directory up to max_depth."""
        results: List[str] = []
        dir_path = nucleus_dir.replace("\\", "/")
        def _walk(path: str, depth: int):
            nonlocal results
            if limit is not None and len(results) >= limit:
                return
            result, entries = omni.client.list(path)
            if result != omni.client.Result.OK or entries is None:
                return
            for e in entries:
                if e.relative_path in (".", ".."):
                    continue
                child_path = f"{path}/{e.relative_path}"
                if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
                    if depth < max_depth:
                        _walk(child_path, depth + 1)
                else:
                    if e.relative_path.lower().endswith(".usd"):
                        results.append(child_path)
                        if limit is not None and len(results) >= limit:
                            return
        _walk(dir_path, 0)
        return results

    def discover_assets(self) -> List[str]:
        """Discover USD assets under configured Nucleus directories."""
        all_usds: List[str] = []
        for d in self.cfg.nucleus_dirs:
            all_usds.extend(self._list_usd_files_recursive(d, max_depth=4, limit=None))
        return sorted(all_usds)

    def plan_grid_on_table(
        self,
        table_center: Tuple[float, float, float],
        count: int,
        *,
        x_offset_from_center: float = 0.45,
        y_span: float = 0.6,
        z_offset: float = 0.10,
        base_prim: str = "/World/Origin1/Objects",
    ) -> PlannedPlacement:
        """Plan a simple row/compact grid of placements on the table.

        - Places objects at x = table_center.x + x_offset_from_center
        - Distributes along y within [-y_span/2, y_span/2]
        - Sets z = table_center.z + z_offset so they gently drop onto the table
        """
        cx, cy, cz = table_center
        if count <= 0:
            return PlannedPlacement(prim_paths=[], positions=[])

        # Derive a compact layout: use a single row up to 6, then wrap to multiple rows
        per_row = max(1, min(6, count))
        spacing_y = y_span / max(1, per_row - 1) if per_row > 1 else 0.0
        spacing_row = 0.12  # meters between rows in +x direction

        prim_paths: List[str] = []
        positions: List[Tuple[float, float, float]] = []
        for i in range(count):
            row = i // per_row
            col = i % per_row
            y0 = cy - (y_span / 2.0) + (col * spacing_y)
            x0 = cx + x_offset_from_center + (row * spacing_row)
            z0 = cz + z_offset
            prim_paths.append(f"{base_prim}/Object_{i}")
            positions.append((x0, y0, z0))
        return PlannedPlacement(prim_paths=prim_paths, positions=positions)

    def _sample_positions(self, parent_center: Tuple[float, float, float], count: int) -> PlannedPlacement:
        """Sample positions within bounds with simple Poisson-like rejection on min distance."""
        (minx, miny, minz) = self.cfg.bounds.min_xyz
        (maxx, maxy, maxz) = self.cfg.bounds.max_xyz
        cx, cy, cz = parent_center
        positions: List[Tuple[float, float, float]] = []
        prim_paths: List[str] = []
        attempts = 0
        max_attempts = max(200, 20 * count)
        while len(positions) < count and attempts < max_attempts:
            attempts += 1
            rx = random.uniform(minx, maxx)
            ry = random.uniform(miny, maxy)
            rz = random.uniform(minz, maxz)
            px = cx + rx
            py = cy + ry
            pz = cz + rz
            if self.cfg.snap_z_to is not None:
                pz = self.cfg.snap_z_to + self.cfg.z_clearance
            candidate = (px, py, pz)
            ok = True
            for qx, qy, qz in positions:
                dx = px - qx
                dy = py - qy
                dz = pz - qz
                if (dx * dx + dy * dy + dz * dz) ** 0.5 < self.cfg.min_distance:
                    ok = False
                    break
            if ok:
                positions.append(candidate)
                prim_paths.append(f"/World/Origin1/Objects/Object_{len(positions)-1}")
        return PlannedPlacement(prim_paths=prim_paths, positions=positions)

    def spawn(self, parent_prim_path: str, num_objects: int) -> List[str]:
        """Discover USDs, spawn them under parent, and attach physics if missing.

        Returns list of spawned prim paths.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.sim import schemas
        from isaaclab.sim.utils import get_all_matching_child_prims

        # Ensure parent prim exists (tolerate existing)
        prim_utils = importlib.import_module("isaacsim.core.utils.prims")
        if not prim_utils.is_prim_path_valid(parent_prim_path):
            prim_utils.create_prim(parent_prim_path, "Xform")

        # Discover assets and sample
        all_assets = self.discover_assets()
        if not all_assets:
            return []
        selected = random.sample(all_assets, k=min(num_objects, len(all_assets)))

        # Plan positions around the parent origin (0,0,0 relative)
        planned = self._sample_positions((0.0, 0.0, 0.0), len(selected))

        # Ensure parent and base Xforms exist (idempotent)
        if not prim_utils.is_prim_path_valid(parent_prim_path):
            prim_utils.create_prim(parent_prim_path, "Xform")
        for base in set(p.rsplit("/", 1)[0] for p in planned.prim_paths):
            if not prim_utils.is_prim_path_valid(base):
                prim_utils.create_prim(base, "Xform")

        spawned: List[str] = []
        for i, usd_path in enumerate(selected):
            if i >= len(planned.prim_paths) or i >= len(planned.positions):
                break
            prim_path = planned.prim_paths[i]
            pos = planned.positions[i]

            # Optional uniform scale
            scale_tuple = None
            if self.cfg.uniform_scale_range is not None:
                smin, smax = self.cfg.uniform_scale_range
                if smin is not None and smax is not None and smax > 0:
                    sval = random.uniform(float(smin), float(smax))
                    scale_tuple = (sval, sval, sval)

            # Create the prim referencing the USD at the planned path
            if prim_utils.is_prim_path_valid(prim_path):
                # Skip creating if already exists (idempotent)
                pass
            else:
                prim_utils.create_prim(
                    prim_path,
                    usd_path=usd_path,
                    translation=pos,
                    scale=scale_tuple,
                )

            # Apply physics: RigidBody + Mass on the body prim, Collision on meshes
            try:
                # Rigid body and mass on prim_path
                rb_cfg = sim_utils.RigidBodyPropertiesCfg()
                schemas.define_rigid_body_properties(prim_path, rb_cfg)

                mp_cfg = sim_utils.MassPropertiesCfg()
                if self.cfg.mass_kg is not None:
                    mp_cfg.mass = float(self.cfg.mass_kg)
                else:
                    mp_cfg.density = float(self.cfg.density)
                schemas.define_mass_properties(prim_path, mp_cfg)

                # Collision on all Mesh children
                meshes = get_all_matching_child_prims(prim_path, lambda p: p.GetTypeName() == "Mesh")
                for mesh_prim in meshes:
                    cp_cfg = sim_utils.CollisionPropertiesCfg()
                    schemas.define_collision_properties(mesh_prim.GetPrimPath(), cp_cfg)
            except Exception:
                # Best-effort: if schema application fails, continue
                pass

            spawned.append(prim_path)
        return spawned


