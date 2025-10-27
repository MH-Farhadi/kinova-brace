from __future__ import annotations

import random
import math
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from isaaclab.sim.spawners.from_files import UsdFileCfg

@dataclass
class SpawnBounds:
    """Axis-aligned bounding box for random spawn (in meters, relative to parent prim)."""
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]


@dataclass
class ObjectLoaderConfig:
    """Configuration for spawning USD objects from dataset folders."""
    
    dataset_dirs: list[str]
    bounds: SpawnBounds
    min_distance: float = 0.08
    max_placement_tries_per_object: int = 500
    parent_objects_relpath: str = "Objects"
    
    # Optional uniform scale range applied per object (x=y=z)
    uniform_scale_range: tuple[float, float] | None = None
    
    # Physics and placement
    enable_physics: bool = True
    mass_kg: float | None = None
    density: float | None = 500.0
    contact_offset: float = 0.005
    rest_offset: float = 0.001
    
    # Z placement control
    snap_z_to: float | None = None
    z_clearance: float = 0.005
    z_jitter_max: float = 0.01
    
    # Orientation at spawn: Euler degrees in XYZ order
    orientation_euler_deg: tuple[float, float, float] | None = None
    
    # Visual fallback
    apply_preview_surface: bool = False
    preview_surface_diffuse: tuple[float, float, float] = (0.7, 0.7, 0.7)


class ObjectLoader:
    """Loads USD objects from directories and spawns them randomly into the scene."""

    def __init__(self, cfg: ObjectLoaderConfig):
        self.cfg = cfg
        self._usd_file_paths: list[str] = self._collect_usd_files(cfg.dataset_dirs)
        if len(self._usd_file_paths) == 0:
            raise FileNotFoundError(f"No .usd files found in dataset_dirs={cfg.dataset_dirs}.")

    @staticmethod
    def _collect_usd_files(dirs: Iterable[str]) -> list[str]:
        """Collect USD files from local directories and/or Nucleus directories."""
        usd_files: list[str] = []
        omni_client = None
        try:
            omni_client = importlib.import_module("omni.client")
        except Exception:
            pass

        def is_nucleus_dir(path_str: str) -> bool:
            if path_str.startswith(("omniverse://", "http://", "https://")):
                return True
            if omni_client is None:
                return False
            try:
                res, _ = omni_client.list(path_str)
                return res == getattr(omni_client, "Result").OK
            except Exception:
                return False

        for d in dirs:
            if not d:
                continue
            d_str = str(d)
            if is_nucleus_dir(d_str):
                usd_files.extend(ObjectLoader._list_usd_files_in_nucleus_dir(d_str, omni_client))
                continue
            # Local filesystem
            base = Path(d_str).expanduser().resolve()
            if not base.exists() or not base.is_dir():
                continue
            for path in base.rglob("*.usd"):
                name = path.name
                if name.startswith(".") or name.endswith("~") or ".thumb" in name.lower():
                    continue
                usd_files.append(str(path))
        return sorted(set(usd_files))

    @staticmethod
    def _list_usd_files_in_nucleus_dir(root_url: str, omni_client) -> list[str]:
        """Recursively list USD files from Nucleus directory."""
        if omni_client is None:
            return []
        urls: list[str] = []
        stack: list[str] = [root_url.rstrip("/")]
        seen: set[str] = set()
        
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            
            try:
                res, entries = omni_client.list(current)
                if res != getattr(omni_client, "Result").OK:
                    continue
                    
                for entry in entries:
                    name = getattr(entry, "name", None) or getattr(entry, "relative_path", None)
                    if not name:
                        continue
                    child = f"{current}/{name}" if not current.endswith("/") else f"{current}{name}"
                    
                    if str(name).lower().endswith(".usd"):
                        if ".thumb" not in str(name).lower():
                            urls.append(child)
                        continue
                        
                    # Try treating as directory
                    try:
                        res2, _ = omni_client.list(child)
                        if res2 == getattr(omni_client, "Result").OK:
                            stack.append(child)
                    except Exception:
                        pass
            except Exception:
                continue
                
        return urls

    @staticmethod
    def _ensure_parent_prim(parent_prim_path: str, child_relpath: str) -> str:
        """Ensures parent Xform exists and returns full prim path to child root."""
        prim_utils = importlib.import_module("isaacsim.core.utils.prims")
        stage_utils = importlib.import_module("isaacsim.core.utils.stage")
        
        stage = stage_utils.get_current_stage()
        
        # Create parent if needed
        parent_prim = stage.GetPrimAtPath(parent_prim_path)
        if not parent_prim.IsValid():
            prim_utils.create_prim(parent_prim_path, "Xform")
            
        # Create objects root if needed
        objects_root = f"{parent_prim_path.rstrip('/')}/{child_relpath.strip('/')}"
        objects_root_prim = stage.GetPrimAtPath(objects_root)
        if not objects_root_prim.IsValid():
            prim_utils.create_prim(objects_root, "Xform")
            
        return objects_root

    @staticmethod
    def _sample_position(bounds: SpawnBounds) -> tuple[float, float, float]:
        x = random.uniform(bounds.min_xyz[0], bounds.max_xyz[0])
        y = random.uniform(bounds.min_xyz[1], bounds.max_xyz[1])
        z = random.uniform(bounds.min_xyz[2], bounds.max_xyz[2])
        return (x, y, z)

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def _generate_positions(self, count: int) -> list[tuple[float, float, float]]:
        """Generate non-overlapping positions for objects."""
        positions: list[tuple[float, float, float]] = []
        tries_left = self.cfg.max_placement_tries_per_object * count
        
        while len(positions) < count and tries_left > 0:
            candidate = list(self._sample_position(self.cfg.bounds))
            
            # Override Z if snapping to table
            if self.cfg.snap_z_to is not None:
                z_jitter = random.uniform(0.0, self.cfg.z_jitter_max)
                candidate[2] = self.cfg.snap_z_to + self.cfg.z_clearance + z_jitter
                
            candidate_t = (candidate[0], candidate[1], candidate[2])
            
            if all(self._distance(candidate_t, p) >= self.cfg.min_distance for p in positions):
                positions.append(candidate_t)
                
            tries_left -= 1
            
        return positions

    def _compute_orientation(self) -> tuple[float, float, float, float] | None:
        """Compute quaternion from Euler degrees config."""
        if self.cfg.orientation_euler_deg is None:
            return None
            
        try:
            import torch
            from isaaclab.utils.math import quat_from_euler_xyz
            
            roll_deg, pitch_deg, yaw_deg = self.cfg.orientation_euler_deg
            r = torch.tensor([math.radians(float(roll_deg))])
            p = torch.tensor([math.radians(float(pitch_deg))])
            y = torch.tensor([math.radians(float(yaw_deg))])
            
            q_tensor = quat_from_euler_xyz(r, p, y)[0]
            qw, qx, qy, qz = [float(v) for v in q_tensor.tolist()]
            return (qw, qx, qy, qz)
            
        except Exception:
            # Fallback for single-axis rotations
            rx, ry, rz = self.cfg.orientation_euler_deg
            if abs(rx) > 0 and abs(ry) == 0 and abs(rz) == 0:
                a = math.radians(rx) * 0.5
                return (math.cos(a), math.sin(a), 0.0, 0.0)
            elif abs(ry) > 0 and abs(rx) == 0 and abs(rz) == 0:
                a = math.radians(ry) * 0.5
                return (math.cos(a), 0.0, math.sin(a), 0.0)
            elif abs(rz) > 0 and abs(rx) == 0 and abs(ry) == 0:
                a = math.radians(rz) * 0.5
                return (math.cos(a), 0.0, 0.0, math.sin(a))
            return None

    def _ensure_object_physics(self, prim_path: str) -> None:
        """Ensure object has proper physics schemas applied."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.sim.schemas import (
                define_rigid_body_properties,
                define_collision_properties,
                define_mass_properties,
            )
            
            # Apply rigid body
            rb_cfg = sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            )
            define_rigid_body_properties(prim_path, rb_cfg)
            
            # Apply collision
            col_cfg = sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=self.cfg.contact_offset,
                rest_offset=self.cfg.rest_offset,
            )
            define_collision_properties(prim_path, col_cfg)
            
            # Apply mass
            mass_val = 0.3  # Default 300g
            if self.cfg.mass_kg is not None:
                mass_val = self.cfg.mass_kg
            elif self.cfg.density is not None:
                mass_val = self.cfg.density * 0.0001  # Rough volume estimate
                
            mass_cfg = sim_utils.MassPropertiesCfg(mass=mass_val)
            define_mass_properties(prim_path, mass_cfg)
            
        except Exception:
            pass  # Silent fallback - some objects may not support schema modification

    def spawn(self, parent_prim_path: str, num_objects: int) -> list[str]:
        """Spawn random objects under the given parent prim."""
        import isaaclab.sim as sim_utils

        # Prepare objects container
        objects_root = self._ensure_parent_prim(parent_prim_path, self.cfg.parent_objects_relpath)

        # Select USDs
        if num_objects <= len(self._usd_file_paths):
            selection = random.sample(self._usd_file_paths, k=num_objects)
        else:
            selection = [random.choice(self._usd_file_paths) for _ in range(num_objects)]

        positions = self._generate_positions(len(selection))
        orientation = self._compute_orientation()
        spawned_prim_paths: list[str] = []

        for idx, (usd_path, pos) in enumerate(zip(selection, positions), start=1):
            # Configure spawn with optional scaling
            scale_arg = None
            if self.cfg.uniform_scale_range is not None:
                smin, smax = self.cfg.uniform_scale_range
                s = random.uniform(smin, smax)
                scale_arg = (s, s, s)

            usd_cfg = UsdFileCfg(
                usd_path=str(usd_path),
                scale=scale_arg,
            )

            # Optional preview material
            if self.cfg.apply_preview_surface:
                vm_cfg = importlib.import_module("isaaclab.sim.spawners.materials.visual_materials_cfg")
                preview = vm_cfg.PreviewSurfaceCfg(diffuse_color=self.cfg.preview_surface_diffuse)
                usd_cfg.visual_material = preview

            prim_path = f"{objects_root}/Obj_{idx:02d}"
            
            try:
                # Spawn object
                if orientation is not None:
                    usd_cfg.func(prim_path, usd_cfg, translation=pos, orientation=orientation)
                else:
                    usd_cfg.func(prim_path, usd_cfg, translation=pos)
                
                # Ensure physics (silent fallback)
                if self.cfg.enable_physics:
                    self._ensure_object_physics(prim_path)
                
                spawned_prim_paths.append(prim_path)
                
            except Exception:
                continue  # Skip failed spawns silently

        return spawned_prim_paths