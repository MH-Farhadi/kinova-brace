from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import importlib


@dataclass
class SpawnBounds:
    """Axis-aligned bounding box for random spawn (in meters, relative to parent prim).

    min_xyz and max_xyz are inclusive ranges for uniform sampling.
    """

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]


@dataclass
class ObjectLoaderConfig:
    """Configuration for spawning USD objects from dataset folders."""

    dataset_dirs: list[str]
    bounds: SpawnBounds
    min_distance: float = 0.08
    max_placement_tries_per_object: int = 200
    parent_objects_relpath: str = "Objects"
    # Optional uniform scale range applied per object (x=y=z).
    uniform_scale_range: tuple[float, float] | None = None
    # Optional fallback material: bind a simple preview surface to override missing textures.
    apply_preview_surface: bool = False
    preview_surface_diffuse: tuple[float, float, float] = (0.7, 0.7, 0.7)
    # Physics:
    # If set, forces dynamic rigid bodies with gravity enabled and collisions on.
    enable_physics: bool = True
    # Mass properties: choose either mass (kg) or density (kg/m^3). If both None, leave as USD default.
    mass_kg: float | None = None
    density: float | None = 500.0
    # Collision tuning
    contact_offset: float = 0.003
    rest_offset: float = 0.0
    # Z placement control: if provided, override sampled z to be (snap_z_to + z_clearance)
    snap_z_to: float | None = None
    z_clearance: float = 0.02


class ObjectLoader:
    """Loads USD objects from directories and spawns them randomly into the scene.

    Usage:
        loader = ObjectLoader(cfg)
        prim_paths = loader.spawn(
            parent_prim_path="/World/Origin1",
            num_objects=5,
        )
    """

    def __init__(self, cfg: ObjectLoaderConfig):
        self.cfg = cfg
        self._usd_file_paths: list[str] = self._collect_usd_files(cfg.dataset_dirs)
        if len(self._usd_file_paths) == 0:
            raise FileNotFoundError(
                f"No .usd files found in dataset_dirs={cfg.dataset_dirs}."
            )

    @staticmethod
    def _collect_usd_files(dirs: Iterable[str]) -> list[str]:
        """Collect USD files from local directories and/or Nucleus directories.

        Supports local filesystem paths and Nucleus URLs (e.g., "omniverse://...") or
        absolute Nucleus paths (e.g., "/Isaac/Props/YCB").
        """
        usd_files: list[str] = []
        omni_client = None
        try:
            omni_client = importlib.import_module("omni.client")
        except Exception:
            omni_client = None

        def is_nucleus_dir(path_str: str) -> bool:
            # Treat omniverse and http(s) asset URLs as nucleus-like roots
            if path_str.startswith("omniverse://") or path_str.startswith("http://") or path_str.startswith("https://"):
                return True
            if omni_client is None:
                return False
            try:
                res, _ = omni_client.list(path_str)
                return res == getattr(omni_client, "Result").OK  # type: ignore[attr-defined]
            except Exception:
                return False

        for d in dirs:
            if not d:
                continue
            d_str = str(d)
            if is_nucleus_dir(d_str):
                usd_files.extend(ObjectLoader._list_usd_files_in_nucleus_dir(d_str, omni_client))
                continue
            # Fallback: local filesystem
            base = Path(d_str).expanduser().resolve()
            if not base.exists() or not base.is_dir():
                continue
            for path in base.rglob("*.usd"):
                name = path.name
                if name.startswith(".") or name.endswith("~"):
                    continue
                usd_files.append(str(path))
        return sorted(set(usd_files))

    @staticmethod
    def _list_usd_files_in_nucleus_dir(root_url: str, omni_client) -> list[str]:
        if omni_client is None:
            return []
        urls: list[str] = []
        stack: list[str] = [root_url.rstrip("/")]
        seen: set[str] = set()
        EntryFlags = getattr(omni_client, "EntryFlags", None)
        CAN_HAVE_CHILDREN = getattr(EntryFlags, "CAN_HAVE_CHILDREN", 0) if EntryFlags is not None else 0
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            try:
                res, entries = omni_client.list(current)
            except Exception:
                continue
            if res != getattr(omni_client, "Result").OK:  # type: ignore[attr-defined]
                continue
            for entry in entries:
                name = getattr(entry, "name", None) or getattr(entry, "relative_path", None)
                if not name:
                    continue
                child = f"{current}/{name}" if not current.endswith("/") else f"{current}{name}"
                # If entry looks like a USD file, add it
                if str(name).lower().endswith(".usd"):
                    urls.append(child)
                    continue
                # Otherwise, try to treat it as a directory by probing list()
                probe_ok = False
                try:
                    res2, _ = omni_client.list(child)
                    probe_ok = res2 == getattr(omni_client, "Result").OK  # type: ignore[attr-defined]
                except Exception:
                    probe_ok = False
                if probe_ok:
                    stack.append(child)
        return urls

    @staticmethod
    def _ensure_parent_prim(parent_prim_path: str, child_relpath: str) -> str:
        """Ensures the parent Xform exists and returns the full prim path to the child root.

        Example: parent_prim_path="/World/Origin1", child_relpath="Objects" ->
                 returns "/World/Origin1/Objects"
        """
        import importlib

        prim_utils = importlib.import_module("isaacsim.core.utils.prims")
        # Create the parent only if it does not exist already
        stage_utils = importlib.import_module("isaacsim.core.utils.stage")
        stage = stage_utils.get_current_stage()
        parent_prim = stage.GetPrimAtPath(parent_prim_path)
        if not parent_prim.IsValid():
            prim_utils.create_prim(parent_prim_path, "Xform")
        objects_root = f"{parent_prim_path.rstrip('/')}/{child_relpath.strip('/')}"
        # Create the objects root only if not existing
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
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def _generate_positions(self, count: int) -> list[tuple[float, float, float]]:
        positions: list[tuple[float, float, float]] = []
        tries_left = max(1, int(self.cfg.max_placement_tries_per_object)) * max(1, count)
        while len(positions) < count and tries_left > 0:
            candidate = list(self._sample_position(self.cfg.bounds))
            # If snapping Z to a plane is requested, override candidate's z
            if self.cfg.snap_z_to is not None:
                candidate[2] = float(self.cfg.snap_z_to) + float(self.cfg.z_clearance)
            candidate_t = (float(candidate[0]), float(candidate[1]), float(candidate[2]))
            if all(self._distance(candidate_t, p) >= self.cfg.min_distance for p in positions):
                positions.append(candidate_t)
            tries_left -= 1
        if len(positions) < count:
            print(
                f"[ObjectLoader] Warning: placed only {len(positions)}/{count} objects given min_distance={self.cfg.min_distance} and bounds={self.cfg.bounds}."
            )
        return positions

    def spawn(self, parent_prim_path: str, num_objects: int) -> list[str]:
        """Spawn random objects under the given parent prim.

        Args:
            parent_prim_path: The parent prim (e.g., "/World/Origin1") under which an
                `Objects` Xform is created and objects are spawned.
            num_objects: Number of objects to spawn (best effort if spacing too tight).

        Returns:
            A list of prim paths of the spawned objects.
        """
        import isaaclab.sim as sim_utils

        # Prepare root for objects
        objects_root = self._ensure_parent_prim(parent_prim_path, self.cfg.parent_objects_relpath)

        # Pick USDs (without replacement if possible; with replacement if not enough files)
        selection: list[str]
        if num_objects <= len(self._usd_file_paths):
            selection = random.sample(self._usd_file_paths, k=num_objects)
        else:
            selection = []
            for _ in range(num_objects):
                selection.append(random.choice(self._usd_file_paths))

        positions = self._generate_positions(len(selection))

        spawned_prim_paths: list[str] = []
        for idx, (usd_path, pos) in enumerate(zip(selection, positions), start=1):
            # Configure spawn from USD with rigid body and collisions enabled for dynamic behavior
            rigid_cfg = None
            collision_cfg = None
            mass_cfg = None
            if self.cfg.enable_physics:
                rigid_cfg = sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=False,
                )
                collision_cfg = sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=float(self.cfg.contact_offset),
                    rest_offset=float(self.cfg.rest_offset),
                )
                if self.cfg.mass_kg is not None:
                    mass_cfg = sim_utils.MassPropertiesCfg(mass=float(self.cfg.mass_kg))
                elif self.cfg.density is not None:
                    mass_cfg = sim_utils.MassPropertiesCfg(density=float(self.cfg.density))
            # Optional uniform random scale
            scale_arg = None
            if self.cfg.uniform_scale_range is not None:
                smin, smax = self.cfg.uniform_scale_range
                s = random.uniform(smin, smax)
                scale_arg = (s, s, s)

            usd_cfg = sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                rigid_props=rigid_cfg,
                collision_props=collision_cfg,
                mass_props=mass_cfg,
                scale=scale_arg,  # type: ignore[arg-type]
            )

            # Optionally override materials to avoid missing texture warnings
            if self.cfg.apply_preview_surface:
                # Create a simple preview surface with the requested color
                vm_cfg = __import__(
                    "isaaclab.sim.spawners.materials.visual_materials_cfg",
                    fromlist=["PreviewSurfaceCfg"],
                )
                preview = vm_cfg.PreviewSurfaceCfg(diffuse_color=self.cfg.preview_surface_diffuse)
                usd_cfg.visual_material = preview  # type: ignore[attr-defined]

            prim_path = f"{objects_root}/Obj_{idx:02d}"
            try:
                usd_cfg.func(prim_path, usd_cfg, translation=pos)
                spawned_prim_paths.append(prim_path)
            except Exception as exc:  # pragma: no cover - robust to malformed USDs
                print(f"[ObjectLoader] Failed to spawn '{usd_path}' at {prim_path}: {exc}")

        return spawned_prim_paths


