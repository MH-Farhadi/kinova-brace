from __future__ import annotations

import argparse
import sys
from typing import Iterable


def list_dir(omni_client, url: str) -> list[str]:
    try:
        res, entries = omni_client.list(url)
        if res != getattr(omni_client, "Result").OK:
            return []
        out: list[str] = []
        for e in entries:
            name = getattr(e, "name", None) or getattr(e, "relative_path", None)
            if not name:
                continue
            out.append(name)
        return out
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser("Probe Omniverse Nucleus to find YCB asset paths")
    # Allow headless launch
    try:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        AppLauncher = None  # type: ignore
    args, _ = parser.parse_known_args()

    # Launch minimal Omniverse app (needed for omni libraries)
    app = None
    if AppLauncher is not None:
        app = AppLauncher(args).app

    import importlib
    try:
        assets = importlib.import_module("isaaclab.utils.assets")
        ISAAC_NUCLEUS_DIR = assets.ISAAC_NUCLEUS_DIR  # e.g., omniverse://localhost/NVIDIA/Assets/Isaac/5.0/Isaac
        NVIDIA_NUCLEUS_DIR = assets.NVIDIA_NUCLEUS_DIR
        print("ISAAC_NUCLEUS_DIR:", ISAAC_NUCLEUS_DIR)
        print("NVIDIA_NUCLEUS_DIR:", NVIDIA_NUCLEUS_DIR)
    except Exception as e:
        print("[warn] isaaclab.utils.assets import failed:", e)
        ISAAC_NUCLEUS_DIR = "/Isaac"  # best-effort fallback

    omni_client = importlib.import_module("omni.client")

    candidates: list[str] = []
    # Primary expected location under configured Isaac assets root
    candidates.append(f"{ISAAC_NUCLEUS_DIR}/Props/YCB")
    # Additional common locations by explicit version directories
    for ver in ("4.5", "5.0", "2022.2.0", "2023.1.0", "2024.1.0"):
        candidates.append(
            f"omniverse://localhost/NVIDIA/Assets/Isaac/{ver}/Isaac/Props/YCB"
        )

    print("\nChecking candidate YCB directories:")
    found: list[str] = []
    for url in candidates:
        try:
            stat_res, _ = omni_client.stat(url)
            ok = stat_res == getattr(omni_client, "Result").OK
        except Exception:
            ok = False
        print(f" - {url} :: {'OK' if ok else 'NOT FOUND'}")
        if ok:
            found.append(url)

    if not found:
        print("\nNo YCB folder found at standard locations. If your Nucleus is remote, replace 'localhost' above with your server host.")
    else:
        # For each found YCB directory, list a few USDs
        for url in found:
            print(f"\nListing USDs under: {url}")
            # breadth listing one level
            entries = list_dir(omni_client, url)
            usd_files: list[str] = []
            for name in entries:
                child = f"{url}/{name}"
                # If it's a directory, list inside
                try:
                    res, ent = omni_client.list(child)
                    if res == getattr(omni_client, "Result").OK:
                        # collect usd files under this child dir
                        for ee in ent:
                            ename = getattr(ee, "name", None) or getattr(ee, "relative_path", None)
                            if ename and ename.lower().endswith(".usd"):
                                usd_files.append(f"{child}/{ename}")
                except Exception:
                    pass
            print(f"Found {len(usd_files)} USD(s) (showing up to 10):")
            for u in usd_files[:10]:
                print("  ", u)

    # Close app cleanly
    if app is not None:
        app.close()


if __name__ == "__main__":
    sys.exit(main())


