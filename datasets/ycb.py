from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description="Test script to list YCB object paths from Nucleus.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Now import after app launch (Omniverse context is active)
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    # Standard YCB object names (21 core models)
    ycb_objects = [
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "010_potted_meat_can",
        "011_banana",
        "019_pitcher_base",
        "021_bleach_cleanser",
        "024_bowl",
        "025_mug",
        "035_power_drill",
        "036_wood_block",
        "037_scissors",
        "040_battery",
        "051_large_clamp",
        "052_tiny_clamp",
        "061_foam_brick",
    ]

    print("[INFO] Testing YCB dataset access via Nucleus paths.")
    print(f"[INFO] ISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")
    print("\nYCB Objects and Paths:")
    print("-" * 50)
    for obj_name in ycb_objects:
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/{obj_name}/{obj_name}.usd"
        print(f"Object: {obj_name} | Path: {usd_path}")

    print("\n[INFO] If paths look correct (e.g., start with /NVIDIA/Assets/...), access is available.")
    print("[INFO] To test spawning, extend this script with RigidObjectCfg and InteractiveScene.")
    print("[INFO] Run with: cd /home/ali/github/ali-rabiee/IsaacLab && ./isaaclab.sh -p /home/ali/github/ali-rabiee/kinova-isaac/datasets/ycb.py")

    simulation_app.close()


if __name__ == "__main__":
    main()