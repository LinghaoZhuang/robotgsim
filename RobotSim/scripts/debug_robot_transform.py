"""
Debug robot transform - trace coordinates at each step.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from Gaussians.util_gau import load_ply
from robot_gaussian.robot_gaussian_model import (
    DEFAULT_R_Y_90, DEFAULT_ICP_ROTATION, DEFAULT_ICP_TRANSLATION, DEFAULT_GENESIS_SCALE
)

def print_stats(name, xyz):
    print(f"\n{name}:")
    print(f"  Center: [{xyz[:, 0].mean():.4f}, {xyz[:, 1].mean():.4f}, {xyz[:, 2].mean():.4f}]")
    print(f"  X: [{xyz[:, 0].min():.4f}, {xyz[:, 0].max():.4f}]")
    print(f"  Y: [{xyz[:, 1].min():.4f}, {xyz[:, 1].max():.4f}]")
    print(f"  Z: [{xyz[:, 2].min():.4f}, {xyz[:, 2].max():.4f}]")

def main():
    print("=" * 70)
    print("Robot Transform Debug")
    print("=" * 70)

    # Load robot PLY
    robot = load_ply('exports/mult-view-scene/robot.ply')
    xyz = robot.xyz
    print_stats("1. Original robot.ply (COLMAP coords)", xyz)

    # Apply R_y_90
    xyz_rotated = (DEFAULT_R_Y_90 @ xyz.T).T
    print_stats("2. After R_y_90 rotation", xyz_rotated)

    # Apply ICP rotation
    R_combined = DEFAULT_ICP_ROTATION @ DEFAULT_R_Y_90
    xyz_combined = (R_combined @ xyz.T).T
    print_stats("3. After combined rotation (icp_rot @ R_y_90)", xyz_combined)

    # Apply ICP translation
    xyz_aligned = xyz_combined + DEFAULT_ICP_TRANSLATION
    print_stats("4. After ICP translation (scaled Genesis coords)", xyz_aligned)

    # Load genesis_center
    try:
        genesis_center = np.load('data/labels/genesis_center.npy')
        print(f"\nGenesis center: {genesis_center}")
    except:
        genesis_center = np.array([0.2, 0.0, 0.2])
        print(f"\nGenesis center (default): {genesis_center}")

    # Unscale
    C = genesis_center
    xyz_unscaled = (xyz_aligned - C) / DEFAULT_GENESIS_SCALE + C
    print_stats("5. After unscale (Genesis coords)", xyz_unscaled)

    # Compare with background
    print("\n" + "=" * 70)
    print("Comparison with Background")
    print("=" * 70)

    bg = load_ply('exports/mult-view-scene/left-transform2.ply')
    bg_above = bg.xyz[bg.xyz[:, 2] > 0]

    print(f"\nBackground (Z > 0) center: [{bg_above[:, 0].mean():.4f}, {bg_above[:, 1].mean():.4f}, {bg_above[:, 2].mean():.4f}]")
    print(f"Robot (final) center: [{xyz_unscaled[:, 0].mean():.4f}, {xyz_unscaled[:, 1].mean():.4f}, {xyz_unscaled[:, 2].mean():.4f}]")

    diff = np.array([bg_above[:, 0].mean(), bg_above[:, 1].mean(), bg_above[:, 2].mean()]) - \
           np.array([xyz_unscaled[:, 0].mean(), xyz_unscaled[:, 1].mean(), xyz_unscaled[:, 2].mean()])
    print(f"\nDifference (BG - Robot): {diff}")
    print(f"Suggested additional translation: {diff}")

if __name__ == "__main__":
    main()
