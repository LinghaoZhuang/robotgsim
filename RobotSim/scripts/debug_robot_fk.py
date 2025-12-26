"""
Debug script for robot FK transformation.
Check key parameters that might cause robot Gaussian deformation.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from Gaussians.util_gau import load_ply

def main():
    print("=" * 70)
    print("Robot FK Debug")
    print("=" * 70)

    # 1. Check genesis_center
    print("\n[1] Genesis Center:")
    try:
        genesis_center = np.load('data/labels/genesis_center.npy')
        print(f"   Value: {genesis_center}")
        print(f"   Shape: {genesis_center.shape}")
    except Exception as e:
        print(f"   Error: {e}")

    # 2. Check segmentation labels
    print("\n[2] Segmentation Labels:")
    try:
        labels = np.load('data/labels/so100_labels.npy')
        print(f"   Shape: {labels.shape}")
        print(f"   Unique labels: {np.unique(labels)}")
        for i in range(7):
            count = (labels == i).sum()
            print(f"   Link {i}: {count} points ({count/len(labels)*100:.1f}%)")
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Check robot PLY coordinates
    print("\n[3] Robot PLY Coordinates:")
    try:
        robot = load_ply('exports/mult-view-scene/robot.ply')
        xyz = robot.xyz
        print(f"   Shape: {xyz.shape}")
        print(f"   Center: [{xyz[:, 0].mean():.4f}, {xyz[:, 1].mean():.4f}, {xyz[:, 2].mean():.4f}]")
        print(f"   X range: [{xyz[:, 0].min():.4f}, {xyz[:, 0].max():.4f}]")
        print(f"   Y range: [{xyz[:, 1].min():.4f}, {xyz[:, 1].max():.4f}]")
        print(f"   Z range: [{xyz[:, 2].min():.4f}, {xyz[:, 2].max():.4f}]")
    except Exception as e:
        print(f"   Error: {e}")

    # 4. Check ICP parameters
    print("\n[4] Default ICP Parameters:")
    from robot_gaussian.robot_gaussian_model import DEFAULT_ICP_ROTATION, DEFAULT_ICP_TRANSLATION, DEFAULT_GENESIS_SCALE
    print(f"   ICP Translation: {DEFAULT_ICP_TRANSLATION}")
    print(f"   Genesis Scale: {DEFAULT_GENESIS_SCALE}")
    print(f"   ICP Rotation det: {np.linalg.det(DEFAULT_ICP_ROTATION):.4f}")

    # 5. Check initial joint states
    print("\n[5] Initial Joint States:")
    initial_joints = [0, -3.32, 3.11, 1.18, 0, -0.174]
    print(f"   Config: {initial_joints}")

    # 6. Calculate expected robot position after transforms
    print("\n[6] Expected Robot Position After splat_to_world:")
    try:
        R_y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
        R_combined = DEFAULT_ICP_ROTATION @ R_y_90
        xyz_aligned = (R_combined @ xyz.T).T + DEFAULT_ICP_TRANSLATION
        print(f"   Center after alignment: [{xyz_aligned[:, 0].mean():.4f}, {xyz_aligned[:, 1].mean():.4f}, {xyz_aligned[:, 2].mean():.4f}]")
        print(f"   X range: [{xyz_aligned[:, 0].min():.4f}, {xyz_aligned[:, 0].max():.4f}]")
        print(f"   Y range: [{xyz_aligned[:, 1].min():.4f}, {xyz_aligned[:, 1].max():.4f}]")
        print(f"   Z range: [{xyz_aligned[:, 2].min():.4f}, {xyz_aligned[:, 2].max():.4f}]")
    except Exception as e:
        print(f"   Error: {e}")

    # 7. Check background PLY for comparison
    print("\n[7] Background PLY Center (for comparison):")
    try:
        bg = load_ply('exports/mult-view-scene/left-transform2.ply')
        print(f"   Center: [{bg.xyz[:, 0].mean():.4f}, {bg.xyz[:, 1].mean():.4f}, {bg.xyz[:, 2].mean():.4f}]")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    print("""
If robot center after alignment is very different from background center:
  -> Coordinate system mismatch

If genesis_center is very different from robot center:
  -> genesis_center needs to be recalculated

If segmentation labels are uneven (most points in one link):
  -> Labels might be incorrect
""")

if __name__ == "__main__":
    main()
