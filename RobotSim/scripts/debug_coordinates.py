"""
Debug script to check coordinate systems of different components.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from Gaussians.util_gau import load_ply

def print_xyz_stats(name, xyz):
    """Print coordinate statistics."""
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()
    print(f"\n{name}:")
    print(f"  Shape: {xyz.shape}")
    print(f"  Center: [{xyz[:, 0].mean():.4f}, {xyz[:, 1].mean():.4f}, {xyz[:, 2].mean():.4f}]")
    print(f"  X range: [{xyz[:, 0].min():.4f}, {xyz[:, 0].max():.4f}]")
    print(f"  Y range: [{xyz[:, 1].min():.4f}, {xyz[:, 1].max():.4f}]")
    print(f"  Z range: [{xyz[:, 2].min():.4f}, {xyz[:, 2].max():.4f}]")

def main():
    print("=" * 70)
    print("Coordinate System Debug")
    print("=" * 70)

    # 1. Background PLY
    print("\n[1] Background PLY files:")
    try:
        bg_left = load_ply('exports/mult-view-scene/left-transform2.ply')
        print_xyz_stats("Background Left (left-transform2.ply)", bg_left.xyz)
    except Exception as e:
        print(f"  Error loading left background: {e}")

    try:
        bg_right = load_ply('exports/mult-view-scene/right-transform.ply')
        print_xyz_stats("Background Right (right-transform.ply)", bg_right.xyz)
    except Exception as e:
        print(f"  Error loading right background: {e}")

    # 2. Robot PLY
    print("\n[2] Robot PLY file:")
    try:
        robot = load_ply('exports/mult-view-scene/robot.ply')
        print_xyz_stats("Robot PLY (robot.ply)", robot.xyz)
    except Exception as e:
        print(f"  Error loading robot: {e}")

    # 3. Object PLY files
    print("\n[3] Object PLY files:")
    object_files = [
        'assets/so100/ply/banana.ply',
        'assets/so100/ply/box.ply',
    ]
    for obj_path in object_files:
        try:
            obj = load_ply(obj_path)
            print_xyz_stats(f"Object ({obj_path})", obj.xyz)
        except Exception as e:
            print(f"  Error loading {obj_path}: {e}")

    # 4. Genesis camera info (if running with Genesis)
    print("\n[4] Genesis camera expected positions:")
    print("  Left camera pos: from init_gs (computed from nerfstudio + supersplat)")
    print("  Expected range for objects in Genesis: ~[0, 0.5] for X, ~[-0.3, 0.3] for Y")

    # 5. Summary
    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    print("""
If background and robot PLY have similar coordinate ranges:
  -> They are in the same coordinate system (good)

If robot PLY center is very different from background:
  -> Coordinate system mismatch (need transformation)

If object PLY coordinates are very different from background:
  -> Objects are in a different coordinate system (need ICP or transform)
""")

if __name__ == "__main__":
    main()
