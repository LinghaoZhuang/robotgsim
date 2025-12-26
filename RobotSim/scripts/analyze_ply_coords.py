"""
Analyze coordinate relationship between robot.ply and background PLY.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from Gaussians.util_gau import load_ply

def analyze_ply(path, name):
    """Analyze a PLY file's coordinates."""
    try:
        gau = load_ply(path)
        xyz = gau.xyz
        print(f"\n{name}:")
        print(f"  Points: {len(xyz)}")
        print(f"  Center: [{xyz[:, 0].mean():.4f}, {xyz[:, 1].mean():.4f}, {xyz[:, 2].mean():.4f}]")
        print(f"  X: [{xyz[:, 0].min():.4f}, {xyz[:, 0].max():.4f}] (range: {xyz[:, 0].max() - xyz[:, 0].min():.4f})")
        print(f"  Y: [{xyz[:, 1].min():.4f}, {xyz[:, 1].max():.4f}] (range: {xyz[:, 1].max() - xyz[:, 1].min():.4f})")
        print(f"  Z: [{xyz[:, 2].min():.4f}, {xyz[:, 2].max():.4f}] (range: {xyz[:, 2].max() - xyz[:, 2].min():.4f})")
        return xyz
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    print("=" * 70)
    print("PLY Coordinate Analysis")
    print("=" * 70)

    # Background PLYs
    bg_left = analyze_ply('exports/mult-view-scene/left-transform2.ply', 'Background (left-transform2.ply)')

    # Robot PLYs
    robot = analyze_ply('exports/mult-view-scene/robot.ply', 'Robot (robot.ply)')

    # Also check robot-right-3.ply
    robot_right = analyze_ply('exports/mult-view-scene/robot-right-3.ply', 'Robot-right-3 (robot-right-3.ply)')

    # Compare
    if bg_left is not None and robot is not None:
        print("\n" + "=" * 70)
        print("Analysis")
        print("=" * 70)

        # Find points in background that are roughly at robot height (Z > 0)
        bg_above_table = bg_left[bg_left[:, 2] > 0]
        print(f"\nBackground points above table (Z > 0): {len(bg_above_table)} / {len(bg_left)}")
        if len(bg_above_table) > 0:
            print(f"  Center: [{bg_above_table[:, 0].mean():.4f}, {bg_above_table[:, 1].mean():.4f}, {bg_above_table[:, 2].mean():.4f}]")

        # Estimate required transform
        bg_center = bg_left.mean(axis=0)
        robot_center = robot.mean(axis=0)
        translation_needed = bg_center - robot_center
        print(f"\nEstimated translation to align robot center to BG center:")
        print(f"  {translation_needed}")

        # Check scale difference
        bg_range = bg_left.max(axis=0) - bg_left.min(axis=0)
        robot_range = robot.max(axis=0) - robot.min(axis=0)
        print(f"\nCoordinate ranges:")
        print(f"  Background: {bg_range}")
        print(f"  Robot: {robot_range}")
        print(f"  Ratio (BG/Robot): {bg_range / robot_range}")

if __name__ == "__main__":
    main()
