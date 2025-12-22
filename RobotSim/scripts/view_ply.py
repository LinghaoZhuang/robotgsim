"""
Simple PLY Viewer
=================

View PLY files with Open3D (run separately from Genesis).

Usage:
    python scripts/view_ply.py <ply_file> [ply_file2] ...
    python scripts/view_ply.py --side-by-side <ply_file1> <ply_file2>   # 并排显示

Examples:
    python scripts/view_ply.py data/genesis_ground_truth.ply
    python scripts/view_ply.py data/genesis_scaled.ply data/robot_icp_aligned.ply  # 重叠显示
    python scripts/view_ply.py --side-by-side data/genesis_ground_truth.ply data/robot_segmented.ply
"""

import sys
import open3d as o3d
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/view_ply.py <ply_file> [ply_file2] ...")
        print("       python scripts/view_ply.py --side-by-side <ply1> <ply2>  # 并排显示")
        print("\nAvailable PLY files:")
        print("  data/genesis_ground_truth.ply  - Genesis ground truth labels")
        print("  data/robot_segmented.ply       - Robot.ply with segmentation")
        print("  data/genesis_scaled.ply        - Genesis scaled 0.8x")
        print("  data/robot_icp_aligned.ply     - Robot.ply after ICP")
        sys.exit(1)

    # Check for side-by-side flag
    side_by_side = False
    files = sys.argv[1:]
    if "--side-by-side" in files:
        side_by_side = True
        files.remove("--side-by-side")

    geometries = []

    for i, ply_path in enumerate(files):
        print(f"Loading: {ply_path}")
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            print(f"  Points: {len(pcd.points)}")

            # Only shift if side-by-side mode
            if side_by_side and len(files) > 1:
                shift = (i - (len(files) - 1) / 2) * 0.5
                pcd.translate([shift, 0, 0])
                print(f"  Shifted by X={shift:.2f} for side-by-side view")

            geometries.append(pcd)
        except Exception as e:
            print(f"  Error: {e}")

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(coord_frame)

    print("\nColor legend:")
    print("  Red     - Base")
    print("  Green   - Rotation_Pitch")
    print("  Blue    - Upper_Arm")
    print("  Yellow  - Lower_Arm")
    print("  Magenta - Wrist_Pitch_Roll")
    print("  Cyan    - Fixed_Jaw")
    print("  Orange  - Moving_Jaw")
    print("  Gray    - Genesis mesh")

    mode = "Side-by-side" if side_by_side else "Overlapped"
    print(f"\nMode: {mode}")
    print("Launching viewer...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"PLY Viewer ({mode})",
        width=1200,
        height=800
    )


if __name__ == "__main__":
    main()
