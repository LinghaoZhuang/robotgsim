"""
Simple PLY Viewer
=================

View PLY files with Open3D (run separately from Genesis).

Usage:
    python scripts/view_ply.py <ply_file> [ply_file2] ...

Examples:
    python scripts/view_ply.py data/genesis_ground_truth.ply
    python scripts/view_ply.py data/genesis_ground_truth.ply data/robot_segmented.ply
"""

import sys
import open3d as o3d
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/view_ply.py <ply_file> [ply_file2] ...")
        print("\nAvailable PLY files:")
        print("  data/genesis_ground_truth.ply  - Genesis ground truth labels")
        print("  data/robot_segmented.ply       - Robot.ply with segmentation")
        sys.exit(1)

    geometries = []

    for i, ply_path in enumerate(sys.argv[1:]):
        print(f"Loading: {ply_path}")
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            print(f"  Points: {len(pcd.points)}")

            # Shift multiple point clouds for side-by-side view
            if len(sys.argv) > 2:
                shift = (i - (len(sys.argv) - 2) / 2) * 0.5
                pcd.translate([shift, 0, 0])
                print(f"  Shifted by X={shift:.2f} for comparison")

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

    print("\nLaunching viewer...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="PLY Viewer",
        width=1200,
        height=800
    )


if __name__ == "__main__":
    main()
