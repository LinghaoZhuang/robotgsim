"""
Visualize Genesis Ground Truth Labels
======================================

Show Genesis mesh vertices colored by link (the "ground truth" for KNN training).

Usage:
    python scripts/visualize_genesis_labels.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
import open3d as o3d
from robot_gaussian.forward_kinematics import LINK_NAMES


def main():
    print("=" * 60)
    print("Visualizing Genesis Ground Truth Labels")
    print("=" * 60)

    # Initialize Genesis
    print("\n[1/3] Initializing Genesis...")
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(substeps=60),
        renderer=gs.renderers.Rasterizer(),
    )
    scene.add_entity(gs.morphs.Plane())
    arm = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="./assets/so100/urdf/so_arm100.xml",
            euler=(0.0, 0.0, 90.0),
            pos=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(),
    )
    scene.build()

    # Set to the same pose used for segmentation
    INITIAL_JOINTS = [0, -3.32, 3.11, 1.18, 0, -0.174]
    arm.set_dofs_position(INITIAL_JOINTS)
    scene.step()
    print(f"   Joint pose: {INITIAL_JOINTS}")

    # Get link point clouds with colors
    print("\n[2/3] Extracting Genesis link vertices...")

    colors_map = np.array([
        [1.0, 0.0, 0.0],    # Base - Red
        [0.0, 1.0, 0.0],    # Rotation_Pitch - Green
        [0.0, 0.0, 1.0],    # Upper_Arm - Blue
        [1.0, 1.0, 0.0],    # Lower_Arm - Yellow
        [1.0, 0.0, 1.0],    # Wrist_Pitch_Roll - Magenta
        [0.0, 1.0, 1.0],    # Fixed_Jaw - Cyan
        [1.0, 0.5, 0.0],    # Moving_Jaw - Orange
    ])

    all_points = []
    all_colors = []

    print("\n   Link vertex counts (Ground Truth):")
    for i, name in enumerate(LINK_NAMES):
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        all_points.append(verts)
        all_colors.append(np.tile(colors_map[i], (len(verts), 1)))

        color_name = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange"][i]
        print(f"   {i}. {name:20s}: {len(verts):6d} vertices - {color_name}")

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    print(f"\n   Total: {len(all_points)} vertices")

    # Create point cloud
    print("\n[3/3] Creating visualization...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Save to PLY for viewing
    output_path = "data/genesis_ground_truth.ply"
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"   Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Genesis Ground Truth Labels (Training Data for KNN)")
    print("=" * 60)
    print("\nColor legend:")
    for i, name in enumerate(LINK_NAMES):
        color_name = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange"][i]
        print(f"  {color_name:10s} - {name}")

    print("\nLaunching visualization...")
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name="Genesis Ground Truth (KNN Training Data)",
        width=1200,
        height=800
    )


if __name__ == "__main__":
    main()
