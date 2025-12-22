"""
Compare Genesis Ground Truth vs Robot.ply Segmentation
========================================================

Side-by-side visualization to see alignment and segmentation quality.

Usage:
    python scripts/compare_segmentation.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
import open3d as o3d
from Gaussians.util_gau import load_ply
from robot_gaussian.forward_kinematics import LINK_NAMES


def main():
    print("=" * 70)
    print("Compare: Genesis Ground Truth vs Robot.ply Segmentation")
    print("=" * 70)

    # Colors
    colors_map = np.array([
        [1.0, 0.0, 0.0],    # Base - Red
        [0.0, 1.0, 0.0],    # Rotation_Pitch - Green
        [0.0, 0.0, 1.0],    # Upper_Arm - Blue
        [1.0, 1.0, 0.0],    # Lower_Arm - Yellow
        [1.0, 0.0, 1.0],    # Wrist_Pitch_Roll - Magenta
        [0.0, 1.0, 1.0],    # Fixed_Jaw - Cyan
        [1.0, 0.5, 0.0],    # Moving_Jaw - Orange
    ])

    # 1. Get Genesis ground truth
    print("\n[1/3] Loading Genesis ground truth...")
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

    INITIAL_JOINTS = [0, -3.32, 3.11, 1.18, 0, -0.174]
    arm.set_dofs_position(INITIAL_JOINTS)
    scene.step()

    genesis_points = []
    genesis_colors = []
    for i, name in enumerate(LINK_NAMES):
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        genesis_points.append(verts)
        genesis_colors.append(np.tile(colors_map[i], (len(verts), 1)))

    genesis_points = np.vstack(genesis_points)
    genesis_colors = np.vstack(genesis_colors)
    print(f"   Genesis: {len(genesis_points)} vertices")

    # 2. Load robot.ply with segmentation
    print("\n[2/3] Loading robot.ply segmentation...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')

    # Apply transformation (same as segment_robot.py)
    R_y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    robot_xyz = (R_y_90 @ robot_gau.xyz.T).T
    translation = np.array([-0.0375, -0.0238, 0.0886])
    robot_xyz = robot_xyz + translation

    # Load labels
    labels = np.load('data/labels/so100_labels.npy')
    robot_colors = colors_map[labels]
    print(f"   Robot.ply: {len(robot_xyz)} Gaussians")

    # 3. Create visualizations
    print("\n[3/3] Creating comparison visualization...")

    # Genesis point cloud (shift left for comparison)
    pcd_genesis = o3d.geometry.PointCloud()
    pcd_genesis.points = o3d.utility.Vector3dVector(genesis_points - np.array([0.3, 0, 0]))
    pcd_genesis.colors = o3d.utility.Vector3dVector(genesis_colors)

    # Robot.ply point cloud (shift right for comparison)
    pcd_robot = o3d.geometry.PointCloud()
    pcd_robot.points = o3d.utility.Vector3dVector(robot_xyz + np.array([0.3, 0, 0]))
    pcd_robot.colors = o3d.utility.Vector3dVector(robot_colors)

    # Overlapped version (no shift)
    pcd_genesis_overlap = o3d.geometry.PointCloud()
    pcd_genesis_overlap.points = o3d.utility.Vector3dVector(genesis_points)
    pcd_genesis_overlap.colors = o3d.utility.Vector3dVector(genesis_colors)

    pcd_robot_overlap = o3d.geometry.PointCloud()
    pcd_robot_overlap.points = o3d.utility.Vector3dVector(robot_xyz)
    pcd_robot_overlap.colors = o3d.utility.Vector3dVector(robot_colors)

    # Coordinate frames
    coord_left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    coord_left.translate([-0.3, 0, 0])
    coord_right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    coord_right.translate([0.3, 0, 0])
    coord_center = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Print statistics
    print("\n" + "=" * 70)
    print("Comparison Statistics")
    print("=" * 70)
    print(f"\n{'Link':<20} {'Genesis':<12} {'Robot.ply':<12} {'Diff':<10}")
    print("-" * 54)
    for i, name in enumerate(LINK_NAMES):
        g_count = np.sum(genesis_colors[:, 0] == colors_map[i, 0])
        # This is approximate since we're checking by color
        r_count = np.sum(labels == i)
        diff = r_count - g_count
        print(f"{name:<20} {g_count:<12} {r_count:<12} {diff:+d}")

    print("\n" + "=" * 70)
    print("Launching visualizations...")
    print("=" * 70)
    print("\nWindow 1: Side-by-side (Genesis LEFT, Robot.ply RIGHT)")
    print("Window 2: Overlapped (to see alignment)")

    # Show side-by-side
    o3d.visualization.draw_geometries(
        [pcd_genesis, pcd_robot, coord_left, coord_right],
        window_name="LEFT: Genesis Ground Truth | RIGHT: Robot.ply Segmentation",
        width=1400,
        height=800
    )

    # Show overlapped
    o3d.visualization.draw_geometries(
        [pcd_genesis_overlap, pcd_robot_overlap, coord_center],
        window_name="Overlapped: Genesis (dense) + Robot.ply (sparse)",
        width=1200,
        height=800
    )


if __name__ == "__main__":
    main()
