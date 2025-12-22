"""
Debug Alignment Script
======================

Visualize both Genesis link point clouds and robot.ply to diagnose alignment issues.
This helps identify whether the coordinate transform is correct.

Usage:
    python scripts/debug_alignment.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
import open3d as o3d
from Gaussians.util_gau import load_ply
from robot_gaussian.forward_kinematics import LINK_NAMES


def rotation_matrix(axis, angle_deg):
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    mat = np.eye(4)
    mat[:3, :3] = R
    return mat


def transform_matrix3(translation, rotation, scale):
    T_translate = np.eye(4)
    T_translate[:3, 3] = translation

    center = translation
    T_neg = np.eye(4)
    T_neg[:3, 3] = -np.array(center)
    T_pos = np.eye(4)
    T_pos[:3, 3] = center

    rx, ry, rz = rotation
    R_x = rotation_matrix('x', rx)
    R_y = rotation_matrix('y', ry)
    R_z = rotation_matrix('z', rz)

    R = T_pos @ R_x @ T_neg
    R = T_pos @ R_y @ T_neg @ R
    R = T_pos @ R_z @ T_neg @ R

    T_scale = np.eye(4)
    T_scale[:3, :3] *= scale

    return R @ T_translate @ T_scale


def transform_points(points, transform_matrix):
    ones = np.ones((len(points), 1))
    points_homo = np.hstack([points, ones])
    transformed = (transform_matrix @ points_homo.T).T
    return transformed[:, :3]


def main():
    print("=" * 60)
    print("Debug: Alignment Visualization")
    print("=" * 60)

    # 1. Initialize Genesis
    print("\n[1/4] Initializing Genesis...")
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

    # 2. Set initial pose and get link point clouds
    print("\n[2/4] Getting Genesis link point clouds...")
    INITIAL_JOINTS = [0, -3.32, 3.11, 1.18, 0, -0.174]
    arm.set_dofs_position(INITIAL_JOINTS)
    scene.step()
    print(f"   Initial joints: {INITIAL_JOINTS}")

    genesis_points = []
    genesis_colors = []
    link_colors = [
        [1.0, 0.0, 0.0],    # Base - Red
        [0.0, 1.0, 0.0],    # Rotation_Pitch - Green
        [0.0, 0.0, 1.0],    # Upper_Arm - Blue
        [1.0, 1.0, 0.0],    # Lower_Arm - Yellow
        [1.0, 0.0, 1.0],    # Wrist_Pitch_Roll - Magenta
        [0.0, 1.0, 1.0],    # Fixed_Jaw - Cyan
        [1.0, 0.5, 0.0],    # Moving_Jaw - Orange
    ]

    for i, name in enumerate(LINK_NAMES):
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        genesis_points.append(verts)
        genesis_colors.append(np.tile(link_colors[i], (len(verts), 1)))
        print(f"   {name}: {len(verts)} vertices, "
              f"range=[{verts.min():.3f}, {verts.max():.3f}]")

    genesis_all = np.vstack(genesis_points)
    genesis_colors_all = np.vstack(genesis_colors)
    print(f"\n   Total Genesis points: {len(genesis_all)}")
    print(f"   Genesis coordinate range: [{genesis_all.min():.4f}, {genesis_all.max():.4f}]")
    print(f"   Genesis center: {genesis_all.mean(axis=0)}")

    # 3. Load robot.ply
    print("\n[3/4] Loading robot.ply...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')
    robot_xyz_original = robot_gau.xyz
    print(f"   Original coordinate range: [{robot_xyz_original.min():.4f}, {robot_xyz_original.max():.4f}]")
    print(f"   Original center: {robot_xyz_original.mean(axis=0)}")

    # 4. Try different transforms
    print("\n[4/4] Creating visualization with multiple transform options...")

    # Option A: No transform (robot.ply as-is)
    pcd_robot_original = o3d.geometry.PointCloud()
    pcd_robot_original.points = o3d.utility.Vector3dVector(robot_xyz_original)
    pcd_robot_original.paint_uniform_color([0.5, 0.5, 0.5])  # Gray

    # Option B: With supersplat_transform
    splat_to_world = transform_matrix3(
        translation=[0.34, 0.09, 0.42],
        rotation=[-34.29, 11.67, -180-47.35],
        scale=0.81
    )
    robot_xyz_transformed = transform_points(robot_xyz_original, splat_to_world)
    print(f"   Transformed coordinate range: [{robot_xyz_transformed.min():.4f}, {robot_xyz_transformed.max():.4f}]")
    print(f"   Transformed center: {robot_xyz_transformed.mean(axis=0)}")

    pcd_robot_transformed = o3d.geometry.PointCloud()
    pcd_robot_transformed.points = o3d.utility.Vector3dVector(robot_xyz_transformed)
    pcd_robot_transformed.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray

    # Genesis point cloud
    pcd_genesis = o3d.geometry.PointCloud()
    pcd_genesis.points = o3d.utility.Vector3dVector(genesis_all)
    pcd_genesis.colors = o3d.utility.Vector3dVector(genesis_colors_all)

    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Print comparison
    print("\n" + "=" * 60)
    print("Coordinate Comparison")
    print("=" * 60)
    print(f"\nGenesis link points:")
    print(f"  X range: [{genesis_all[:, 0].min():.4f}, {genesis_all[:, 0].max():.4f}]")
    print(f"  Y range: [{genesis_all[:, 1].min():.4f}, {genesis_all[:, 1].max():.4f}]")
    print(f"  Z range: [{genesis_all[:, 2].min():.4f}, {genesis_all[:, 2].max():.4f}]")

    print(f"\nrobot.ply (original - no transform):")
    print(f"  X range: [{robot_xyz_original[:, 0].min():.4f}, {robot_xyz_original[:, 0].max():.4f}]")
    print(f"  Y range: [{robot_xyz_original[:, 1].min():.4f}, {robot_xyz_original[:, 1].max():.4f}]")
    print(f"  Z range: [{robot_xyz_original[:, 2].min():.4f}, {robot_xyz_original[:, 2].max():.4f}]")

    print(f"\nrobot.ply (with splat_to_world transform):")
    print(f"  X range: [{robot_xyz_transformed[:, 0].min():.4f}, {robot_xyz_transformed[:, 0].max():.4f}]")
    print(f"  Y range: [{robot_xyz_transformed[:, 1].min():.4f}, {robot_xyz_transformed[:, 1].max():.4f}]")
    print(f"  Z range: [{robot_xyz_transformed[:, 2].min():.4f}, {robot_xyz_transformed[:, 2].max():.4f}]")

    print("\n" + "=" * 60)
    print("Launching visualization...")
    print("=" * 60)
    print("\nColor legend:")
    print("  Colored points = Genesis link meshes (ground truth)")
    print("  Gray points = robot.ply (original, no transform)")
    print("  Light gray = robot.ply (with splat_to_world)")
    print("\nIf robot.ply overlaps Genesis, segmentation should work.")
    print("If not, we need to find the correct transform or initial pose.")

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd_genesis, pcd_robot_original, coord_frame],
        window_name="Genesis (colored) vs robot.ply original (gray)",
        width=1200,
        height=800
    )

    o3d.visualization.draw_geometries(
        [pcd_genesis, pcd_robot_transformed, coord_frame],
        window_name="Genesis (colored) vs robot.ply transformed (gray)",
        width=1200,
        height=800
    )


if __name__ == "__main__":
    main()
