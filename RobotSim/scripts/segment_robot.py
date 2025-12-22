"""
Robot Segmentation Script
==========================

Run KNN segmentation to assign each Gaussian in robot.ply to a link.

Usage:
    python scripts/segment_robot.py

Prerequisites:
1. Genesis initialized
2. Robot at initial pose [0, -3.32, 3.11, 1.18, 0, -0.174]
3. robot.ply exists at exports/mult-view-scene/robot.ply

IMPORTANT: robot.ply is in Splat coordinate system, Genesis link point clouds
are in World coordinate system. We must apply splat_to_world transform before
KNN segmentation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
from robot_gaussian.segmentation import (
    get_link_point_clouds_genesis,
    train_segmentation_knn,
    segment_gaussians,
    LINK_NAMES
)
from Gaussians.util_gau import load_ply


def rotation_matrix(axis, angle_deg):
    """Create 4x4 rotation matrix around specified axis."""
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


def transform_matrix3(translation=(0.5, 0.5, -0.5), rotation=(30, 60, -180), scale=1.0):
    """
    Build splat_to_world transformation matrix (from SuperSplat parameters).
    This is the same transform used in base_task.py for background PLY alignment.
    """
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


def get_splat_to_world_transform():
    """
    Get the transformation matrix from Splat coordinates to World coordinates.
    These parameters must match base_task.py's supersplat_transform.
    """
    supersplat_transform = transform_matrix3(
        translation=[0.34, 0.09, 0.42],
        rotation=[-34.29, 11.67, -180-47.35],
        scale=0.81
    )
    return supersplat_transform


def transform_points(points, transform_matrix):
    """Apply 4x4 transformation matrix to Nx3 points."""
    ones = np.ones((len(points), 1))
    points_homo = np.hstack([points, ones])  # (N, 4)
    transformed = (transform_matrix @ points_homo.T).T  # (N, 4)
    return transformed[:, :3]


def main():
    print("=" * 60)
    print("Robot Gaussian Segmentation")
    print("=" * 60)

    # 1. Initialize Genesis
    print("\n[1/6] Initializing Genesis...")
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(substeps=60),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=0.35,
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    # Add plane
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0))
    )

    # Add robot arm
    arm = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="./assets/so100/urdf/so_arm100.xml",
            euler=(0.0, 0.0, 90.0),
            pos=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(),
    )

    scene.build()
    print("   Genesis initialized successfully")

    # 2. Set initial pose
    print("\n[2/6] Setting initial pose...")
    INITIAL_JOINTS = [0, -3.32, 3.11, 1.18, 0, -0.174]
    arm.set_dofs_position(INITIAL_JOINTS)
    scene.step()
    print(f"   Initial joints: {INITIAL_JOINTS}")

    # 3. Get link point clouds (world coordinates)
    print("\n[3/6] Extracting link point clouds...")
    point_clouds = get_link_point_clouds_genesis(arm, LINK_NAMES)

    for i, (name, pc) in enumerate(zip(LINK_NAMES, point_clouds)):
        print(f"   {i}. {name:20s}: {len(pc):6d} vertices")

    # 4. Train KNN
    print("\n[4/6] Training KNN classifier...")
    knn = train_segmentation_knn(point_clouds, n_neighbors=10)
    print("   KNN training completed")

    # 5. Load robot.ply and transform to world coordinates
    print("\n[5/6] Loading robot.ply and transforming to world coordinates...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')
    print(f"   Loaded {len(robot_gau.xyz)} Gaussians")
    print(f"   Splat coordinate range: [{robot_gau.xyz.min():.4f}, {robot_gau.xyz.max():.4f}]")

    # Transform robot.ply from Splat to World coordinates
    splat_to_world = get_splat_to_world_transform()
    robot_xyz_world = transform_points(robot_gau.xyz, splat_to_world)
    print(f"   World coordinate range: [{robot_xyz_world.min():.4f}, {robot_xyz_world.max():.4f}]")

    # 6. Segment using world coordinates
    print("\n[6/6] Segmenting Gaussians (in world coordinates)...")
    labels = segment_gaussians(robot_xyz_world, knn)

    # Save labels
    output_path = 'data/labels/so100_labels.npy'
    np.save(output_path, labels)
    print(f"   Saved labels to: {output_path}")
    print(f"   Shape: {labels.shape}")
    print(f"   Unique labels: {np.unique(labels)}")

    # Print distribution
    print("\n   Gaussian distribution by link:")
    for i, name in enumerate(LINK_NAMES):
        count = np.sum(labels == i)
        percentage = count / len(labels) * 100
        print(f"   {i}. {name:20s}: {count:6d} ({percentage:5.1f}%)")

    # Optional: Visualize segmentation
    print("\n" + "=" * 60)
    print("Segmentation complete!")
    print("=" * 60)
    print("\nTo visualize segmentation, run:")
    print("   python scripts/visualize_segmentation.py")


if __name__ == "__main__":
    main()
