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

    # 2. Set initial pose (matching robot.ply capture pose)
    print("\n[2/6] Setting initial pose...")
    # ICP analysis shows robot.ply was captured at this pose, not zero pose
    INITIAL_JOINTS = [0, -3.32, 3.11, 1.18, 0, -0.174]
    arm.set_dofs_position(INITIAL_JOINTS)
    scene.step()
    print(f"   Initial joints: {INITIAL_JOINTS}")

    # 3. Get link point clouds (world coordinates) and scale by 0.8
    print("\n[3/6] Extracting link point clouds...")
    point_clouds = get_link_point_clouds_genesis(arm, LINK_NAMES)

    # Scale Genesis mesh by 0.8 around center
    all_points = np.vstack(point_clouds)
    genesis_center = all_points.mean(axis=0)
    scale_factor = 0.8

    scaled_point_clouds = []
    for pc in point_clouds:
        scaled_pc = (pc - genesis_center) * scale_factor + genesis_center
        scaled_point_clouds.append(scaled_pc)
    point_clouds = scaled_point_clouds

    print(f"   Applied scale factor: {scale_factor}")
    for i, (name, pc) in enumerate(zip(LINK_NAMES, point_clouds)):
        print(f"   {i}. {name:20s}: {len(pc):6d} vertices")

    # 4. Train KNN
    print("\n[4/6] Training KNN classifier...")
    knn = train_segmentation_knn(point_clouds, n_neighbors=10)
    print("   KNN training completed")

    # 5. Load robot.ply and apply full transformation
    print("\n[5/6] Loading robot.ply and applying transformation...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')
    print(f"   Loaded {len(robot_gau.xyz)} Gaussians")
    print(f"   Original coordinate range: [{robot_gau.xyz.min():.4f}, {robot_gau.xyz.max():.4f}]")

    # Step 1: Rotate 90° around Y axis
    R_y_90 = np.array([
        [0,  0,  1],
        [0,  1,  0],
        [-1, 0,  0]
    ])
    robot_xyz = (R_y_90 @ robot_gau.xyz.T).T

    # Step 2: Apply ICP-refined rotation (from icp_with_scale.py)
    icp_rotation = np.array([
        [ 0.98012743, -0.09234054,  0.17556605],
        [ 0.09228557,  0.99569634,  0.00849548],
        [-0.17559495,  0.00787556,  0.984431  ]
    ])
    robot_xyz = (icp_rotation @ robot_xyz.T).T

    # Step 3: Apply ICP-refined translation
    icp_translation = np.array([0.0021, -0.0206, 0.1048])
    robot_xyz = robot_xyz + icp_translation

    print(f"   Transformed X range: [{robot_xyz[:,0].min():.4f}, {robot_xyz[:,0].max():.4f}]")
    print(f"   Transformed Z range: [{robot_xyz[:,2].min():.4f}, {robot_xyz[:,2].max():.4f}]")

    # 6. Segment using transformed coordinates
    print("\n[6/6] Segmenting Gaussians...")
    labels = segment_gaussians(robot_xyz, knn)

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
