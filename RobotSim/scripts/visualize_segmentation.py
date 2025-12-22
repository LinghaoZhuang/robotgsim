"""
Segmentation Visualization Script
==================================

Visualize KNN segmentation results by coloring each link differently.

Usage:
    python scripts/visualize_segmentation.py

Prerequisites:
- data/labels/so100_labels.npy must exist (run segment_robot.py first)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import open3d as o3d
from Gaussians.util_gau import load_ply
from robot_gaussian.forward_kinematics import LINK_NAMES


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
    """Build splat_to_world transformation matrix."""
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
    """Get transformation from Splat to World coordinates."""
    return transform_matrix3(
        translation=[0.34, 0.09, 0.42],
        rotation=[-34.29, 11.67, -180-47.35],
        scale=0.81
    )


def transform_points(points, transform_matrix):
    """Apply 4x4 transformation matrix to Nx3 points."""
    ones = np.ones((len(points), 1))
    points_homo = np.hstack([points, ones])
    transformed = (transform_matrix @ points_homo.T).T
    return transformed[:, :3]


def main():
    print("=" * 60)
    print("Visualizing Robot Segmentation")
    print("=" * 60)

    # Load segmentation results
    print("\n[1/3] Loading segmentation labels...")
    labels_path = 'data/labels/so100_labels.npy'
    try:
        labels = np.load(labels_path)
        print(f"   Loaded labels from: {labels_path}")
        print(f"   Shape: {labels.shape}")
    except FileNotFoundError:
        print(f"   ERROR: {labels_path} not found!")
        print("   Please run 'python scripts/segment_robot.py' first")
        sys.exit(1)

    # Load robot.ply and apply axis rotation
    print("\n[2/3] Loading robot.ply and applying axis rotation...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')
    print(f"   Loaded {len(robot_gau.xyz)} Gaussians")

    # Apply same rotation as segment_robot.py: 90° around Y axis
    R_y_90 = np.array([
        [0,  0,  1],
        [0,  1,  0],
        [-1, 0,  0]
    ])
    robot_xyz = (R_y_90 @ robot_gau.xyz.T).T

    # Assign colors to each link
    print("\n[3/3] Creating colored point cloud...")
    colors = np.array([
        [1.0, 0.0, 0.0],    # Base - Red
        [0.0, 1.0, 0.0],    # Rotation_Pitch - Green
        [0.0, 0.0, 1.0],    # Upper_Arm - Blue
        [1.0, 1.0, 0.0],    # Lower_Arm - Yellow
        [1.0, 0.0, 1.0],    # Wrist_Pitch_Roll - Magenta
        [0.0, 1.0, 1.0],    # Fixed_Jaw - Cyan
        [1.0, 0.5, 0.0],    # Moving_Jaw - Orange
    ])

    # Create colored point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(robot_xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors[labels])

    # Print statistics
    print("\n   Gaussian distribution by link:")
    for i, name in enumerate(LINK_NAMES):
        count = np.sum(labels == i)
        percentage = count / len(labels) * 100
        color_name = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange"][i]
        print(f"   {i}. {name:20s}: {count:6d} ({percentage:5.1f}%) - {color_name}")

    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Visualize
    print("\n" + "=" * 60)
    print("Launching visualization...")
    print("=" * 60)
    print("\nColor legend:")
    for i, name in enumerate(LINK_NAMES):
        color_name = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange"][i]
        print(f"  {color_name:10s} - {name}")
    print("\nCheck that:")
    print("  - Each link forms a continuous color block")
    print("  - No scattered isolated points")
    print("  - Clear boundaries between links")

    o3d.visualization.draw_geometries(
        [pcd, coordinate_frame],
        window_name="Robot Segmentation Visualization",
        width=1200,
        height=800
    )


if __name__ == "__main__":
    main()
