"""
Test Robot Gaussian Model
==========================

Simple test to verify that the splat_to_world transformation and FK are working correctly.

Key insight: robot.ply is aligned to Genesis*0.8 (scaled Genesis).
We test FK in the scaled coordinate system, comparing with scaled Genesis.

Usage:
    python scripts/test_robot_gs.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
import torch
import open3d as o3d
from robot_gaussian.robot_gaussian_model import RobotGaussianConfig, RobotGaussianModel
from robot_gaussian.forward_kinematics import LINK_NAMES, get_transformation_list_scaled


def transform_matrix3(translation=(0.5, 0.5, -0.5), rotation=(30, 60, -180), scale=1.0):
    """Build transformation matrix (same as base_task.py)."""
    def rotation_matrix(axis, angle_deg):
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        if axis == 'x':
            return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
        elif axis == 'y':
            return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
        elif axis == 'z':
            return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

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

    S = T_pos @ T_scale @ T_neg
    return S @ R @ T_translate


def get_genesis_points_scaled(arm, link_names, genesis_center, scale_factor=0.8):
    """Get Genesis mesh vertices scaled by scale_factor around genesis_center."""
    genesis_points = []
    for name in link_names:
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        genesis_points.append(verts)
    genesis_all = np.vstack(genesis_points)

    # Scale around genesis_center
    genesis_scaled = (genesis_all - genesis_center) * scale_factor + genesis_center
    return genesis_scaled


def main():
    print("=" * 60)
    print("Testing Robot Gaussian Model (Scaled FK Approach)")
    print("=" * 60)

    # 1. Initialize Genesis
    print("\n[1/5] Initializing Genesis...")
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
    print("   Genesis initialized")

    # 2. Create config
    print("\n[2/5] Creating Robot Gaussian config...")
    supersplat_transform = transform_matrix3(
        translation=[0.34, 0.09, 0.42],
        rotation=[-34.29, 11.67, -180-47.35],
        scale=0.81
    )
    world_to_splat = np.linalg.inv(supersplat_transform)

    config = RobotGaussianConfig(
        robot_ply_path='exports/mult-view-scene/robot.ply',
        labels_path='data/labels/so100_labels.npy',
        initial_joint_states=[0, -3.32, 3.11, 1.18, 0, -0.174],
        world_to_splat=world_to_splat
    )
    print(f"   Initial joints: {config.initial_joint_states}")
    print(f"   Genesis scale factor: {config.genesis_scale}")

    # Load genesis_center
    genesis_center = np.load(config.genesis_center_path)
    print(f"   Genesis center: {genesis_center}")

    # 3. Initialize Robot Gaussian Model
    print("\n[3/5] Initializing Robot Gaussian Model...")
    # CRITICAL: Set arm to initial pose AND step BEFORE creating RobotGaussianModel
    arm.set_dofs_position(config.initial_joint_states)
    scene.step()
    robot_model = RobotGaussianModel(config, arm)

    print(f"   Robot Gaussians: {len(robot_model.gaussians.xyz)}")
    print(f"   Segments: {[len(s) for s in robot_model.segmented_list]}")

    # 4. Test static alignment (compare with SCALED Genesis)
    print("\n[4/5] Testing static alignment (scaled coordinates)...")

    # Get SCALED Genesis at initial pose
    genesis_scaled = get_genesis_points_scaled(arm, LINK_NAMES, genesis_center, config.genesis_scale)

    # Get robot Gaussian positions (in scaled world coords, NO inverse scaling)
    robot_xyz = robot_model.backup_xyz.cpu().numpy()

    print(f"   Genesis*0.8 range: X=[{genesis_scaled[:,0].min():.4f}, {genesis_scaled[:,0].max():.4f}]")
    print(f"   Genesis*0.8 range: Z=[{genesis_scaled[:,2].min():.4f}, {genesis_scaled[:,2].max():.4f}]")
    print(f"   Robot GS range: X=[{robot_xyz[:,0].min():.4f}, {robot_xyz[:,0].max():.4f}]")
    print(f"   Robot GS range: Z=[{robot_xyz[:,2].min():.4f}, {robot_xyz[:,2].max():.4f}]")

    # Compute alignment error
    pcd_genesis = o3d.geometry.PointCloud()
    pcd_genesis.points = o3d.utility.Vector3dVector(genesis_scaled)
    pcd_robot = o3d.geometry.PointCloud()
    pcd_robot.points = o3d.utility.Vector3dVector(robot_xyz)

    dists = pcd_robot.compute_point_cloud_distance(pcd_genesis)
    mean_dist = np.mean(dists)
    print(f"   Mean alignment error: {mean_dist:.4f}")

    # 5. Test FK by changing joint angles
    print("\n[5/5] Testing FK with new joint angles...")

    # Set new joint angles
    new_joints = [0.5, -2.5, 2.5, 0.8, 0.3, -0.1]
    arm.set_dofs_position(new_joints)
    scene.step()

    # Get SCALED Genesis at new pose
    genesis_new_scaled = get_genesis_points_scaled(arm, LINK_NAMES, genesis_center, config.genesis_scale)

    # Apply SCALED FK to robot Gaussians
    transformations = get_transformation_list_scaled(
        arm, robot_model.initial_link_states, LINK_NAMES,
        genesis_center, config.genesis_scale
    )

    # Start from backup (scaled world coords)
    robot_xyz_fk = robot_model.backup_xyz.clone()

    # Apply FK transforms per link (in scaled space)
    for joint_idx, (R_rel, T_scaled) in enumerate(transformations):
        segment = robot_model.segmented_list[joint_idx + 1]  # +1 to skip Base
        if len(segment) > 0:
            robot_xyz_fk[segment] = (R_rel @ robot_xyz_fk[segment].T).T + T_scaled

    robot_xyz_fk_np = robot_xyz_fk.cpu().numpy()

    print(f"   Genesis*0.8 new pose X range: [{genesis_new_scaled[:,0].min():.4f}, {genesis_new_scaled[:,0].max():.4f}]")
    print(f"   Robot FK X range: [{robot_xyz_fk_np[:,0].min():.4f}, {robot_xyz_fk_np[:,0].max():.4f}]")

    # Compute alignment error at new pose
    pcd_genesis_new = o3d.geometry.PointCloud()
    pcd_genesis_new.points = o3d.utility.Vector3dVector(genesis_new_scaled)
    pcd_robot_fk = o3d.geometry.PointCloud()
    pcd_robot_fk.points = o3d.utility.Vector3dVector(robot_xyz_fk_np)

    dists_fk = pcd_robot_fk.compute_point_cloud_distance(pcd_genesis_new)
    mean_dist_fk = np.mean(dists_fk)
    print(f"   Mean FK alignment error: {mean_dist_fk:.4f}")

    # Save point clouds for visualization
    print("\n" + "=" * 60)
    print("Saving point clouds for visualization...")

    # Save Genesis*0.8 at initial pose (gray)
    pcd_genesis.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.io.write_point_cloud("data/test_genesis_initial.ply", pcd_genesis)

    # Save robot GS at initial pose (green)
    pcd_robot.paint_uniform_color([0.0, 0.8, 0.0])
    o3d.io.write_point_cloud("data/test_robot_initial.ply", pcd_robot)

    # Save Genesis*0.8 at NEW pose (gray)
    pcd_genesis_new.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.io.write_point_cloud("data/test_genesis_new.ply", pcd_genesis_new)

    # Save robot GS after FK (blue)
    pcd_robot_fk.paint_uniform_color([0.0, 0.0, 0.8])
    o3d.io.write_point_cloud("data/test_robot_fk.ply", pcd_robot_fk)

    print("   Saved: data/test_genesis_initial.ply (gray) - Genesis*0.8 initial pose")
    print("   Saved: data/test_robot_initial.ply (green) - robot.ply initial pose")
    print("   Saved: data/test_genesis_new.ply (gray) - Genesis*0.8 new pose")
    print("   Saved: data/test_robot_fk.ply (blue) - robot.ply FK result")
    print("\nTo visualize initial alignment:")
    print("   python scripts/view_ply.py data/test_genesis_initial.ply data/test_robot_initial.ply")
    print("\nTo visualize FK result:")
    print("   python scripts/view_ply.py data/test_genesis_new.ply data/test_robot_fk.ply")
    print("=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    main()
