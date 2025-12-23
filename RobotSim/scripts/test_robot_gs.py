"""
Test Robot Gaussian Model
==========================

Simple test to verify that the splat_to_world transformation and FK are working correctly.

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
from robot_gaussian.forward_kinematics import LINK_NAMES


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


def main():
    print("=" * 60)
    print("Testing Robot Gaussian Model")
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
    print(f"   ICP translation: {config.icp_translation}")

    # 3. Initialize Robot Gaussian Model
    print("\n[3/5] Initializing Robot Gaussian Model...")
    scene.step()  # Ensure arm is at initial pose
    robot_model = RobotGaussianModel(config, arm)
    scene.step()  # Step after model initialization

    print(f"   Robot Gaussians: {len(robot_model.gaussians.xyz)}")
    print(f"   Segments: {[len(s) for s in robot_model.segmented_list]}")

    # 4. Test static alignment
    print("\n[4/5] Testing static alignment...")

    # Get Genesis link point clouds
    genesis_points = []
    for name in LINK_NAMES:
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        genesis_points.append(verts)
    genesis_all = np.vstack(genesis_points)

    # Get robot Gaussian positions (after splat_to_world transform)
    robot_xyz = robot_model.backup_xyz.cpu().numpy()

    print(f"   Genesis range: X=[{genesis_all[:,0].min():.4f}, {genesis_all[:,0].max():.4f}]")
    print(f"   Genesis range: Z=[{genesis_all[:,2].min():.4f}, {genesis_all[:,2].max():.4f}]")
    print(f"   Robot GS range: X=[{robot_xyz[:,0].min():.4f}, {robot_xyz[:,0].max():.4f}]")
    print(f"   Robot GS range: Z=[{robot_xyz[:,2].min():.4f}, {robot_xyz[:,2].max():.4f}]")

    # Compute alignment error
    pcd_genesis = o3d.geometry.PointCloud()
    pcd_genesis.points = o3d.utility.Vector3dVector(genesis_all)
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

    # Update robot model
    robot_model.update(arm)

    # Get transformed positions
    transformed_xyz = robot_model.gaussians.xyz.cpu().numpy()
    print(f"   After FK - X range: [{transformed_xyz[:,0].min():.4f}, {transformed_xyz[:,0].max():.4f}]")
    print(f"   After FK - Z range: [{transformed_xyz[:,2].min():.4f}, {transformed_xyz[:,2].max():.4f}]")

    # Save point clouds for visualization
    print("\n" + "=" * 60)
    print("Saving point clouds for visualization...")

    # Save Genesis (gray)
    pcd_genesis.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.io.write_point_cloud("data/test_genesis.ply", pcd_genesis)

    # Save robot GS at initial pose (green)
    pcd_robot.paint_uniform_color([0.0, 0.8, 0.0])
    o3d.io.write_point_cloud("data/test_robot_initial.ply", pcd_robot)

    # Save robot GS after FK (blue)
    pcd_fk = o3d.geometry.PointCloud()
    pcd_fk.points = o3d.utility.Vector3dVector(transformed_xyz)
    pcd_fk.paint_uniform_color([0.0, 0.0, 0.8])
    o3d.io.write_point_cloud("data/test_robot_fk.ply", pcd_fk)

    print("   Saved: data/test_genesis.ply (gray)")
    print("   Saved: data/test_robot_initial.ply (green)")
    print("   Saved: data/test_robot_fk.ply (blue)")
    print("\nTo visualize:")
    print("   python scripts/view_ply.py data/test_genesis.ply data/test_robot_initial.ply")
    print("=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    main()
