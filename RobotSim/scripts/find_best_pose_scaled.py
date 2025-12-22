"""
Find Best Pose with Scale
==========================

Search for optimal alignment with Genesis scaled to 0.8.

Usage:
    python scripts/find_best_pose_scaled.py
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
    print("Find Best Alignment (Genesis scaled 0.8)")
    print("=" * 70)

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

    # Set pose
    INITIAL_JOINTS = [0, -3.32, 3.11, 1.18, 0, -0.174]
    arm.set_dofs_position(INITIAL_JOINTS)
    scene.step()
    print(f"   Joint pose: {INITIAL_JOINTS}")

    # 2. Get Genesis points and scale
    print("\n[2/4] Getting Genesis points (scaled 0.8)...")
    genesis_points = []
    for name in LINK_NAMES:
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        genesis_points.append(verts)
    genesis_all = np.vstack(genesis_points)

    # Apply scale 0.8
    SCALE = 0.8
    genesis_scaled = genesis_all * SCALE
    print(f"   Genesis points: {len(genesis_scaled)}")
    print(f"   Scaled range: [{genesis_scaled.min():.4f}, {genesis_scaled.max():.4f}]")
    print(f"   Scaled center: {genesis_scaled.mean(axis=0)}")

    # 3. Load robot.ply and rotate
    print("\n[3/4] Loading robot.ply and applying Y-90 rotation...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')

    # Rotate 90 degrees around Y axis
    R_y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    robot_rotated = (R_y_90 @ robot_gau.xyz.T).T

    print(f"   Robot points: {len(robot_rotated)}")
    print(f"   Rotated range: [{robot_rotated.min():.4f}, {robot_rotated.max():.4f}]")
    print(f"   Rotated center: {robot_rotated.mean(axis=0)}")

    # 4. ICP alignment
    print("\n[4/4] Running ICP to find translation...")

    pcd_genesis = o3d.geometry.PointCloud()
    pcd_genesis.points = o3d.utility.Vector3dVector(genesis_scaled)

    pcd_robot = o3d.geometry.PointCloud()
    pcd_robot.points = o3d.utility.Vector3dVector(robot_rotated)

    # Run ICP
    threshold = 0.05
    reg = o3d.pipelines.registration.registration_icp(
        pcd_robot, pcd_genesis, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    print(f"\n   ICP Results:")
    print(f"   Fitness: {reg.fitness:.4f}")
    print(f"   RMSE: {reg.inlier_rmse:.4f}")

    icp_transform = reg.transformation
    icp_rotation = icp_transform[:3, :3]
    icp_translation = icp_transform[:3, 3]

    print(f"\n   ICP Rotation matrix:")
    print(f"   {icp_rotation[0]}")
    print(f"   {icp_rotation[1]}")
    print(f"   {icp_rotation[2]}")
    print(f"\n   ICP Translation: [{icp_translation[0]:.4f}, {icp_translation[1]:.4f}, {icp_translation[2]:.4f}]")

    # Calculate center-based translation as backup
    center_offset = genesis_scaled.mean(axis=0) - robot_rotated.mean(axis=0)
    print(f"\n   Center-based offset: [{center_offset[0]:.4f}, {center_offset[1]:.4f}, {center_offset[2]:.4f}]")

    print("\n" + "=" * 70)
    print("UPDATE segment_robot_v2.py with these values:")
    print("=" * 70)
    print(f"""
GENESIS_SCALE = {SCALE}

# Only rotation, no fine transform needed
ROTATION_Y_90 = np.array([
    [0,  0,  1],
    [0,  1,  0],
    [-1, 0,  0]
])

# ICP-optimized translation
TRANSLATION = np.array([{icp_translation[0]:.4f}, {icp_translation[1]:.4f}, {icp_translation[2]:.4f}])
""")


if __name__ == "__main__":
    main()
