"""
ICP with Scale Factor
======================

Apply scale to Genesis mesh before ICP registration.

Usage:
    python scripts/icp_with_scale.py
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
    print("ICP with Scale Factor")
    print("=" * 70)

    # Initialize Genesis
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

    # Get Genesis points
    print("\n[2/4] Getting Genesis points...")
    genesis_points = []
    for name in LINK_NAMES:
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        genesis_points.append(verts)
    genesis_points = np.vstack(genesis_points)
    print(f"   Genesis: {len(genesis_points)} vertices")

    # Scale Genesis by 0.8
    scale_factor = 0.8
    genesis_center = genesis_points.mean(axis=0)
    genesis_scaled = (genesis_points - genesis_center) * scale_factor + genesis_center
    print(f"   Applied scale factor: {scale_factor}")
    print(f"   Original range: [{genesis_points.min():.4f}, {genesis_points.max():.4f}]")
    print(f"   Scaled range: [{genesis_scaled.min():.4f}, {genesis_scaled.max():.4f}]")

    # Load robot.ply
    print("\n[3/4] Loading robot.ply...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')

    # Apply rotation (same as before)
    R_y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    robot_xyz = (R_y_90 @ robot_gau.xyz.T).T
    print(f"   Robot.ply: {len(robot_xyz)} Gaussians")
    print(f"   Robot range: [{robot_xyz.min():.4f}, {robot_xyz.max():.4f}]")

    # ICP registration
    print("\n[4/4] Running ICP registration...")

    pcd_genesis = o3d.geometry.PointCloud()
    pcd_genesis.points = o3d.utility.Vector3dVector(genesis_scaled)

    pcd_robot = o3d.geometry.PointCloud()
    pcd_robot.points = o3d.utility.Vector3dVector(robot_xyz)

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
    icp_translation = icp_transform[:3, 3]
    icp_rotation = icp_transform[:3, :3]

    print(f"\n   ICP Translation: [{icp_translation[0]:.4f}, {icp_translation[1]:.4f}, {icp_translation[2]:.4f}]")
    print(f"   ICP Rotation matrix:")
    print(f"   {icp_rotation}")

    # Print recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS for segment_robot.py")
    print("=" * 70)
    print(f"\n1. Scale factor: {scale_factor}")
    print(f"\n2. Translation (after rotation):")
    print(f"   translation = np.array([{icp_translation[0]:.4f}, {icp_translation[1]:.4f}, {icp_translation[2]:.4f}])")
    print(f"\n3. Genesis center (for scaling):")
    print(f"   genesis_center = np.array([{genesis_center[0]:.4f}, {genesis_center[1]:.4f}, {genesis_center[2]:.4f}])")

    # Save scaled Genesis for visualization
    pcd_genesis_colored = o3d.geometry.PointCloud()
    pcd_genesis_colored.points = o3d.utility.Vector3dVector(genesis_scaled)
    pcd_genesis_colored.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.io.write_point_cloud("data/genesis_scaled.ply", pcd_genesis_colored)

    # Apply ICP transform to robot and save
    robot_transformed = (icp_rotation @ robot_xyz.T).T + icp_translation
    pcd_robot_transformed = o3d.geometry.PointCloud()
    pcd_robot_transformed.points = o3d.utility.Vector3dVector(robot_transformed)
    pcd_robot_transformed.paint_uniform_color([0.0, 0.8, 0.0])
    o3d.io.write_point_cloud("data/robot_icp_aligned.ply", pcd_robot_transformed)

    print(f"\n   Saved: data/genesis_scaled.ply (gray)")
    print(f"   Saved: data/robot_icp_aligned.ply (green)")
    print(f"\nTo compare alignment:")
    print(f"   python scripts/view_ply.py data/genesis_scaled.ply data/robot_icp_aligned.ply")


if __name__ == "__main__":
    main()
