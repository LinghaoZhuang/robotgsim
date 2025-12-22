"""
Find Best Pose for Segmentation
================================

Search for the Genesis joint pose that best aligns with robot.ply.
Uses ICP (Iterative Closest Point) to find optimal alignment.

Usage:
    python scripts/find_best_pose.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from Gaussians.util_gau import load_ply
from robot_gaussian.forward_kinematics import LINK_NAMES


def get_genesis_points(arm, scene):
    """Get all Genesis link vertices as single point cloud."""
    all_points = []
    for name in LINK_NAMES:
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        all_points.append(verts)
    return np.vstack(all_points)


def compute_alignment_error(genesis_points, robot_points):
    """Compute alignment error using nearest neighbor distance."""
    # Subsample for speed
    if len(genesis_points) > 5000:
        idx = np.random.choice(len(genesis_points), 5000, replace=False)
        genesis_points = genesis_points[idx]
    if len(robot_points) > 5000:
        idx = np.random.choice(len(robot_points), 5000, replace=False)
        robot_points = robot_points[idx]

    # Use Open3D for fast nearest neighbor
    pcd_genesis = o3d.geometry.PointCloud()
    pcd_genesis.points = o3d.utility.Vector3dVector(genesis_points)

    pcd_robot = o3d.geometry.PointCloud()
    pcd_robot.points = o3d.utility.Vector3dVector(robot_points)

    # Compute point-to-point distance
    dists = pcd_robot.compute_point_cloud_distance(pcd_genesis)
    return np.mean(dists)


def transform_robot_ply(robot_xyz_original, rotation_deg, translation):
    """Apply rotation and translation to robot.ply."""
    # Rotation around Y axis
    angle = np.radians(rotation_deg)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])
    xyz = (R @ robot_xyz_original.T).T + translation
    return xyz


def main():
    print("=" * 70)
    print("Find Best Pose for Robot Segmentation")
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

    # Load robot.ply
    print("\n[2/4] Loading robot.ply...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')
    robot_xyz_original = robot_gau.xyz
    print(f"   Loaded {len(robot_xyz_original)} Gaussians")

    # Current best transform (from previous work)
    R_y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    robot_xyz_rotated = (R_y_90 @ robot_xyz_original.T).T
    current_translation = np.array([0.0376, -0.0048, 0.0788])

    # Test different joint poses
    print("\n[3/4] Searching for best joint pose...")
    print("-" * 70)

    # Poses to test
    test_poses = [
        ("Zero pose", [0, 0, 0, 0, 0, 0]),
        ("Current default", [0, -3.32, 3.11, 1.18, 0, -0.174]),
        ("Slightly bent 1", [0, -1.0, 1.0, 0.5, 0, 0]),
        ("Slightly bent 2", [0, -0.5, 0.5, 0.3, 0, 0]),
        ("Straight up", [0, -1.57, 1.57, 0, 0, 0]),
        ("Home pose 1", [0, -1.5, 1.5, 0.5, 0, 0]),
        ("Home pose 2", [0, -2.0, 2.0, 0.8, 0, 0]),
    ]

    best_error = float('inf')
    best_pose = None
    best_pose_name = None

    for pose_name, joints in test_poses:
        arm.set_dofs_position(joints)
        scene.step()

        genesis_points = get_genesis_points(arm, scene)

        # Try current rotation + translation
        robot_xyz = robot_xyz_rotated + current_translation
        error = compute_alignment_error(genesis_points, robot_xyz)

        print(f"   {pose_name:20s}: error = {error:.4f}")

        if error < best_error:
            best_error = error
            best_pose = joints
            best_pose_name = pose_name

    print("-" * 70)
    print(f"\n   Best pose: {best_pose_name}")
    print(f"   Joints: {best_pose}")
    print(f"   Error: {best_error:.4f}")

    # Fine-tune translation for best pose
    print("\n[4/4] Fine-tuning translation for best pose...")
    arm.set_dofs_position(best_pose)
    scene.step()
    genesis_points = get_genesis_points(arm, scene)

    # Try ICP alignment
    pcd_genesis = o3d.geometry.PointCloud()
    pcd_genesis.points = o3d.utility.Vector3dVector(genesis_points)

    pcd_robot = o3d.geometry.PointCloud()
    pcd_robot.points = o3d.utility.Vector3dVector(robot_xyz_rotated)

    # Run ICP
    print("   Running ICP registration...")
    threshold = 0.05
    reg = o3d.pipelines.registration.registration_icp(
        pcd_robot, pcd_genesis, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    print(f"   ICP fitness: {reg.fitness:.4f}")
    print(f"   ICP RMSE: {reg.inlier_rmse:.4f}")

    # Extract translation from ICP result
    icp_transform = reg.transformation
    icp_translation = icp_transform[:3, 3]
    icp_rotation = icp_transform[:3, :3]

    print(f"\n   ICP Translation: [{icp_translation[0]:.4f}, {icp_translation[1]:.4f}, {icp_translation[2]:.4f}]")

    # Print final recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(f"\n1. Best joint pose to use in segment_robot.py:")
    print(f"   INITIAL_JOINTS = {best_pose}")
    print(f"\n2. ICP-refined translation:")
    print(f"   translation = np.array([{icp_translation[0]:.4f}, {icp_translation[1]:.4f}, {icp_translation[2]:.4f}])")
    print(f"\n3. Combined with existing Y-axis 90° rotation")


if __name__ == "__main__":
    main()
