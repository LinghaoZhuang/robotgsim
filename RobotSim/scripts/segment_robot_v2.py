"""
Robot Segmentation Script (SplatSim Style)
==========================================

KNN segmentation with AABB filtering, similar to SplatSim.

Usage:
    python scripts/segment_robot_v2.py

Coordinate Transform:
    Manually align robot.ply to Genesis mesh in CloudCompare/URDF visualizer,
    then update TRANSFORMATION_MATRIX below.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from Gaussians.util_gau import load_ply

# =============================================================================
# CONFIGURATION - Update these after manual alignment
# =============================================================================

# Robot initial joint pose (should match robot.ply capture pose)
INITIAL_JOINTS = [0, -3.32, 3.11, 1.18, 0, -0.174]

# Scale factor for Genesis points (from CloudCompare alignment)
# Genesis was scaled to 0.8 to match robot.ply
GENESIS_SCALE = 0.8

# 4x4 Transformation matrix from CloudCompare registration
# This transforms robot.ply to align with scaled Genesis
TRANSFORMATION_MATRIX = np.array([
    [0.995,  -0.095,  0.019,  -0.009],
    [0.095,   0.996, -0.005,  -0.008],
    [-0.018,  0.007,  1.000,   0.024],
    [0.000,   0.000,  0.000,   1.000]
])

# AABB adjustment (expand bounding box slightly)
AABB_PADDING = 0.02  # 2cm padding

# Link names for SO-100 robot
LINK_NAMES = [
    "Base",
    "Rotation_Pitch",
    "Upper_Arm",
    "Lower_Arm",
    "Wrist_Pitch_Roll",
    "Fixed_Jaw",
    "Moving_Jaw"
]

# =============================================================================


def get_link_point_clouds(arm):
    """Get mesh vertices for each link from Genesis."""
    point_clouds = []
    for name in LINK_NAMES:
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        point_clouds.append(verts)
    return point_clouds


def transform_points(points, matrix):
    """Apply 4x4 transformation matrix to points."""
    ones = np.ones((len(points), 1))
    points_homo = np.hstack([points, ones])
    transformed = (matrix @ points_homo.T).T
    return transformed[:, :3]


def compute_aabb(points, padding=0.0):
    """Compute axis-aligned bounding box with optional padding."""
    min_pt = points.min(axis=0) - padding
    max_pt = points.max(axis=0) + padding
    return min_pt, max_pt


def filter_by_aabb(points, aabb_min, aabb_max):
    """Filter points to keep only those inside AABB."""
    condition = (
        (points[:, 0] > aabb_min[0]) & (points[:, 0] < aabb_max[0]) &
        (points[:, 1] > aabb_min[1]) & (points[:, 1] < aabb_max[1]) &
        (points[:, 2] > aabb_min[2]) & (points[:, 2] < aabb_max[2])
    )
    return condition


def main():
    print("=" * 70)
    print("Robot Segmentation (SplatSim Style)")
    print("=" * 70)

    # 1. Initialize Genesis
    print("\n[1/6] Initializing Genesis...")
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

    # 2. Set initial pose
    print("\n[2/6] Setting initial pose...")
    arm.set_dofs_position(INITIAL_JOINTS)
    scene.step()
    print(f"   Joints: {INITIAL_JOINTS}")

    # 3. Get Genesis ground truth
    print("\n[3/6] Extracting Genesis ground truth...")
    point_clouds = get_link_point_clouds(arm)

    # Apply scale to Genesis points (to match CloudCompare alignment)
    point_clouds = [pc * GENESIS_SCALE for pc in point_clouds]
    print(f"   Applied scale factor: {GENESIS_SCALE}")

    all_genesis_points = np.vstack(point_clouds)
    print(f"   Total Genesis points: {len(all_genesis_points)}")

    for i, (name, pc) in enumerate(zip(LINK_NAMES, point_clouds)):
        print(f"   {i}. {name:20s}: {len(pc):6d} vertices")

    # Compute AABB from Genesis points
    aabb_min, aabb_max = compute_aabb(all_genesis_points, AABB_PADDING)
    print(f"\n   AABB (with {AABB_PADDING*100:.0f}cm padding):")
    print(f"   Min: [{aabb_min[0]:.4f}, {aabb_min[1]:.4f}, {aabb_min[2]:.4f}]")
    print(f"   Max: [{aabb_max[0]:.4f}, {aabb_max[1]:.4f}, {aabb_max[2]:.4f}]")

    # 4. Train KNN
    print("\n[4/6] Training KNN classifier...")
    X = np.vstack(point_clouds)
    y = np.hstack([np.full(len(pc), i) for i, pc in enumerate(point_clouds)])
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, y)
    print("   KNN trained")

    # 5. Load and transform robot.ply
    print("\n[5/6] Loading and transforming robot.ply...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')
    robot_xyz_original = robot_gau.xyz
    print(f"   Original points: {len(robot_xyz_original)}")

    # Apply transformation
    robot_xyz = transform_points(robot_xyz_original, TRANSFORMATION_MATRIX)
    print(f"   Transformed range: [{robot_xyz.min():.4f}, {robot_xyz.max():.4f}]")

    # AABB filtering (SplatSim style)
    print("\n   Applying AABB filter...")
    mask = filter_by_aabb(robot_xyz, aabb_min, aabb_max)
    robot_xyz_filtered = robot_xyz[mask]
    print(f"   After AABB filter: {len(robot_xyz_filtered)} / {len(robot_xyz)} "
          f"({len(robot_xyz_filtered)/len(robot_xyz)*100:.1f}%)")

    # 6. Segment
    print("\n[6/6] Segmenting Gaussians...")
    labels = knn.predict(robot_xyz_filtered)

    # Save results
    # Labels for filtered points
    np.save('data/labels/so100_labels_filtered.npy', labels)
    # Mask to know which original points were kept
    np.save('data/labels/so100_aabb_mask.npy', mask)

    # Also create full labels array (for points outside AABB, assign -1 or nearest)
    full_labels = np.full(len(robot_xyz), -1, dtype=np.int32)
    full_labels[mask] = labels
    np.save('data/labels/so100_labels.npy', full_labels)

    print(f"\n   Saved to: data/labels/so100_labels.npy")
    print(f"   Saved filtered to: data/labels/so100_labels_filtered.npy")
    print(f"   Saved AABB mask to: data/labels/so100_aabb_mask.npy")

    # Statistics
    print("\n" + "=" * 70)
    print("Segmentation Results")
    print("=" * 70)
    print(f"\n   Gaussian distribution by link:")
    for i, name in enumerate(LINK_NAMES):
        count = np.sum(labels == i)
        pct = count / len(labels) * 100
        print(f"   {i}. {name:20s}: {count:6d} ({pct:5.1f}%)")

    # Save colored PLY for visualization
    import open3d as o3d
    colors_map = np.array([
        [1.0, 0.0, 0.0],    # Base - Red
        [0.0, 1.0, 0.0],    # Rotation_Pitch - Green
        [0.0, 0.0, 1.0],    # Upper_Arm - Blue
        [1.0, 1.0, 0.0],    # Lower_Arm - Yellow
        [1.0, 0.0, 1.0],    # Wrist_Pitch_Roll - Magenta
        [0.0, 1.0, 1.0],    # Fixed_Jaw - Cyan
        [1.0, 0.5, 0.0],    # Moving_Jaw - Orange
    ])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(robot_xyz_filtered)
    pcd.colors = o3d.utility.Vector3dVector(colors_map[labels])
    o3d.io.write_point_cloud("data/robot_segmented_v2.ply", pcd)

    print(f"\n   Saved visualization to: data/robot_segmented_v2.ply")
    print("\n" + "=" * 70)
    print("To visualize:")
    print("  python scripts/view_ply.py data/robot_segmented_v2.ply")
    print("\nTo compare with Genesis ground truth:")
    print("  python scripts/view_ply.py data/genesis_ground_truth.ply data/robot_segmented_v2.ply")
    print("=" * 70)


if __name__ == "__main__":
    main()
