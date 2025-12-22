"""
Filter Robot.ply Noise
=======================

Remove Gaussian points that are too far from Genesis mesh.
This filters out background and 3DGS noise.

Usage:
    python scripts/filter_robot_ply.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
import open3d as o3d
from Gaussians.util_gau import load_ply
from robot_gaussian.forward_kinematics import LINK_NAMES
from robot_gaussian.segmentation import train_segmentation_knn, segment_gaussians


def main():
    print("=" * 70)
    print("Filter Robot.ply - Remove Noise Points")
    print("=" * 70)

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

    INITIAL_JOINTS = [0, -3.32, 3.11, 1.18, 0, -0.174]
    arm.set_dofs_position(INITIAL_JOINTS)
    scene.step()

    # 2. Get Genesis points
    print("\n[2/5] Getting Genesis reference points...")
    genesis_points = []
    for name in LINK_NAMES:
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        genesis_points.append(verts)
    genesis_all = np.vstack(genesis_points)
    print(f"   Genesis points: {len(genesis_all)}")

    # 3. Load and transform robot.ply
    print("\n[3/5] Loading and transforming robot.ply...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')

    R_y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    robot_xyz = (R_y_90 @ robot_gau.xyz.T).T
    translation = np.array([-0.0375, -0.0238, 0.0886])
    robot_xyz = robot_xyz + translation

    print(f"   Original robot.ply points: {len(robot_xyz)}")

    # 4. Filter by distance to Genesis
    print("\n[4/5] Filtering noise points...")

    # Build KD-tree for Genesis points
    pcd_genesis = o3d.geometry.PointCloud()
    pcd_genesis.points = o3d.utility.Vector3dVector(genesis_all)
    kdtree = o3d.geometry.KDTreeFlann(pcd_genesis)

    # Find distance to nearest Genesis point for each robot.ply point
    distances = []
    for point in robot_xyz:
        [k, idx, dist] = kdtree.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(dist[0]))
    distances = np.array(distances)

    # Statistics
    print(f"\n   Distance statistics:")
    print(f"   Min: {distances.min():.4f} m")
    print(f"   Max: {distances.max():.4f} m")
    print(f"   Mean: {distances.mean():.4f} m")
    print(f"   Median: {np.median(distances):.4f} m")

    # Filter with different thresholds
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.1]
    print(f"\n   Points kept at different thresholds:")
    for thresh in thresholds:
        kept = np.sum(distances < thresh)
        pct = kept / len(distances) * 100
        print(f"   {thresh:.2f}m: {kept:6d} ({pct:5.1f}%)")

    # Use 0.03m threshold (3cm)
    THRESHOLD = 0.03
    mask = distances < THRESHOLD
    filtered_xyz = robot_xyz[mask]
    print(f"\n   Using threshold: {THRESHOLD}m")
    print(f"   Kept: {len(filtered_xyz)} / {len(robot_xyz)} ({len(filtered_xyz)/len(robot_xyz)*100:.1f}%)")

    # 5. Re-segment filtered points
    print("\n[5/5] Re-segmenting filtered points...")

    # Train KNN on Genesis
    knn = train_segmentation_knn(genesis_points, n_neighbors=10)

    # Segment filtered points
    labels = segment_gaussians(filtered_xyz, knn)

    # Save filtered labels
    np.save('data/labels/so100_labels_filtered.npy', labels)

    # Save mask for later use
    np.save('data/labels/so100_filter_mask.npy', mask)

    # Visualize
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
    pcd.points = o3d.utility.Vector3dVector(filtered_xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors_map[labels])

    output_path = "data/robot_segmented_filtered.ply"
    o3d.io.write_point_cloud(output_path, pcd)

    print("\n" + "=" * 70)
    print("Filtered Segmentation Results")
    print("=" * 70)
    print(f"\n   Saved to: {output_path}")
    print(f"\n   Gaussian distribution by link:")
    for i, name in enumerate(LINK_NAMES):
        count = np.sum(labels == i)
        pct = count / len(labels) * 100
        print(f"   {i}. {name:20s}: {count:6d} ({pct:5.1f}%)")

    print("\n" + "=" * 70)
    print("To visualize:")
    print(f"  python scripts/view_ply.py {output_path}")
    print("\nTo compare all three:")
    print("  python scripts/view_ply.py data/genesis_ground_truth.ply data/robot_segmented.ply data/robot_segmented_filtered.ply")
    print("=" * 70)


if __name__ == "__main__":
    main()
