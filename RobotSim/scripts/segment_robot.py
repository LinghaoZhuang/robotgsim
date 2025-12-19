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

    # 5. Load robot.ply (world coordinates)
    print("\n[5/6] Loading robot.ply...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')
    print(f"   Loaded {len(robot_gau.xyz)} Gaussians")
    print(f"   Coordinate range: [{robot_gau.xyz.min():.4f}, {robot_gau.xyz.max():.4f}]")

    # 6. Segment (no transformation needed)
    print("\n[6/6] Segmenting Gaussians...")
    labels = segment_gaussians(robot_gau.xyz, knn)

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
