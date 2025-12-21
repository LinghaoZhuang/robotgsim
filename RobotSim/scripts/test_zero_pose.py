"""
Quick Test: Try Zero Pose for Segmentation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
from robot_gaussian.segmentation import *
from Gaussians.util_gau import load_ply

# Initialize Genesis
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

# Try ZERO POSE
print("Testing ZERO POSE: [0, 0, 0, 0, 0, 0]")
INITIAL_JOINTS = [0, 0, 0, 0, 0, 0]
arm.set_dofs_position(INITIAL_JOINTS)
scene.step()

# Get link point clouds
point_clouds = get_link_point_clouds_genesis(arm, LINK_NAMES)
print(f"\nLink point cloud sizes:")
for i, (name, pc) in enumerate(zip(LINK_NAMES, point_clouds)):
    print(f"  {i}. {name:20s}: {len(pc)} vertices")

# Train KNN
knn = train_segmentation_knn(point_clouds, n_neighbors=10)

# Load and segment robot.ply
robot_gau = load_ply('exports/mult-view-scene/robot.ply')
labels = segment_gaussians(robot_gau.xyz, knn)
np.save('data/labels/so100_labels_zero_pose.npy', labels)

print(f"\nSaved to: data/labels/so100_labels_zero_pose.npy")
print(f"Unique labels: {np.unique(labels)}")
print("\nRun visualize_segmentation.py to check alignment!")
