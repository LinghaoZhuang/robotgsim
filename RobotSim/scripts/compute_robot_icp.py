"""
Get Genesis robot base position and compute correct ICP translation.
This script needs to run with Genesis.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import genesis as gs
from Gaussians.util_gau import load_ply
from robot_gaussian.robot_gaussian_model import (
    DEFAULT_R_Y_90, DEFAULT_ICP_ROTATION, DEFAULT_ICP_TRANSLATION, DEFAULT_GENESIS_SCALE
)

def main():
    print("=" * 70)
    print("Genesis Robot Position Analysis")
    print("=" * 70)

    # Initialize Genesis
    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False)

    # Add robot (same as pick_banana.py)
    arm = scene.add_entity(
        morph=gs.morphs.MJCF(file='./assets/so100/mjcf/so_arm100.xml'),
    )

    scene.build()

    # Set to reference pose
    initial_joints = [0, -3.32, 3.11, 1.18, 0, -0.174]
    arm.set_dofs_position(initial_joints)
    scene.step()

    # Get robot base position
    base_link = arm.get_link("Base")
    base_pos = base_link.get_pos().cpu().numpy()
    base_quat = base_link.get_quat().cpu().numpy()

    print(f"\nGenesis robot base position: {base_pos}")
    print(f"Genesis robot base quaternion: {base_quat}")

    # Now compute where robot.ply ends up after transforms
    robot = load_ply('exports/mult-view-scene/robot.ply')
    xyz = robot.xyz

    # Apply transforms
    R_combined = DEFAULT_ICP_ROTATION @ DEFAULT_R_Y_90
    xyz_aligned = (R_combined @ xyz.T).T + DEFAULT_ICP_TRANSLATION

    # Load genesis_center
    genesis_center = np.load('data/labels/genesis_center.npy')

    # Unscale
    C = genesis_center
    xyz_unscaled = (xyz_aligned - C) / DEFAULT_GENESIS_SCALE + C

    robot_center = xyz_unscaled.mean(axis=0)
    print(f"\nRobot PLY center after transforms: {robot_center}")

    # The robot base should be at the bottom of the robot
    # Find the lowest Z point as approximate base position
    robot_base_approx = np.array([
        xyz_unscaled[:, 0].mean(),  # X center
        xyz_unscaled[:, 1].mean(),  # Y center
        xyz_unscaled[:, 2].min()    # Z min (bottom)
    ])
    print(f"Robot PLY base (approx): {robot_base_approx}")

    # Compute required adjustment
    diff = base_pos - robot_base_approx
    print(f"\nDifference (Genesis - PLY): {diff}")

    # Convert to ICP translation adjustment (need to account for scale)
    # The adjustment in aligned space = diff * scale
    icp_adjustment = diff * DEFAULT_GENESIS_SCALE
    new_icp_translation = DEFAULT_ICP_TRANSLATION + icp_adjustment
    print(f"\nCurrent ICP translation: {DEFAULT_ICP_TRANSLATION}")
    print(f"Suggested new ICP translation: {new_icp_translation}")

    gs.destroy()

if __name__ == "__main__":
    main()
