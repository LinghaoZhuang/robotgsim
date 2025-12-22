"""
Debug Translation Alignment
============================
Compare Genesis link centers with robot.ply to find translation offset.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
from Gaussians.util_gau import load_ply
from robot_gaussian.forward_kinematics import LINK_NAMES


def main():
    print("=" * 70)
    print("Translation Alignment Debug")
    print("=" * 70)

    # Initialize Genesis with euler=(0,0,90)
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
            euler=(0.0, 0.0, 90.0),  # Same as segment_robot.py
            pos=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(),
    )
    scene.build()

    # Zero pose
    arm.set_dofs_position([0, 0, 0, 0, 0, 0])
    scene.step()

    # Get Genesis link coordinates
    print("\n[Genesis] Link coordinate ranges (with euler=(0,0,90), zero pose):")
    print("-" * 70)
    genesis_all = []
    for name in LINK_NAMES:
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        genesis_all.append(verts)
        x_min, x_max = verts[:,0].min(), verts[:,0].max()
        y_min, y_max = verts[:,1].min(), verts[:,1].max()
        z_min, z_max = verts[:,2].min(), verts[:,2].max()
        print(f"{name:20s}: X[{x_min:6.3f}, {x_max:6.3f}] "
              f"Y[{y_min:6.3f}, {y_max:6.3f}] "
              f"Z[{z_min:6.3f}, {z_max:6.3f}]")

    genesis_all = np.vstack(genesis_all)
    genesis_center = genesis_all.mean(axis=0)
    print(f"\n{'Total':20s}: X[{genesis_all[:,0].min():6.3f}, {genesis_all[:,0].max():6.3f}] "
          f"Y[{genesis_all[:,1].min():6.3f}, {genesis_all[:,1].max():6.3f}] "
          f"Z[{genesis_all[:,2].min():6.3f}, {genesis_all[:,2].max():6.3f}]")
    print(f"{'Center':20s}: [{genesis_center[0]:6.3f}, {genesis_center[1]:6.3f}, {genesis_center[2]:6.3f}]")

    # Load robot.ply and rotate
    print("\n" + "=" * 70)
    print("[robot.ply] After rotation (90° around Y axis):")
    print("-" * 70)
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')

    R_y_90 = np.array([
        [0,  0,  1],
        [0,  1,  0],
        [-1, 0,  0]
    ])
    robot_xyz = (R_y_90 @ robot_gau.xyz.T).T
    robot_center = robot_xyz.mean(axis=0)

    print(f"{'Total':20s}: X[{robot_xyz[:,0].min():6.3f}, {robot_xyz[:,0].max():6.3f}] "
          f"Y[{robot_xyz[:,1].min():6.3f}, {robot_xyz[:,1].max():6.3f}] "
          f"Z[{robot_xyz[:,2].min():6.3f}, {robot_xyz[:,2].max():6.3f}]")
    print(f"{'Center':20s}: [{robot_center[0]:6.3f}, {robot_center[1]:6.3f}, {robot_center[2]:6.3f}]")

    # Calculate translation offset
    print("\n" + "=" * 70)
    print("Translation Analysis:")
    print("=" * 70)

    offset = genesis_center - robot_center
    print(f"\nCenter offset (Genesis - robot.ply):")
    print(f"  X: {offset[0]:+.4f}")
    print(f"  Y: {offset[1]:+.4f}")
    print(f"  Z: {offset[2]:+.4f}")

    # Apply offset and show new range
    robot_xyz_aligned = robot_xyz + offset
    print(f"\n[robot.ply] After rotation + translation:")
    print(f"{'Total':20s}: X[{robot_xyz_aligned[:,0].min():6.3f}, {robot_xyz_aligned[:,0].max():6.3f}] "
          f"Y[{robot_xyz_aligned[:,1].min():6.3f}, {robot_xyz_aligned[:,1].max():6.3f}] "
          f"Z[{robot_xyz_aligned[:,2].min():6.3f}, {robot_xyz_aligned[:,2].max():6.3f}]")
    print(f"{'Center':20s}: [{robot_xyz_aligned.mean(axis=0)[0]:6.3f}, "
          f"{robot_xyz_aligned.mean(axis=0)[1]:6.3f}, "
          f"{robot_xyz_aligned.mean(axis=0)[2]:6.3f}]")

    print("\n" + "=" * 70)
    print("Suggested translation to add in segment_robot.py:")
    print("=" * 70)
    print(f"\n  translation = np.array([{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}])")
    print(f"  robot_xyz = robot_xyz + translation")


if __name__ == "__main__":
    main()
