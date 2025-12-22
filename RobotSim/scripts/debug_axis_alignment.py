"""
Debug Axis Alignment
====================
Compare Genesis link coordinates with robot.ply to find axis mapping.
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
    print("Axis Alignment Debug")
    print("=" * 70)

    # Initialize Genesis
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(substeps=60),
        renderer=gs.renderers.Rasterizer(),
    )
    scene.add_entity(gs.morphs.Plane())

    # Load robot WITHOUT rotation
    print("\n[Test 1] Genesis robot WITHOUT euler rotation:")
    arm_no_rot = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="./assets/so100/urdf/so_arm100.xml",
            euler=(0.0, 0.0, 0.0),  # No rotation
            pos=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(),
    )
    scene.build()

    # Zero pose
    arm_no_rot.set_dofs_position([0, 0, 0, 0, 0, 0])
    scene.step()

    genesis_no_rot = []
    print("\n   Link coordinates (no rotation, zero pose):")
    for name in LINK_NAMES:
        link = arm_no_rot.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        genesis_no_rot.append(verts)
        print(f"   {name:20s}: X[{verts[:,0].min():.3f}, {verts[:,0].max():.3f}] "
              f"Y[{verts[:,1].min():.3f}, {verts[:,1].max():.3f}] "
              f"Z[{verts[:,2].min():.3f}, {verts[:,2].max():.3f}]")

    all_genesis_no_rot = np.vstack(genesis_no_rot)
    print(f"\n   Total range:")
    print(f"   X: [{all_genesis_no_rot[:,0].min():.4f}, {all_genesis_no_rot[:,0].max():.4f}]")
    print(f"   Y: [{all_genesis_no_rot[:,1].min():.4f}, {all_genesis_no_rot[:,1].max():.4f}]")
    print(f"   Z: [{all_genesis_no_rot[:,2].min():.4f}, {all_genesis_no_rot[:,2].max():.4f}]")

    # Load robot.ply
    print("\n" + "=" * 70)
    print("[robot.ply] Original coordinates:")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')
    xyz = robot_gau.xyz
    print(f"   X: [{xyz[:,0].min():.4f}, {xyz[:,0].max():.4f}]")
    print(f"   Y: [{xyz[:,1].min():.4f}, {xyz[:,1].max():.4f}]")
    print(f"   Z: [{xyz[:,2].min():.4f}, {xyz[:,2].max():.4f}]")
    print(f"   Center: [{xyz[:,0].mean():.4f}, {xyz[:,1].mean():.4f}, {xyz[:,2].mean():.4f}]")

    # Analyze axis correspondence
    print("\n" + "=" * 70)
    print("Axis Analysis:")
    print("=" * 70)

    g_ranges = [
        all_genesis_no_rot[:,0].max() - all_genesis_no_rot[:,0].min(),
        all_genesis_no_rot[:,1].max() - all_genesis_no_rot[:,1].min(),
        all_genesis_no_rot[:,2].max() - all_genesis_no_rot[:,2].min(),
    ]
    r_ranges = [
        xyz[:,0].max() - xyz[:,0].min(),
        xyz[:,1].max() - xyz[:,1].min(),
        xyz[:,2].max() - xyz[:,2].min(),
    ]

    print(f"\nGenesis ranges: X={g_ranges[0]:.3f}, Y={g_ranges[1]:.3f}, Z={g_ranges[2]:.3f}")
    print(f"robot.ply ranges: X={r_ranges[0]:.3f}, Y={r_ranges[1]:.3f}, Z={r_ranges[2]:.3f}")

    # Find which robot.ply axis corresponds to Genesis longest axis
    g_longest = np.argmax(g_ranges)
    r_longest = np.argmax(r_ranges)
    print(f"\nGenesis longest axis: {'XYZ'[g_longest]} ({g_ranges[g_longest]:.3f})")
    print(f"robot.ply longest axis: {'XYZ'[r_longest]} ({r_ranges[r_longest]:.3f})")

    # Suggest rotation
    print("\n" + "=" * 70)
    print("Potential axis mappings:")
    print("=" * 70)

    # Try different rotations and see which aligns best
    rotations = [
        ("No rotation", np.eye(3)),
        ("Rotate 90° around X", np.array([[1,0,0],[0,0,-1],[0,1,0]])),
        ("Rotate -90° around X", np.array([[1,0,0],[0,0,1],[0,-1,0]])),
        ("Rotate 90° around Y", np.array([[0,0,1],[0,1,0],[-1,0,0]])),
        ("Rotate -90° around Y", np.array([[0,0,-1],[0,1,0],[1,0,0]])),
        ("Rotate 90° around Z", np.array([[0,-1,0],[1,0,0],[0,0,1]])),
        ("Rotate -90° around Z", np.array([[0,1,0],[-1,0,0],[0,0,1]])),
        ("Swap X-Z", np.array([[0,0,1],[0,1,0],[1,0,0]])),
        ("Swap X-Y", np.array([[0,1,0],[1,0,0],[0,0,1]])),
        ("Swap Y-Z", np.array([[1,0,0],[0,0,1],[0,1,0]])),
    ]

    for name, R in rotations:
        xyz_rot = (R @ xyz.T).T
        ranges_rot = [
            xyz_rot[:,0].max() - xyz_rot[:,0].min(),
            xyz_rot[:,1].max() - xyz_rot[:,1].min(),
            xyz_rot[:,2].max() - xyz_rot[:,2].min(),
        ]
        # Calculate similarity to Genesis ranges
        diff = np.abs(np.array(ranges_rot) - np.array(g_ranges)).sum()
        print(f"\n{name}:")
        print(f"  Ranges: X={ranges_rot[0]:.3f}, Y={ranges_rot[1]:.3f}, Z={ranges_rot[2]:.3f}")
        print(f"  Range diff from Genesis: {diff:.3f}")


if __name__ == "__main__":
    main()
