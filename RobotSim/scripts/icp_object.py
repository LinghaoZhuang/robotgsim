"""
ICP Object Alignment
====================

Align object PLY files to Genesis mesh coordinates.

Usage:
    python scripts/icp_object.py --object banana
    python scripts/icp_object.py --object toy
    python scripts/icp_object.py --object all
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import genesis as gs
import numpy as np
import open3d as o3d
import torch
import json
from Gaussians.util_gau import load_ply

# Object configuration: PLY path -> mesh path and Genesis loading info
OBJECT_CONFIGS = {
    'banana': {
        'ply_path': 'assets/so100/ply/banana.ply',
        'mesh_file': './assets/objects/banana/banana.obj',
        'pos': (0.32, 0.1, 0.04),
        'euler': (0.0, 0.0, 250.0),
        'scale': 1.0,
        'mesh_type': 'obj',
    },
    'toy': {
        'ply_path': 'assets/so100/ply/toy.ply',
        'mesh_file': './assets/objects/toy/toy.obj',
        'pos': (0.32, 0.1, 0.05),
        'euler': (0.0, 45.0, 0.0),
        'scale': 1.0,
        'mesh_type': 'obj',
    },
    'box': {
        'ply_path': 'assets/so100/ply/box.ply',
        'mesh_file': './assets/objects/box/box.obj',
        'pos': (0.2, -0.15, -0.004),
        'euler': (90.0, 0.0, 180.0),
        'scale': 1.0,
        'mesh_type': 'obj',
    },
    'water': {
        'ply_path': 'assets/so100/ply/water.ply',
        'mesh_file': './assets/objects/bottle_up/bottle1.obj',
        'pos': (0.3, 0.0, 0.005),
        'euler': (270.0, 0.0, 0.0),
        'scale': 1.0,
        'mesh_type': 'obj',
    },
    'redbox': {
        'ply_path': 'assets/so100/ply/redbox.ply',
        'mesh_file': './assets/objects/cube/cube_red_texture.urdf',
        'pos': (0.25, -0.1, 0.04),
        'euler': (0.0, 0.0, 225.0),
        'scale': 1.0,
        'mesh_type': 'urdf',
    },
    'bluebox': {
        'ply_path': 'assets/so100/ply/bluebox.ply',
        'mesh_file': './assets/objects/cube/cube_blue_texture.urdf',
        'pos': (0.25, 0.1, 0.04),
        'euler': (0.0, 0.0, 225.0),
        'scale': 1.0,
        'mesh_type': 'urdf',
    },
}


def align_object(object_name: str, visualize: bool = False):
    """Align a single object PLY to Genesis mesh."""

    if object_name not in OBJECT_CONFIGS:
        print(f"Unknown object: {object_name}")
        print(f"Available objects: {list(OBJECT_CONFIGS.keys())}")
        return None

    config = OBJECT_CONFIGS[object_name]
    print("=" * 70)
    print(f"ICP Alignment for: {object_name}")
    print("=" * 70)

    # Initialize Genesis
    print("\n[1/5] Initializing Genesis...")
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(substeps=60),
        renderer=gs.renderers.Rasterizer(),
    )
    scene.add_entity(gs.morphs.Plane())

    # Add object based on mesh type
    print(f"\n[2/5] Loading object mesh: {config['mesh_file']}")
    if config['mesh_type'] == 'obj':
        obj_entity = scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Mesh(
                file=config['mesh_file'],
                pos=config['pos'],
                euler=config['euler'],
                scale=config['scale'],
                collision=True,
                visualization=True,
                convexify=False,
                decompose_nonconvex=True,
            ),
            surface=gs.surfaces.Default(vis_mode='visual'),
        )
    else:  # urdf
        obj_entity = scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.URDF(
                file=config['mesh_file'],
                pos=config['pos'],
                euler=config['euler'],
                collision=True,
                visualization=True,
                convexify=True,
            ),
        )

    scene.build()
    scene.step()

    # Get Genesis mesh vertices
    print("\n[3/5] Getting Genesis mesh vertices...")
    try:
        genesis_points = obj_entity.get_vverts().cpu().numpy()
    except:
        # For some entities, try getting from links
        genesis_points = []
        for i in range(obj_entity.n_links):
            link = obj_entity.get_link(i)
            verts = link.get_vverts().cpu().numpy()
            genesis_points.append(verts)
        genesis_points = np.vstack(genesis_points)

    print(f"   Genesis mesh: {len(genesis_points)} vertices")
    print(f"   Range: [{genesis_points.min():.4f}, {genesis_points.max():.4f}]")
    print(f"   Center: [{genesis_points.mean(axis=0)}]")

    # Load object PLY
    print(f"\n[4/5] Loading PLY: {config['ply_path']}")
    obj_gau = load_ply(config['ply_path'])
    ply_xyz = obj_gau.xyz
    print(f"   PLY Gaussians: {len(ply_xyz)} points")
    print(f"   Range: [{ply_xyz.min():.4f}, {ply_xyz.max():.4f}]")
    print(f"   Center: [{ply_xyz.mean(axis=0)}]")

    # Run ICP registration
    print("\n[5/5] Running ICP registration...")

    pcd_genesis = o3d.geometry.PointCloud()
    pcd_genesis.points = o3d.utility.Vector3dVector(genesis_points)

    pcd_ply = o3d.geometry.PointCloud()
    pcd_ply.points = o3d.utility.Vector3dVector(ply_xyz)

    # Compute initial alignment using centroids
    genesis_center = genesis_points.mean(axis=0)
    ply_center = ply_xyz.mean(axis=0)
    initial_translation = genesis_center - ply_center

    init_transform = np.eye(4)
    init_transform[:3, 3] = initial_translation

    # Run ICP
    threshold = 0.05
    reg = o3d.pipelines.registration.registration_icp(
        pcd_ply, pcd_genesis, threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
    )

    print(f"\n   ICP Results:")
    print(f"   Fitness: {reg.fitness:.4f}")
    print(f"   RMSE: {reg.inlier_rmse:.4f}")

    icp_transform = reg.transformation
    icp_rotation = icp_transform[:3, :3]
    icp_translation = icp_transform[:3, 3]

    print(f"\n   ICP Rotation matrix:")
    for row in icp_rotation:
        print(f"   [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    print(f"\n   ICP Translation: [{icp_translation[0]:.6f}, {icp_translation[1]:.6f}, {icp_translation[2]:.6f}]")

    # Save aligned PLY for visualization
    output_dir = Path('exports/objects')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply transform to PLY points
    ply_transformed = (icp_rotation @ ply_xyz.T).T + icp_translation

    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(ply_transformed)
    pcd_aligned.paint_uniform_color([0.0, 0.8, 0.0])

    aligned_path = output_dir / f'{object_name}_aligned.ply'
    o3d.io.write_point_cloud(str(aligned_path), pcd_aligned)
    print(f"\n   Saved aligned PLY: {aligned_path}")

    # Save Genesis mesh for comparison
    pcd_genesis_colored = o3d.geometry.PointCloud()
    pcd_genesis_colored.points = o3d.utility.Vector3dVector(genesis_points)
    pcd_genesis_colored.paint_uniform_color([0.5, 0.5, 0.5])
    genesis_path = output_dir / f'{object_name}_genesis.ply'
    o3d.io.write_point_cloud(str(genesis_path), pcd_genesis_colored)
    print(f"   Saved Genesis mesh: {genesis_path}")

    # Save alignment parameters
    params = {
        'object_name': object_name,
        'ply_path': config['ply_path'],
        'genesis_pos': list(config['pos']),
        'genesis_euler': list(config['euler']),
        'icp_rotation': icp_rotation.tolist(),
        'icp_translation': icp_translation.tolist(),
        'fitness': reg.fitness,
        'rmse': reg.inlier_rmse,
    }

    params_path = output_dir / f'{object_name}_icp_params.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"   Saved ICP parameters: {params_path}")

    print(f"\nTo visualize alignment:")
    print(f"   python scripts/view_ply.py {genesis_path} {aligned_path}")

    return params


def main():
    parser = argparse.ArgumentParser(description='ICP Object Alignment')
    parser.add_argument('--object', type=str, required=True,
                        help=f'Object name or "all". Options: {list(OBJECT_CONFIGS.keys())}')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize alignment (requires display)')
    args = parser.parse_args()

    if args.object == 'all':
        all_params = {}
        for obj_name in OBJECT_CONFIGS.keys():
            print(f"\n{'#' * 80}")
            print(f"# Processing: {obj_name}")
            print(f"{'#' * 80}")
            params = align_object(obj_name, args.visualize)
            if params:
                all_params[obj_name] = params
            # Need to restart Genesis for each object
            gs.destroy()

        # Save all parameters
        output_dir = Path('exports/objects')
        all_params_path = output_dir / 'all_icp_params.json'
        with open(all_params_path, 'w') as f:
            json.dump(all_params, f, indent=2)
        print(f"\nSaved all ICP parameters: {all_params_path}")
    else:
        align_object(args.object, args.visualize)


if __name__ == "__main__":
    main()
