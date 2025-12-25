"""
ICP Object Alignment
====================

Align object PLY files to mesh coordinates.
Uses trimesh to load meshes directly, avoiding Genesis SDF computation issues.

Usage:
    python scripts/icp_object.py --object banana
    python scripts/icp_object.py --object toy
    python scripts/icp_object.py --object all
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import open3d as o3d
import trimesh
import json
from scipy.spatial.transform import Rotation as R
from Gaussians.util_gau import load_ply

# Object configuration: PLY path -> mesh path and transform info
OBJECT_CONFIGS = {
    'banana': {
        'ply_path': 'assets/so100/ply/banana.ply',
        'mesh_file': './assets/objects/banana/banana.obj',
        'pos': (0.32, 0.1, 0.04),
        'euler': (0.0, 0.0, 250.0),
        'scale': 0.95,  # Match Genesis scale
    },
    'toy': {
        'ply_path': 'assets/so100/ply/toy.ply',
        'mesh_file': './assets/objects/toy/toy.obj',
        'pos': (0.32, 0.1, 0.05),
        'euler': (0.0, 45.0, 0.0),
        'scale': 1.0,
    },
    'box': {
        'ply_path': 'assets/so100/ply/box.ply',
        'mesh_file': './assets/objects/box/box.obj',
        'pos': (0.2, -0.15, -0.003),  # Match Genesis pos
        'euler': (90.0, 0.0, 180.0),
        'scale': 0.95,  # Match Genesis scale
    },
    'water': {
        'ply_path': 'assets/so100/ply/water.ply',
        'mesh_file': './assets/objects/bottle_up/bottle1.obj',
        'pos': (0.3, 0.0, 0.005),
        'euler': (270.0, 0.0, 0.0),
        'scale': 1.0,
    },
    'redbox': {
        'ply_path': 'assets/so100/ply/redbox.ply',
        'mesh_file': './assets/objects/cube/red-box.glb',
        'pos': (0.25, -0.1, 0.04),
        'euler': (0.0, 0.0, 225.0),
        'scale': 1.0,
    },
    'bluebox': {
        'ply_path': 'assets/so100/ply/bluebox.ply',
        'mesh_file': './assets/objects/cube/blue-box.glb',
        'pos': (0.25, 0.1, 0.04),
        'euler': (0.0, 0.0, 225.0),
        'scale': 1.0,
    },
}


def load_mesh_vertices(mesh_file: str, pos: tuple, euler: tuple, scale: float) -> np.ndarray:
    """
    Load mesh file and transform vertices to world coordinates.

    Args:
        mesh_file: Path to mesh file (OBJ, PLY, etc.)
        pos: Position (x, y, z)
        euler: Euler angles in degrees (rx, ry, rz)
        scale: Scale factor

    Returns:
        Transformed vertices as numpy array (N, 3)
    """
    # Load mesh with trimesh
    mesh = trimesh.load(mesh_file, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, combine all meshes
        meshes = []
        for geom in mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError(f"No valid mesh found in {mesh_file}")

    vertices = mesh.vertices.copy()

    # Apply scale
    vertices = vertices * scale

    # Apply rotation (euler XYZ in degrees)
    rot = R.from_euler('xyz', euler, degrees=True)
    vertices = rot.apply(vertices)

    # Apply translation
    vertices = vertices + np.array(pos)

    return vertices


def align_object(object_name: str, visualize: bool = False):
    """Align a single object PLY to mesh coordinates."""

    if object_name not in OBJECT_CONFIGS:
        print(f"Unknown object: {object_name}")
        print(f"Available objects: {list(OBJECT_CONFIGS.keys())}")
        return None

    config = OBJECT_CONFIGS[object_name]
    print("=" * 70)
    print(f"ICP Alignment for: {object_name}")
    print("=" * 70)

    # Load mesh vertices directly with trimesh
    print(f"\n[1/4] Loading mesh: {config['mesh_file']}")
    try:
        mesh_points = load_mesh_vertices(
            config['mesh_file'],
            config['pos'],
            config['euler'],
            config['scale']
        )
        print(f"   Mesh vertices: {len(mesh_points)}")
        print(f"   Range: [{mesh_points.min():.4f}, {mesh_points.max():.4f}]")
        print(f"   Center: {mesh_points.mean(axis=0)}")
    except Exception as e:
        print(f"   Error loading mesh: {e}")
        print(f"   Trying alternative approach...")

        # Try loading as point cloud
        try:
            pcd = o3d.io.read_point_cloud(config['mesh_file'])
            if len(pcd.points) == 0:
                mesh = o3d.io.read_triangle_mesh(config['mesh_file'])
                mesh_points = np.asarray(mesh.vertices)
            else:
                mesh_points = np.asarray(pcd.points)

            # Apply transform
            rot = R.from_euler('xyz', config['euler'], degrees=True)
            mesh_points = rot.apply(mesh_points * config['scale']) + np.array(config['pos'])
            print(f"   Loaded with Open3D: {len(mesh_points)} points")
        except Exception as e2:
            print(f"   Failed to load mesh: {e2}")
            return None

    # Load object PLY
    print(f"\n[2/4] Loading PLY: {config['ply_path']}")
    obj_gau = load_ply(config['ply_path'])
    ply_xyz = obj_gau.xyz
    print(f"   PLY Gaussians: {len(ply_xyz)} points")
    print(f"   Range: [{ply_xyz.min():.4f}, {ply_xyz.max():.4f}]")
    print(f"   Center: {ply_xyz.mean(axis=0)}")

    # Run ICP registration
    print("\n[3/4] Running ICP registration...")

    pcd_mesh = o3d.geometry.PointCloud()
    pcd_mesh.points = o3d.utility.Vector3dVector(mesh_points)

    pcd_ply = o3d.geometry.PointCloud()
    pcd_ply.points = o3d.utility.Vector3dVector(ply_xyz)

    # Compute initial alignment using centroids
    mesh_center = mesh_points.mean(axis=0)
    ply_center = ply_xyz.mean(axis=0)
    initial_translation = mesh_center - ply_center

    init_transform = np.eye(4)
    init_transform[:3, 3] = initial_translation

    # Run ICP
    threshold = 0.05
    reg = o3d.pipelines.registration.registration_icp(
        pcd_ply, pcd_mesh, threshold,
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
    print("\n[4/4] Saving results...")
    output_dir = Path('exports/objects')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply transform to PLY points
    ply_transformed = (icp_rotation @ ply_xyz.T).T + icp_translation

    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(ply_transformed)
    pcd_aligned.paint_uniform_color([0.0, 0.8, 0.0])

    aligned_path = output_dir / f'{object_name}_aligned.ply'
    o3d.io.write_point_cloud(str(aligned_path), pcd_aligned)
    print(f"   Saved aligned PLY: {aligned_path}")

    # Save mesh for comparison
    pcd_mesh_colored = o3d.geometry.PointCloud()
    pcd_mesh_colored.points = o3d.utility.Vector3dVector(mesh_points)
    pcd_mesh_colored.paint_uniform_color([0.5, 0.5, 0.5])
    mesh_path = output_dir / f'{object_name}_mesh.ply'
    o3d.io.write_point_cloud(str(mesh_path), pcd_mesh_colored)
    print(f"   Saved mesh points: {mesh_path}")

    # Save alignment parameters
    params = {
        'object_name': object_name,
        'ply_path': config['ply_path'],
        'mesh_pos': list(config['pos']),
        'mesh_euler': list(config['euler']),
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
    print(f"   python scripts/view_ply.py {mesh_path} {aligned_path}")

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
