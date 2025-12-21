"""
Export robot.ply to JSON for web visualization
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
from Gaussians.util_gau import load_ply


def export_ply_to_json(ply_path, output_path):
    """Export PLY point cloud to JSON for Three.js."""
    print(f"Loading {ply_path}...")
    gau = load_ply(ply_path)

    # Sample points if too many (for web performance)
    n_points = len(gau.xyz)
    if n_points > 50000:
        print(f"Sampling {n_points} points to 50000...")
        indices = np.random.choice(n_points, 50000, replace=False)
        xyz = gau.xyz[indices]
    else:
        xyz = gau.xyz

    # Convert to list
    data = {
        'positions': xyz.tolist(),
        'count': len(xyz)
    }

    print(f"Exporting to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f)

    print(f"✓ Exported {len(xyz)} points to {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    export_ply_to_json(
        'exports/mult-view-scene/robot.ply',
        'scripts/robot_points.json'
    )
