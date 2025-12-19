"""
Gaussian Transform Module
==========================

Applies FK transformations to robot Gaussians and transforms to COLMAP coordinates.

Key strategy:
1. robot.ply is initially in world coordinates
2. Apply FK transformations in world coordinate system
3. Transform to COLMAP coordinate system to align with background

Key functions:
- transform_means: Main function applying FK and coordinate transformation
"""

import torch
from e3nn import o3
from .sh_rotation import transform_shs


def transform_means(gau, segmented_list, transformations_list, world_to_splat):
    """
    Apply FK transformations to robot Gaussians, then transform to COLMAP coordinate.

    Args:
        gau: GaussianDataCUDA with attributes (xyz, rot, scale, opacity, sh)
        segmented_list: List of indices for each link (7 elements)
            [indices_base, indices_j0, ...] - Gaussian indices for each link
        transformations_list: List of (R_rel, T) tuples (6 elements, skipping Base)
            Each tuple contains relative transformation for one joint
        world_to_splat: (4,4) transformation matrix from world to COLMAP

    Returns:
        Modified gau with updated xyz, rot, and sh
    """
    xyz = gau.xyz.clone()  # Initially in world coordinates
    rot = gau.rot.clone()  # wxyz format
    sh = gau.sh.clone()    # (N, 16, 3)

    # ===== Step 1: Apply FK transformations in world coordinate system =====
    # Base (index 0) does not transform, start from joint 0
    for joint_idx, (R_rel, T) in enumerate(transformations_list):
        segment = segmented_list[joint_idx + 1]  # +1 to skip Base
        if len(segment) == 0:
            continue

        # 1.1 Transform position
        xyz[segment] = (R_rel @ xyz[segment].T).T + T

        # 1.2 Transform Gaussian rotation (covariance direction)
        # PLY rot is wxyz, e3nn also uses wxyz, can use directly
        rot_mat = o3.quaternion_to_matrix(rot[segment])  # wxyz → matrix
        rot_mat = R_rel @ rot_mat
        rot[segment] = o3.matrix_to_quaternion(rot_mat)  # matrix → wxyz

        # 1.3 Rotate SH coefficients (only non-DC part)
        shs_dc = sh[segment, :1, :]       # (M, 1, 3) - DC unchanged
        shs_rest = sh[segment, 1:, :]     # (M, 15, 3) - rest rotated
        shs_rest = transform_shs(shs_rest, R_rel)
        sh[segment] = torch.cat([shs_dc, shs_rest], dim=1)

    # ===== Step 2: Transform to COLMAP coordinate system (align with background) =====
    # Extract rotation and translation from world_to_splat
    R_w2s = world_to_splat[:3, :3]
    T_w2s = world_to_splat[:3, 3]

    # 2.1 Transform position
    xyz = (R_w2s @ xyz.T).T + T_w2s

    # 2.2 Transform Gaussian rotation
    rot_mat_all = o3.quaternion_to_matrix(rot)  # (N, 3, 3)
    rot_mat_all = R_w2s @ rot_mat_all
    rot = o3.matrix_to_quaternion(rot_mat_all)

    # 2.3 Rotate SH coefficients
    shs_dc_all = sh[:, :1, :]       # (N, 1, 3)
    shs_rest_all = sh[:, 1:, :]     # (N, 15, 3)
    shs_rest_all = transform_shs(shs_rest_all, R_w2s)
    sh = torch.cat([shs_dc_all, shs_rest_all], dim=1)

    # ===== Step 3: Update Gaussian data =====
    gau.xyz = xyz
    gau.rot = rot
    gau.sh = sh
    return gau
