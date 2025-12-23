"""
Gaussian Transform Module
==========================

Applies FK transformations to robot Gaussians and transforms to COLMAP coordinates.

Key strategy:
1. robot.ply is in scaled world coordinates (aligned to Genesis*0.8)
2. Apply FK transformations in scaled coordinate system
3. Convert from scaled to unscaled world coordinates
4. Transform to COLMAP coordinate system to align with background

Key functions:
- transform_means: Main function applying FK and coordinate transformation (unscaled)
- transform_means_scaled: FK in scaled space, then convert to COLMAP
"""

import torch
import numpy as np
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
    # Note: world_to_splat may contain scale, so we need to handle it carefully
    R_w2s_with_scale = world_to_splat[:3, :3]
    T_w2s = world_to_splat[:3, 3]

    # Extract pure rotation matrix (normalize to remove scale)
    # SVD decomposition: R_scaled = U @ S @ V^T, pure rotation = U @ V^T
    U, S, Vh = torch.linalg.svd(R_w2s_with_scale)
    R_w2s_pure = U @ Vh  # Pure rotation matrix (det = 1)

    # Extract scale factor (geometric mean of singular values)
    scale_factor = S.prod().pow(1/3)

    # 2.1 Transform position (with scale)
    xyz = (R_w2s_with_scale @ xyz.T).T + T_w2s

    # 2.2 Transform Gaussian rotation (pure rotation only, no scale)
    rot_mat_all = o3.quaternion_to_matrix(rot)  # (N, 3, 3)
    rot_mat_all = R_w2s_pure @ rot_mat_all
    rot = o3.matrix_to_quaternion(rot_mat_all)

    # 2.3 Scale Gaussian scale parameters
    scale = gau.scale * scale_factor

    # 2.4 Rotate SH coefficients (pure rotation only)
    shs_dc_all = sh[:, :1, :]       # (N, 1, 3)
    shs_rest_all = sh[:, 1:, :]     # (N, 15, 3)
    shs_rest_all = transform_shs(shs_rest_all, R_w2s_pure)
    sh = torch.cat([shs_dc_all, shs_rest_all], dim=1)

    # ===== Step 3: Update Gaussian data =====
    gau.xyz = xyz
    gau.rot = rot
    gau.scale = scale
    gau.sh = sh
    return gau


def transform_means_scaled(gau, segmented_list, transformations_list, world_to_splat,
                           genesis_center, genesis_scale):
    """
    Apply FK transformations in scaled coordinate system, then transform to COLMAP.

    This function is used when robot.ply is aligned to Genesis*scale (e.g., 0.8).
    The FK transforms should also be computed in scaled space using
    get_transformation_list_scaled().

    Args:
        gau: GaussianDataCUDA with attributes (xyz, rot, scale, opacity, sh)
        segmented_list: List of indices for each link (7 elements)
        transformations_list: List of (R_rel, T_scaled) from get_transformation_list_scaled
        world_to_splat: (4,4) transformation matrix from unscaled world to COLMAP
        genesis_center: Center point used for scaling (numpy array, shape (3,))
        genesis_scale: Scale factor (e.g., 0.8)

    Returns:
        Modified gau with updated xyz, rot, scale, and sh
    """
    xyz = gau.xyz.clone()  # In scaled world coordinates
    rot = gau.rot.clone()  # wxyz format
    sh = gau.sh.clone()    # (N, 16, 3)

    # Convert genesis_center to tensor
    C = torch.tensor(genesis_center, device='cuda', dtype=torch.float32)

    # ===== Step 1: Apply FK transformations in scaled coordinate system =====
    # Base (index 0) does not transform, start from joint 0
    for joint_idx, (R_rel, T_scaled) in enumerate(transformations_list):
        segment = segmented_list[joint_idx + 1]  # +1 to skip Base
        if len(segment) == 0:
            continue

        # 1.1 Transform position (in scaled space)
        xyz[segment] = (R_rel @ xyz[segment].T).T + T_scaled

        # 1.2 Transform Gaussian rotation
        rot_mat = o3.quaternion_to_matrix(rot[segment])
        rot_mat = R_rel @ rot_mat
        rot[segment] = o3.matrix_to_quaternion(rot_mat)

        # 1.3 Rotate SH coefficients
        shs_dc = sh[segment, :1, :]
        shs_rest = sh[segment, 1:, :]
        shs_rest = transform_shs(shs_rest, R_rel)
        sh[segment] = torch.cat([shs_dc, shs_rest], dim=1)

    # ===== Step 2: Convert from scaled to unscaled world coordinates =====
    # robot_scaled = (robot_world - C) * scale + C
    # Inverse: robot_world = (robot_scaled - C) / scale + C
    xyz_unscaled = (xyz - C) / genesis_scale + C

    # Scale Gaussian scale parameters (they were in scaled space)
    scale_unscaled = gau.scale / genesis_scale

    # ===== Step 3: Transform to COLMAP coordinate system =====
    R_w2s_with_scale = world_to_splat[:3, :3]
    T_w2s = world_to_splat[:3, 3]

    # Extract pure rotation and scale factor
    U, S, Vh = torch.linalg.svd(R_w2s_with_scale)
    R_w2s_pure = U @ Vh
    scale_factor = S.prod().pow(1/3)

    # 3.1 Transform position (with scale from world_to_splat)
    xyz_colmap = (R_w2s_with_scale @ xyz_unscaled.T).T + T_w2s

    # 3.2 Transform Gaussian rotation
    rot_mat_all = o3.quaternion_to_matrix(rot)
    rot_mat_all = R_w2s_pure @ rot_mat_all
    rot_colmap = o3.matrix_to_quaternion(rot_mat_all)

    # 3.3 Scale Gaussian scale parameters (combine both scales)
    scale_colmap = scale_unscaled * scale_factor

    # 3.4 Rotate SH coefficients
    shs_dc_all = sh[:, :1, :]
    shs_rest_all = sh[:, 1:, :]
    shs_rest_all = transform_shs(shs_rest_all, R_w2s_pure)
    sh_colmap = torch.cat([shs_dc_all, shs_rest_all], dim=1)

    # ===== Step 4: Update Gaussian data =====
    gau.xyz = xyz_colmap
    gau.rot = rot_colmap
    gau.scale = scale_colmap
    gau.sh = sh_colmap
    return gau
