"""
Spherical Harmonics Rotation Module
====================================

Implements spherical harmonics rotation using Wigner-D matrices.
Only rotates non-DC SH coefficients (l=1,2,3), DC coefficient remains unchanged.

Key functions:
- transform_shs: Rotate SH coefficients using Wigner-D matrices
"""

import torch
from e3nn import o3
import einops


def transform_shs(shs_rest, rotation_matrix):
    """
    Rotate SH coefficients using Wigner-D matrices.
    Only processes l=1,2,3 bands (15 coefficients), not DC (l=0).

    Args:
        shs_rest: (N, 15, 3) - Non-DC SH coefficients
        rotation_matrix: (3, 3) rotation matrix

    Returns:
        Rotated SH coefficients (N, 15, 3)
    """
    # Axis permutation (align with SplatSim coordinate convention)
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                     device=rotation_matrix.device, dtype=rotation_matrix.dtype)
    permuted_rot = torch.linalg.inv(P) @ rotation_matrix @ P

    # Extract Euler angles
    rot_angles = o3._rotation.matrix_to_angles(permuted_rot)
    alpha, beta, gamma = rot_angles[0].cpu(), rot_angles[1].cpu(), rot_angles[2].cpu()

    # Compute Wigner-D matrices (note negative beta)
    D_1 = o3.wigner_D(1, alpha, -beta, gamma).to(shs_rest.device)  # 3×3
    D_2 = o3.wigner_D(2, alpha, -beta, gamma).to(shs_rest.device)  # 5×5
    D_3 = o3.wigner_D(3, alpha, -beta, gamma).to(shs_rest.device)  # 7×7

    # Block diagonal combination
    D = torch.block_diag(D_1, D_2, D_3)  # 15×15

    # Apply rotation: (N,15,3) → (N,3,15) → einsum → (N,3,15) → (N,15,3)
    sh = einops.rearrange(shs_rest, 'n s r -> n r s')  # (N, 3, 15)
    sh_rot = torch.einsum('ij, nrj -> nri', D, sh)     # (N, 3, 15)
    shs_rest_rotated = einops.rearrange(sh_rot, 'n r s -> n s r')  # (N, 15, 3)

    return shs_rest_rotated
