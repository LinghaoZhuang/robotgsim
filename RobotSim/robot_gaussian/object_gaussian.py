"""
Object Gaussian Model
=====================

Manage Gaussian Splatting representation of rigid objects.
Unlike the robot which requires FK, objects only need rigid body transforms.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.spatial.transform import Rotation as R

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from Gaussians.util_gau import load_ply
from Gaussians.render import GaussianDataCUDA


@dataclass
class ObjectGaussianConfig:
    """Configuration for an object's Gaussian representation."""

    ply_path: str  # Path to object PLY file

    # ICP alignment parameters (PLY -> Genesis initial pose)
    icp_rotation: List[List[float]] = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    icp_translation: List[float] = field(default_factory=lambda: [0, 0, 0])

    # Genesis initial pose (for computing relative transforms)
    initial_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    initial_quat: List[float] = field(default_factory=lambda: [1, 0, 0, 0])  # w, x, y, z


class ObjectGaussianModel:
    """
    Manages Gaussian representation of a single rigid object.

    The object undergoes rigid body transformation:
    1. Load PLY Gaussians
    2. Apply ICP alignment to match Genesis initial pose
    3. Track relative transform from initial to current pose
    4. Apply combined transform for rendering
    """

    def __init__(self, config: ObjectGaussianConfig):
        self.config = config
        self.device = 'cuda'

        # Load Gaussians from PLY
        gau_data = load_ply(config.ply_path)
        self.original_xyz = torch.tensor(gau_data.xyz, device=self.device, dtype=torch.float32)
        self.original_rot = torch.tensor(gau_data.rot, device=self.device, dtype=torch.float32)
        self.scale = torch.tensor(gau_data.scale, device=self.device, dtype=torch.float32)
        self.opacity = torch.tensor(gau_data.opacity, device=self.device, dtype=torch.float32)
        self.sh = torch.tensor(gau_data.sh, device=self.device, dtype=torch.float32)

        # Reshape SH coefficients
        self.sh = self.sh.reshape(len(self.original_xyz), -1, 3).contiguous()

        # ICP alignment parameters
        self.icp_rotation = torch.tensor(config.icp_rotation, device=self.device, dtype=torch.float32)
        self.icp_translation = torch.tensor(config.icp_translation, device=self.device, dtype=torch.float32)

        # Initial Genesis pose
        self.initial_pos = torch.tensor(config.initial_pos, device=self.device, dtype=torch.float32)
        self.initial_quat = torch.tensor(config.initial_quat, device=self.device, dtype=torch.float32)

        # Compute initial rotation matrix from quaternion (w, x, y, z)
        self.initial_rot_mat = self._quat_to_matrix(self.initial_quat)

        # Apply ICP alignment to get Gaussians at Genesis initial pose
        self.aligned_xyz = (self.icp_rotation @ self.original_xyz.T).T + self.icp_translation
        self.aligned_rot = self._rotate_quaternions(self.original_rot, self.icp_rotation)

        # Current transformed Gaussians
        self.current_xyz = self.aligned_xyz.clone()
        self.current_rot = self.aligned_rot.clone()

    def _quat_to_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (w, x, y, z) to rotation matrix."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        R_mat = torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], device=self.device, dtype=torch.float32)

        return R_mat

    def _rotate_quaternions(self, quats: torch.Tensor, R_mat: torch.Tensor) -> torch.Tensor:
        """Rotate quaternions by a rotation matrix."""
        # Convert rotation matrix to quaternion
        # Using scipy for simplicity
        R_np = R_mat.cpu().numpy()
        r = R.from_matrix(R_np)
        delta_quat = r.as_quat()  # x, y, z, w format
        delta_quat_wxyz = torch.tensor(
            [delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]],
            device=self.device, dtype=torch.float32
        )

        # Quaternion multiplication: q_new = q_delta * q_original
        return self._quat_multiply(delta_quat_wxyz, quats)

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiply quaternions. q1 is a single quaternion (4,), q2 is batch (N, 4).
        Both in (w, x, y, z) format.
        """
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]

        if q2.dim() == 1:
            w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        else:
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        if q2.dim() == 1:
            return torch.stack([w, x, y, z])
        else:
            return torch.stack([w, x, y, z], dim=1)

    def update(self, pos: np.ndarray, quat: np.ndarray):
        """
        Update Gaussians based on current Genesis pose.

        Args:
            pos: Current position [x, y, z]
            quat: Current quaternion [w, x, y, z]
        """
        pos = torch.tensor(pos, device=self.device, dtype=torch.float32)
        quat = torch.tensor(quat, device=self.device, dtype=torch.float32)

        # Compute relative transform from initial to current pose
        current_rot_mat = self._quat_to_matrix(quat)

        # Relative rotation: R_rel = R_current @ R_initial^T
        R_rel = current_rot_mat @ self.initial_rot_mat.T

        # Apply rotation around object center (initial_pos), then translate to current pos
        # 1. Center the aligned points around initial_pos
        centered = self.aligned_xyz - self.initial_pos
        # 2. Apply relative rotation
        rotated = (R_rel @ centered.T).T
        # 3. Translate to current position
        self.current_xyz = rotated + pos

        # Rotate quaternions
        self.current_rot = self._rotate_quaternions(self.aligned_rot, R_rel)

    def get_gaussians(self) -> GaussianDataCUDA:
        """Return current Gaussian data for rendering."""
        return GaussianDataCUDA(
            xyz=self.current_xyz,
            rot=self.current_rot,
            scale=self.scale,
            opacity=self.opacity,
            sh=self.sh
        )

    def reset(self):
        """Reset to aligned (initial) pose."""
        self.current_xyz = self.aligned_xyz.clone()
        self.current_rot = self.aligned_rot.clone()
