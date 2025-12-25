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
    # Set to identity if PLY is already aligned (e.g., cropped from scene)
    icp_rotation: List[List[float]] = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    icp_translation: List[float] = field(default_factory=lambda: [0, 0, 0])

    # Genesis initial pose (for computing relative transforms)
    initial_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    initial_quat: List[float] = field(default_factory=lambda: [1, 0, 0, 0])  # w, x, y, z

    # If True, use PLY center as initial_pos (for cropped objects)
    use_ply_center: bool = False

    # Scene transform parameters (to align with background Gaussian)
    # These should match the transform used for background PLY
    scene_translation: List[float] = field(default_factory=lambda: [0, 0, 0])
    scene_rotation_degrees: List[float] = field(default_factory=lambda: [0, 0, 0])
    scene_scale: float = 1.0


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

        # Apply ICP alignment (identity if cropped from scene)
        self.aligned_xyz = (self.icp_rotation @ self.original_xyz.T).T + self.icp_translation
        self.aligned_rot = self._rotate_quaternions(self.original_rot, self.icp_rotation)

        # Build scene transform matrix (to align with background Gaussian)
        self.scene_transform = self._build_scene_transform(
            config.scene_translation,
            config.scene_rotation_degrees,
            config.scene_scale
        )

        # Apply scene transform to initial_pos as well
        if config.use_ply_center:
            # Use PLY center as initial position (for cropped objects)
            self.initial_pos = self.aligned_xyz.mean(dim=0)
            print(f"[ObjectGaussian] Using PLY center as initial_pos: {self.initial_pos.cpu().numpy()}")
        else:
            self.initial_pos = torch.tensor(config.initial_pos, device=self.device, dtype=torch.float32)

        # Transform initial_pos to scene coordinate system
        self.initial_pos_scene = self._apply_scene_transform_point(self.initial_pos)

        self.initial_quat = torch.tensor(config.initial_quat, device=self.device, dtype=torch.float32)

        # Compute initial rotation matrix from quaternion (w, x, y, z)
        self.initial_rot_mat = self._quat_to_matrix(self.initial_quat)

        # Current transformed Gaussians
        self.current_xyz = self.aligned_xyz.clone()
        self.current_rot = self.aligned_rot.clone()

    def _build_scene_transform(self, translation, rotation_degrees, scale):
        """Build 4x4 scene transform matrix."""
        import numpy as np

        # Convert to numpy for easier manipulation
        t = np.array(translation)
        r_deg = np.array(rotation_degrees)

        # Build rotation matrix from euler angles
        r = R.from_euler('xyz', r_deg, degrees=True)
        R_mat = r.as_matrix()

        # Build 4x4 transform: T @ R @ S
        transform = np.eye(4)
        transform[:3, :3] = R_mat * scale
        transform[:3, 3] = t

        return torch.tensor(transform, device=self.device, dtype=torch.float32)

    def _apply_scene_transform_point(self, point):
        """Apply scene transform to a single 3D point."""
        point_homo = torch.cat([point, torch.ones(1, device=self.device)])
        transformed = self.scene_transform @ point_homo
        return transformed[:3]

    def _apply_scene_transform_points(self, points):
        """Apply scene transform to multiple 3D points."""
        N = points.shape[0]
        points_homo = torch.cat([points, torch.ones(N, 1, device=self.device)], dim=1)
        transformed = (self.scene_transform @ points_homo.T).T
        return transformed[:, :3]

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
            pos: Current position [x, y, z] in Genesis coordinate
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
        # 3. Translate to current position (in Genesis coordinate)
        genesis_xyz = rotated + pos

        # 4. Apply scene transform to convert to rendering coordinate system
        self.current_xyz = self._apply_scene_transform_points(genesis_xyz)

        # Rotate quaternions (scene rotation also affects quaternions)
        scene_rot = self.scene_transform[:3, :3]
        combined_rot = scene_rot @ R_rel
        self.current_rot = self._rotate_quaternions(self.aligned_rot, combined_rot)

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
