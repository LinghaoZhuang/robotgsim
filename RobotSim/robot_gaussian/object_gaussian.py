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

    # Path to ICP params JSON file (if set, overrides icp_rotation/icp_translation)
    icp_params_path: Optional[str] = None

    # Genesis initial pose (for computing relative transforms)
    initial_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    initial_quat: List[float] = field(default_factory=lambda: [1, 0, 0, 0])  # w, x, y, z

    # If True, use PLY center as initial_pos (for cropped objects)
    use_ply_center: bool = False

    # Supersplat transform parameters (Genesis → Background PLY coordinate system)
    # These should match the transform used for background PLY in supersplat
    # Set to None to skip supersplat transform
    supersplat_translation: Optional[List[float]] = None
    supersplat_rotation_degrees: Optional[List[float]] = None
    supersplat_scale: Optional[float] = None


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

        # Load ICP parameters from JSON if provided
        icp_rotation = config.icp_rotation
        icp_translation = config.icp_translation

        if config.icp_params_path:
            import json
            from pathlib import Path
            params_path = Path(config.icp_params_path)
            if params_path.exists():
                with open(params_path, 'r') as f:
                    icp_params = json.load(f)
                    icp_rotation = icp_params['icp_rotation']
                    icp_translation = icp_params['icp_translation']
                    print(f"[ObjectGaussian] Loaded ICP params from {params_path}")
                    print(f"   Fitness: {icp_params.get('fitness', 'N/A')}, RMSE: {icp_params.get('rmse', 'N/A')}")
            else:
                print(f"[ObjectGaussian] WARNING: ICP params file not found: {params_path}")
                print(f"   Using default ICP parameters (identity)")

        # ICP alignment parameters
        self.icp_rotation = torch.tensor(icp_rotation, device=self.device, dtype=torch.float32)
        self.icp_translation = torch.tensor(icp_translation, device=self.device, dtype=torch.float32)

        # Apply ICP alignment (identity if cropped from scene)
        self.aligned_xyz = (self.icp_rotation @ self.original_xyz.T).T + self.icp_translation
        self.aligned_rot = self._rotate_quaternions(self.original_rot, self.icp_rotation)

        # Build supersplat transform matrix (if provided)
        self.supersplat_transform = None
        if config.supersplat_translation is not None:
            self.supersplat_transform = self._build_supersplat_transform(
                config.supersplat_translation,
                config.supersplat_rotation_degrees,
                config.supersplat_scale
            )
            print(f"[ObjectGaussian] Supersplat transform enabled")

            # Apply supersplat transform to aligned Gaussians
            self.aligned_xyz, self.aligned_rot, self.scale = self._apply_supersplat_to_gaussians(
                self.aligned_xyz, self.aligned_rot, self.scale
            )

        # Set initial position
        if config.use_ply_center:
            # Use PLY center as initial position (for cropped objects)
            self.initial_pos = self.aligned_xyz.mean(dim=0)
            print(f"[ObjectGaussian] Using PLY center as initial_pos: {self.initial_pos.cpu().numpy()}")
        else:
            self.initial_pos = torch.tensor(config.initial_pos, device=self.device, dtype=torch.float32)
            # Apply supersplat transform to initial_pos if enabled
            if self.supersplat_transform is not None:
                pos_homo = torch.cat([self.initial_pos, torch.ones(1, device=self.device)])
                self.initial_pos = (self.supersplat_transform @ pos_homo)[:3]

        self.initial_quat = torch.tensor(config.initial_quat, device=self.device, dtype=torch.float32)

        # Compute initial rotation matrix from quaternion (w, x, y, z)
        self.initial_rot_mat = self._quat_to_matrix(self.initial_quat)

        # Current transformed Gaussians
        self.current_xyz = self.aligned_xyz.clone()
        self.current_rot = self.aligned_rot.clone()

    def _build_supersplat_transform(self, translation, rotation_degrees, scale):
        """
        Build supersplat transform matrix (Genesis → Background PLY coordinate system).

        This matches the transform_matrix3 function used in base_task.py for supersplat alignment.
        """
        import numpy as np

        t = np.array(translation)
        r_deg = np.array(rotation_degrees) if rotation_degrees else np.zeros(3)
        s = scale if scale is not None else 1.0

        center = t

        # Translation matrix
        T_translate = np.eye(4)
        T_translate[:3, 3] = t

        # Rotation around center
        T_neg = np.eye(4)
        T_neg[:3, 3] = -center
        T_pos = np.eye(4)
        T_pos[:3, 3] = center

        def rot_mat_4x4(axis, angle_deg):
            angle = np.radians(angle_deg)
            c, s_ = np.cos(angle), np.sin(angle)
            if axis == 'x':
                return np.array([[1,0,0,0],[0,c,-s_,0],[0,s_,c,0],[0,0,0,1]])
            elif axis == 'y':
                return np.array([[c,0,s_,0],[0,1,0,0],[-s_,0,c,0],[0,0,0,1]])
            else:
                return np.array([[c,-s_,0,0],[s_,c,0,0],[0,0,1,0],[0,0,0,1]])

        R_x = rot_mat_4x4('x', r_deg[0])
        R_y = rot_mat_4x4('y', r_deg[1])
        R_z = rot_mat_4x4('z', r_deg[2])

        R_combined = T_pos @ R_x @ T_neg
        R_combined = T_pos @ R_y @ T_neg @ R_combined
        R_combined = T_pos @ R_z @ T_neg @ R_combined

        S = np.diag([s, s, s, 1])
        S_center = T_pos @ S @ T_neg

        T_final = S_center @ R_combined @ T_translate

        return torch.tensor(T_final, device=self.device, dtype=torch.float32)

    def _apply_supersplat_to_gaussians(self, xyz, rot, scale):
        """Apply supersplat transform to Gaussians."""
        T = self.supersplat_transform

        # Extract rotation and translation
        R_mat = T[:3, :3]
        t_vec = T[:3, 3]

        # Extract pure rotation (remove scale via SVD)
        U, S, Vh = torch.linalg.svd(R_mat)
        R_pure = U @ Vh
        scale_factor = S.prod().pow(1/3)

        # Transform position
        xyz_transformed = (R_mat @ xyz.T).T + t_vec

        # Transform Gaussian rotation
        rot_transformed = self._rotate_quaternions(rot, R_pure)

        # Scale Gaussian scale parameters
        scale_transformed = scale * scale_factor

        return xyz_transformed, rot_transformed, scale_transformed

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

        For now, keep static (don't track movement) to debug position.
        """
        # DEBUG: Keep original position, don't track Genesis movement
        # This ensures object appears at scanned position in scene
        pass  # Do nothing - keep aligned_xyz as current_xyz

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
