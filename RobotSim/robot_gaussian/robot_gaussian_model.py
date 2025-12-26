"""
Robot Gaussian Model
====================

Main model class coordinating all components.
Manages robot Gaussian data and applies FK transformations.

Key insight: robot.ply is ICP-aligned to Genesis*0.8 (scaled Genesis).
Instead of inverse-scaling robot.ply (which introduces errors), we scale
the FK transforms to work in the same scaled coordinate system.

Classes:
- RobotGaussianConfig: Configuration dataclass
- RobotGaussianModel: Main model class
"""

from dataclasses import dataclass, field
import torch
import numpy as np
from e3nn import o3
from Gaussians.util_gau import load_ply
from Gaussians.render import gaus_cuda_from_cpu
from .forward_kinematics import (
    get_link_states_genesis,
    get_transformation_list_scaled,
    LINK_NAMES
)
from .gaussian_transform import transform_means_scaled
from .segmentation import get_segmented_indices
from .sh_rotation import transform_shs


# Default ICP parameters from segment_robot.py / icp_with_scale.py
DEFAULT_R_Y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
DEFAULT_ICP_ROTATION = np.array([
    [0.98012743, -0.09234054, 0.17556605],
    [0.09228557, 0.99569634, 0.00849548],
    [-0.17559495, 0.00787556, 0.984431]
], dtype=np.float32)
# Updated translation to better align with background (X offset ~0.5)
DEFAULT_ICP_TRANSLATION = np.array([0.5, 0.1, 0.0], dtype=np.float32)

# Scale factor used during KNN training (Genesis was scaled by this)
DEFAULT_GENESIS_SCALE = 0.8


@dataclass
class RobotGaussianConfig:
    """Configuration for Robot Gaussian Model."""
    robot_ply_path: str = 'exports/mult-view-scene/robot.ply'
    labels_path: str = 'data/labels/so100_labels.npy'
    genesis_center_path: str = 'data/labels/genesis_center.npy'  # Saved from segment_robot.py
    initial_joint_states: list = None
    world_to_splat: np.ndarray = None  # world→COLMAP transformation

    # splat_to_world transform parameters (COLMAP → World)
    # These must match the parameters used in segment_robot.py
    R_y_90: np.ndarray = field(default_factory=lambda: DEFAULT_R_Y_90)
    icp_rotation: np.ndarray = field(default_factory=lambda: DEFAULT_ICP_ROTATION)
    icp_translation: np.ndarray = field(default_factory=lambda: DEFAULT_ICP_TRANSLATION)

    # Scale factor: Genesis was scaled by this for ICP alignment
    # robot.ply is aligned to Genesis*scale, FK is computed in scaled space
    genesis_scale: float = DEFAULT_GENESIS_SCALE

    # Supersplat transform parameters (Genesis → Background PLY coordinate system)
    # These should match the transform used for background PLY in supersplat
    # Set to None to skip supersplat transform (if background is in Genesis coords)
    supersplat_translation: list = None  # e.g., [0.34, 0.09, 0.42]
    supersplat_rotation_degrees: list = None  # e.g., [-34.29, 11.67, -227.35]
    supersplat_scale: float = None  # e.g., 0.81

    def __post_init__(self):
        if self.initial_joint_states is None:
            self.initial_joint_states = [0, -3.32, 3.11, 1.18, 0, -0.174]


class RobotGaussianModel:
    """
    Main model class for robot Gaussian splatting.

    Key strategy:
    - robot.ply is ICP-aligned to Genesis*0.8 (scaled Genesis)
    - We keep robot.ply in this scaled coordinate system (no inverse scaling)
    - FK transforms are computed for scaled Genesis positions
    - This ensures FK works correctly without introducing scaling errors
    """

    def __init__(self, config: RobotGaussianConfig, arm):
        """
        Initialize Robot Gaussian Model.

        Args:
            config: Configuration object
            arm: Genesis arm entity (must be at initial pose before calling)
        """
        self.config = config
        self.arm = arm

        # 1. Load robot.ply (in COLMAP coordinates)
        gau_cpu = load_ply(config.robot_ply_path)
        self.gaussians = gaus_cuda_from_cpu(gau_cpu)

        # 2. Load segmentation labels
        labels = np.load(config.labels_path)
        self.segmented_list = get_segmented_indices(labels, num_links=7)

        # 3. Compute world_to_splat as EXACT INVERSE of splat_to_world (ICP-based)
        # This ensures consistent coordinate transformation
        R_y_90 = torch.tensor(config.R_y_90, device='cuda', dtype=torch.float32)
        icp_rotation = torch.tensor(config.icp_rotation, device='cuda', dtype=torch.float32)
        icp_translation = torch.tensor(config.icp_translation, device='cuda', dtype=torch.float32)
        R_combined = icp_rotation @ R_y_90

        # Inverse: xyz_colmap = R_combined^T @ (xyz_world - icp_translation)
        R_combined_inv = R_combined.T
        t_inv = -R_combined_inv @ icp_translation

        self.world_to_splat = torch.eye(4, device='cuda', dtype=torch.float32)
        self.world_to_splat[:3, :3] = R_combined_inv
        self.world_to_splat[:3, 3] = t_inv

        # Store R_combined for later use
        self.R_combined = R_combined
        self.icp_translation = icp_translation

        # 4. Load genesis_center (saved by segment_robot.py)
        # This is used for scaled FK computation
        self.genesis_center = np.load(config.genesis_center_path)

        # 5. IMPORTANT: Arm must already be at initial pose before calling this!
        # The caller should do: arm.set_dofs_position(initial_joints); scene.step()

        # 6. Apply splat_to_world transformation (COLMAP → scaled World)
        # This converts robot.ply to match Genesis*0.8 (NO inverse scaling)
        self._apply_splat_to_world_transform(config)

        # 7. Save initial link states (at reference pose)
        self.initial_link_states = get_link_states_genesis(arm, LINK_NAMES)

        # 8. Backup original Gaussian data (in scaled World coordinates)
        self.backup_xyz = self.gaussians.xyz.clone()
        self.backup_rot = self.gaussians.rot.clone()
        self.backup_scale = self.gaussians.scale.clone()
        self.backup_sh = self.gaussians.sh.clone()

        # 9. Build supersplat transform matrix (if parameters provided)
        self.supersplat_transform = None
        if config.supersplat_translation is not None:
            self.supersplat_transform = self._build_supersplat_transform(
                config.supersplat_translation,
                config.supersplat_rotation_degrees,
                config.supersplat_scale
            )
            print(f"[RobotGaussian] Supersplat transform enabled")
            print(f"   Translation: {config.supersplat_translation}")
            print(f"   Rotation: {config.supersplat_rotation_degrees}")
            print(f"   Scale: {config.supersplat_scale}")

    def _build_supersplat_transform(self, translation, rotation_degrees, scale):
        """
        Build supersplat transform matrix (Genesis → Background PLY coordinate system).

        This matches the transform_matrix3 function used in base_task.py for supersplat alignment.
        The transformation order is: translate → rotate around translated center → scale around center

        Args:
            translation: [tx, ty, tz]
            rotation_degrees: [rx, ry, rz] in degrees
            scale: scale factor

        Returns:
            4x4 transformation matrix
        """
        from scipy.spatial.transform import Rotation as R

        t = np.array(translation)
        r_deg = np.array(rotation_degrees)
        s = scale if scale is not None else 1.0

        # Build rotation matrix
        rot = R.from_euler('xyz', r_deg, degrees=True)
        R_mat = rot.as_matrix()

        # Build 4x4 transform: scale and rotate around translation center
        # This matches supersplat's behavior: T_final = S_center @ R_center @ T_translate
        center = t

        # Translation matrix
        T_translate = np.eye(4)
        T_translate[:3, 3] = t

        # Rotation around center
        T_neg = np.eye(4)
        T_neg[:3, 3] = -center
        T_pos = np.eye(4)
        T_pos[:3, 3] = center

        # Individual rotation matrices
        def rot_mat_4x4(axis, angle_deg):
            angle = np.radians(angle_deg)
            c, s = np.cos(angle), np.sin(angle)
            if axis == 'x':
                return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])
            elif axis == 'y':
                return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])
            else:  # z
                return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])

        R_x = rot_mat_4x4('x', r_deg[0])
        R_y = rot_mat_4x4('y', r_deg[1])
        R_z = rot_mat_4x4('z', r_deg[2])

        # Combined rotation around center
        R_combined = T_pos @ R_x @ T_neg
        R_combined = T_pos @ R_y @ T_neg @ R_combined
        R_combined = T_pos @ R_z @ T_neg @ R_combined

        # Scale around center
        S = np.diag([s, s, s, 1])
        S_center = T_pos @ S @ T_neg

        # Final transform
        T_final = S_center @ R_combined @ T_translate

        return torch.tensor(T_final, device='cuda', dtype=torch.float32)

    def _apply_splat_to_world_transform(self, config):
        """
        Apply splat_to_world transformation to convert robot.ply from COLMAP to scaled World coords.
        This matches the transformation used in segment_robot.py for KNN training.

        The transformation is:
            xyz_aligned = icp_rotation @ (R_y_90 @ xyz_colmap) + icp_translation

        NOTE: We do NOT apply inverse scaling here. robot.ply stays aligned to Genesis*0.8.
        FK will be computed in the same scaled coordinate system.

        Args:
            config: Configuration with transformation parameters
        """
        # Get transformation parameters
        R_y_90 = torch.tensor(config.R_y_90, device='cuda', dtype=torch.float32)
        icp_rotation = torch.tensor(config.icp_rotation, device='cuda', dtype=torch.float32)
        icp_translation = torch.tensor(config.icp_translation, device='cuda', dtype=torch.float32)

        # Combined rotation: R_combined = icp_rotation @ R_y_90
        R_combined = icp_rotation @ R_y_90

        # Transform position to align with scaled Genesis (NO inverse scaling!)
        xyz = self.gaussians.xyz
        xyz_aligned = (R_combined @ xyz.T).T + icp_translation
        self.gaussians.xyz = xyz_aligned

        # Transform Gaussian rotation (quaternion)
        # PLY uses wxyz quaternion format, e3nn also uses wxyz
        rot = self.gaussians.rot
        rot_mat = o3.quaternion_to_matrix(rot)  # (N, 3, 3)
        rot_mat = R_combined @ rot_mat  # Apply rotation to each Gaussian's rotation matrix
        self.gaussians.rot = o3.matrix_to_quaternion(rot_mat)

        # Transform SH coefficients (rotate non-DC part)
        sh = self.gaussians.sh  # (N, 16, 3)
        shs_dc = sh[:, :1, :]       # (N, 1, 3) - DC unchanged
        shs_rest = sh[:, 1:, :]     # (N, 15, 3) - rest rotated
        shs_rest = transform_shs(shs_rest, R_combined)
        self.gaussians.sh = torch.cat([shs_dc, shs_rest], dim=1)

        # NOTE: We do NOT scale Gaussian scale parameters here
        # because we're staying in the scaled coordinate system

    def update(self, arm, skip_fk=False):
        """
        Update Gaussian positions based on current joint states.

        Args:
            arm: Genesis arm entity
            skip_fk: If True, skip FK transform (for debugging base alignment)
        """
        # Restore from backup (scaled world coordinates)
        self.gaussians.xyz = self.backup_xyz.clone()
        self.gaussians.rot = self.backup_rot.clone()
        self.gaussians.scale = self.backup_scale.clone()
        self.gaussians.sh = self.backup_sh.clone()

        if not skip_fk:
            # Compute FK transformations in SCALED coordinate system
            # This matches robot.ply which is aligned to Genesis*0.8
            transformations = get_transformation_list_scaled(
                arm,
                self.initial_link_states,
                LINK_NAMES,
                self.genesis_center,
                self.config.genesis_scale
            )

            # Transform Gaussians: scaled FK + transform to unscaled Genesis
            self.gaussians = transform_means_scaled(
                self.gaussians,
                self.segmented_list,
                transformations,
                self.world_to_splat,
                self.genesis_center,
                self.config.genesis_scale
            )
        else:
            # Even without FK, need to unscale from Genesis*0.8 to Genesis*1.0
            C = torch.tensor(self.genesis_center, device='cuda', dtype=torch.float32)
            self.gaussians.xyz = (self.gaussians.xyz - C) / self.config.genesis_scale + C
            self.gaussians.scale = self.gaussians.scale / self.config.genesis_scale

        # Apply supersplat transform to align with background PLY
        if self.supersplat_transform is not None:
            self._apply_supersplat_transform()

    def _apply_supersplat_transform(self):
        """
        Apply supersplat transform to align robot Gaussians with background PLY.

        Background PLY has supersplat transform baked in, so robot needs the same
        transform to appear in the correct position relative to the background.
        """
        from e3nn import o3
        from .sh_rotation import transform_shs

        T = self.supersplat_transform

        # Extract rotation and translation
        R_mat = T[:3, :3]
        t_vec = T[:3, 3]

        # Extract pure rotation (remove scale via SVD)
        U, S, Vh = torch.linalg.svd(R_mat)
        R_pure = U @ Vh
        scale_factor = S.prod().pow(1/3)

        # Transform position
        xyz = self.gaussians.xyz
        xyz_transformed = (R_mat @ xyz.T).T + t_vec
        self.gaussians.xyz = xyz_transformed

        # Transform Gaussian rotation (pure rotation only)
        rot = self.gaussians.rot
        rot_mat = o3.quaternion_to_matrix(rot)
        rot_mat = R_pure @ rot_mat
        self.gaussians.rot = o3.matrix_to_quaternion(rot_mat)

        # Scale Gaussian scale parameters
        self.gaussians.scale = self.gaussians.scale * scale_factor

        # Rotate SH coefficients
        sh = self.gaussians.sh
        shs_dc = sh[:, :1, :]
        shs_rest = sh[:, 1:, :]
        shs_rest = transform_shs(shs_rest, R_pure)
        self.gaussians.sh = torch.cat([shs_dc, shs_rest], dim=1)

    def get_gaussians(self):
        """
        Get current Gaussian data.

        Returns:
            GaussianDataCUDA object
        """
        return self.gaussians
