"""
Robot Gaussian Model
====================

Main model class coordinating all components.
Manages robot Gaussian data and applies FK transformations.

Classes:
- RobotGaussianConfig: Configuration dataclass
- RobotGaussianModel: Main model class
"""

from dataclasses import dataclass
import torch
import numpy as np
from Gaussians.util_gau import load_ply
from Gaussians.render import gaus_cuda_from_cpu
from .forward_kinematics import get_link_states_genesis, get_transformation_list, LINK_NAMES
from .gaussian_transform import transform_means
from .segmentation import get_segmented_indices


@dataclass
class RobotGaussianConfig:
    """Configuration for Robot Gaussian Model."""
    robot_ply_path: str = 'exports/mult-view-scene/robot.ply'
    labels_path: str = 'data/labels/so100_labels.npy'
    initial_joint_states: list = None
    world_to_splat: np.ndarray = None  # world→COLMAP transformation

    def __post_init__(self):
        if self.initial_joint_states is None:
            self.initial_joint_states = [0, -3.32, 3.11, 1.18, 0, -0.174]


class RobotGaussianModel:
    """
    Main model class for robot Gaussian splatting.

    Manages robot Gaussian data, applies FK transformations, and
    transforms to COLMAP coordinate system.
    """

    def __init__(self, config: RobotGaussianConfig, arm):
        """
        Initialize Robot Gaussian Model.

        Args:
            config: Configuration object
            arm: Genesis arm entity
        """
        self.config = config
        self.arm = arm

        # 1. Load robot.ply (in world coordinates)
        gau_cpu = load_ply(config.robot_ply_path)
        self.gaussians = gaus_cuda_from_cpu(gau_cpu)

        # 2. Load segmentation labels
        labels = np.load(config.labels_path)
        self.segmented_list = get_segmented_indices(labels, num_links=7)

        # 3. world_to_splat transformation
        self.world_to_splat = torch.tensor(
            config.world_to_splat, device='cuda', dtype=torch.float32
        )

        # 4. Save initial link states
        arm.set_dofs_position(config.initial_joint_states)
        # scene.step() should be called externally
        self.initial_link_states = get_link_states_genesis(arm, LINK_NAMES)

        # 5. Backup original Gaussian data (in world coordinates)
        self.backup_xyz = self.gaussians.xyz.clone()
        self.backup_rot = self.gaussians.rot.clone()
        self.backup_sh = self.gaussians.sh.clone()

    def update(self, arm):
        """
        Update Gaussian positions based on current joint states.

        Args:
            arm: Genesis arm entity
        """
        # Restore from backup (world coordinates)
        self.gaussians.xyz = self.backup_xyz.clone()
        self.gaussians.rot = self.backup_rot.clone()
        self.gaussians.sh = self.backup_sh.clone()

        # Compute FK transformations
        transformations = get_transformation_list(
            arm, self.initial_link_states, LINK_NAMES
        )

        # Transform Gaussians: FK in world + transform to COLMAP
        self.gaussians = transform_means(
            self.gaussians,
            self.segmented_list,
            transformations,
            self.world_to_splat
        )

    def get_gaussians(self):
        """
        Get current Gaussian data.

        Returns:
            GaussianDataCUDA object
        """
        return self.gaussians
