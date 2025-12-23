"""
Robot Gaussian Module
=====================

This module provides Gaussian Splatting control for articulated robots.
It implements FK-based Gaussian transformation for robot arms in the Genesis simulator.

Key components:
- forward_kinematics: FK calculation for robot joints
- sh_rotation: Spherical harmonics rotation using Wigner-D matrices
- gaussian_transform: Apply FK transformations to Gaussians
- segmentation: KNN-based segmentation of robot Gaussians by joint
- robot_gaussian_model: Main model class coordinating all components
"""

from .forward_kinematics import (
    LINK_NAMES,
    quaternion_to_rot_matrix,
    compute_transformation,
    get_link_states_genesis,
    get_transformation_list,
    get_transformation_list_scaled
)

from .sh_rotation import transform_shs

from .gaussian_transform import transform_means, transform_means_scaled

from .segmentation import (
    get_link_point_clouds_genesis,
    train_segmentation_knn,
    segment_gaussians,
    get_segmented_indices
)

from .robot_gaussian_model import (
    RobotGaussianConfig,
    RobotGaussianModel
)

__all__ = [
    'LINK_NAMES',
    'quaternion_to_rot_matrix',
    'compute_transformation',
    'get_link_states_genesis',
    'get_transformation_list',
    'get_transformation_list_scaled',
    'transform_shs',
    'transform_means',
    'transform_means_scaled',
    'get_link_point_clouds_genesis',
    'train_segmentation_knn',
    'segment_gaussians',
    'get_segmented_indices',
    'RobotGaussianConfig',
    'RobotGaussianModel',
]
