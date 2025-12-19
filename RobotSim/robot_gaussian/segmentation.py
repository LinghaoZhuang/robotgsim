"""
Segmentation Module
===================

KNN-based segmentation of robot Gaussians by joint.
Verified: link.get_vverts() and robot.ply are both in world coordinates.

Key functions:
- get_link_point_clouds_genesis: Extract link point clouds from Genesis
- train_segmentation_knn: Train KNN classifier
- segment_gaussians: Predict Gaussian labels
- get_segmented_indices: Convert labels to index lists
"""

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from .forward_kinematics import LINK_NAMES


def get_link_point_clouds_genesis(arm, link_names):
    """
    Get mesh vertices for each link (world coordinate system).

    Args:
        arm: Genesis arm entity
        link_names: List of link names

    Returns:
        List of point clouds (each is numpy array of shape (M, 3))
    """
    point_clouds = []
    for name in link_names:
        link = arm.get_link(name)
        # vverts are already in world coordinates, use directly
        verts_world = link.get_vverts().cpu().numpy()  # (M, 3)
        point_clouds.append(verts_world)
    return point_clouds


def train_segmentation_knn(point_clouds, n_neighbors=10):
    """
    Train KNN classifier for segmentation.

    Args:
        point_clouds: List of point clouds for each link
        n_neighbors: Number of neighbors for KNN

    Returns:
        Trained KNeighborsClassifier
    """
    X = np.vstack(point_clouds)
    y = np.hstack([np.full(len(pc), i) for i, pc in enumerate(point_clouds)])
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    return knn


def segment_gaussians(gaussian_xyz, knn):
    """
    Predict link assignment for each Gaussian.

    Args:
        gaussian_xyz: (N, 3) robot.ply xyz coordinates (world coordinates)
        knn: Trained KNN classifier

    Returns:
        labels: (N,) link index for each Gaussian (0=Base, 1=Rotation_Pitch, ...)
    """
    # robot.ply is already in world coordinates, predict directly
    labels = knn.predict(gaussian_xyz)
    return labels


def get_segmented_indices(labels, num_links=7):
    """
    Convert labels to index lists for each link.

    Args:
        labels: (N,) numpy array of link indices
        num_links: Number of links (default 7)

    Returns:
        List of torch.Tensor indices for each link
    """
    labels_tensor = torch.from_numpy(labels).cuda().long()
    segmented_list = []
    for i in range(num_links):
        indices = torch.where(labels_tensor == i)[0]
        segmented_list.append(indices)
    return segmented_list
