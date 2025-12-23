"""
Forward Kinematics Module
==========================

Implements forward kinematics for Genesis robot arm.
Computes relative transformations between initial and current joint poses.

Key functions:
- quaternion_to_rot_matrix: Convert quaternion (xyzw) to rotation matrix
- compute_transformation: Calculate relative transform between two poses
- get_link_states_genesis: Get current link states from Genesis
- get_transformation_list: Compute FK transforms for all links
"""

import numpy as np
import torch

# Genesis link names from so_arm100.xml
LINK_NAMES = [
    "Base",              # Fixed, label=0, identity transform
    "Rotation_Pitch",    # joint 0: Rotation (Y-axis)
    "Upper_Arm",         # joint 1: Pitch (X-axis)
    "Lower_Arm",         # joint 2: Elbow (X-axis)
    "Wrist_Pitch_Roll",  # joint 3: Wrist_Pitch (X-axis)
    "Fixed_Jaw",         # joint 4: Wrist_Roll (Y-axis)
    "Moving_Jaw"         # joint 5: Jaw (Z-axis)
]


def quaternion_to_rot_matrix(quat):
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: Quaternion in (x, y, z, w) format

    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = quat
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def compute_transformation(pose_a, pose_b):
    """
    Compute relative transformation from pose_a to pose_b.

    Args:
        pose_a: Initial pose with attributes 'pos' and 'quat'
        pose_b: Current pose with attributes 'pos' and 'quat'

    Returns:
        r_rel: Relative rotation matrix (3x3)
        t: Relative translation vector (3,)
    """
    r1 = quaternion_to_rot_matrix(pose_a.quat)
    r2 = quaternion_to_rot_matrix(pose_b.quat)
    r_rel = r2 @ r1.T
    t = pose_b.pos - r_rel @ pose_a.pos
    return r_rel, t


def get_link_states_genesis(arm, link_names):
    """
    Get Genesis link states (world coordinate system).

    Args:
        arm: Genesis arm entity
        link_names: List of link names

    Returns:
        List of dicts with keys 'pos' and 'quat' (wxyz format)
    """
    states = []
    for name in link_names:
        link = arm.get_link(name)
        pos = link.get_pos().cpu().numpy()
        quat_wxyz = link.get_quat().cpu().numpy()  # wxyz format from Genesis
        states.append({'pos': pos, 'quat': quat_wxyz})
    return states


def get_transformation_list(arm, initial_states, link_names):
    """
    Compute FK transformations for all links (skipping Base).

    Args:
        arm: Genesis arm entity
        initial_states: Initial link states from get_link_states_genesis
        link_names: List of link names

    Returns:
        List of tuples (R_rel, T) for each joint (length = len(link_names) - 1)
        Each tuple contains:
        - R_rel: torch.Tensor (3, 3) relative rotation matrix
        - T: torch.Tensor (3,) relative translation vector
    """
    current_states = get_link_states_genesis(arm, link_names)
    transformations = []

    for i in range(1, len(link_names)):  # Start from 1, skip Base
        # Convert wxyz → xyzw for FK calculation
        quat_init_xyzw = np.roll(initial_states[i]['quat'], -1)  # wxyz → xyzw
        quat_curr_xyzw = np.roll(current_states[i]['quat'], -1)

        # Create pose objects
        pose_init = type('obj', (object,), {
            'pos': initial_states[i]['pos'],
            'quat': quat_init_xyzw
        })
        pose_curr = type('obj', (object,), {
            'pos': current_states[i]['pos'],
            'quat': quat_curr_xyzw
        })

        r_rel, t = compute_transformation(pose_init, pose_curr)
        transformations.append((
            torch.from_numpy(r_rel).cuda().float(),
            torch.from_numpy(t).cuda().float()
        ))

    return transformations


def get_transformation_list_scaled(arm, initial_states, link_names, genesis_center, scale_factor=0.8):
    """
    Compute FK transformations for scaled coordinate system.

    Use this when robot.ply is aligned to Genesis*scale_factor (e.g., 0.8).
    The transforms are computed for the scaled positions so FK works correctly
    without needing to inverse-scale robot.ply.

    Args:
        arm: Genesis arm entity
        initial_states: Initial link states from get_link_states_genesis
        link_names: List of link names
        genesis_center: Center point used for scaling (numpy array, shape (3,))
        scale_factor: Scale factor applied to Genesis during ICP (default 0.8)

    Returns:
        List of tuples (R_rel, T_scaled) for each joint
    """
    current_states = get_link_states_genesis(arm, link_names)
    transformations = []

    C = genesis_center  # Center for scaling

    for i in range(1, len(link_names)):  # Start from 1, skip Base
        # Convert wxyz → xyzw for FK calculation
        quat_init_xyzw = np.roll(initial_states[i]['quat'], -1)
        quat_curr_xyzw = np.roll(current_states[i]['quat'], -1)

        # Get positions
        pos_init = initial_states[i]['pos']
        pos_curr = current_states[i]['pos']

        # Scale positions to match robot.ply's coordinate system
        # robot.ply is aligned to Genesis*0.8, so scale Genesis positions by 0.8
        pos_init_scaled = (pos_init - C) * scale_factor + C
        pos_curr_scaled = (pos_curr - C) * scale_factor + C

        # Create pose objects with scaled positions
        pose_init = type('obj', (object,), {
            'pos': pos_init_scaled,
            'quat': quat_init_xyzw
        })
        pose_curr = type('obj', (object,), {
            'pos': pos_curr_scaled,
            'quat': quat_curr_xyzw
        })

        r_rel, t_scaled = compute_transformation(pose_init, pose_curr)
        transformations.append((
            torch.from_numpy(r_rel).cuda().float(),
            torch.from_numpy(t_scaled).cuda().float()
        ))

    return transformations
