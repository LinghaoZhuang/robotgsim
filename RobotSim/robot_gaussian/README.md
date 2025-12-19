# Robot Gaussian Module

Pure Gaussian Splatting rendering for articulated robots in RobotSim.

## Overview

This module implements FK-based Gaussian transformation, enabling pure 3DGS rendering that replaces the original background-GS + simulator-foreground overlay approach.

**Key features:**
- KNN segmentation to associate Gaussians with robot joints
- Forward kinematics (FK) to compute joint transformations
- Spherical harmonics (SH) rotation using Wigner-D matrices
- Dual-view stereo rendering support

## Quick Start

### 1. Install Dependencies

```bash
pip install e3nn>=0.5.0 scikit-learn>=1.0 einops>=0.6.0
```

### 2. Run Segmentation (one-time setup)

```bash
python scripts/segment_robot.py
```

This generates `data/labels/so100_labels.npy` containing link assignments for each Gaussian.

**Optional: Visualize segmentation**
```bash
python scripts/visualize_segmentation.py
```

### 3. Run Tasks with Pure GS Rendering

```python
from tasks.pick_banana import PickBanana

collector = PickBanana(
    case=0,
    use_gs=True,
    use_robot_gs=True,  # Enable pure GS rendering
    data_augmentation=False,
    single_view=False
)

collector.run(num_steps=1)
```

## Module Structure

```
robot_gaussian/
├── __init__.py                 # Package exports
├── forward_kinematics.py       # FK calculation
├── sh_rotation.py              # SH rotation (Wigner-D)
├── gaussian_transform.py       # Gaussian transformation
├── segmentation.py             # KNN segmentation
└── robot_gaussian_model.py     # Main model class
```

## Technical Details

### Coordinate Systems

```
Genesis World (robot physics)
├── robot.ply: ~0.4m range
├── link.get_vverts(): world coordinates
└── FK transforms: computed in world

COLMAP/NeRF (3DGS background)
├── left-transform2.ply: ~tens of meters
├── right-transform.ply: ~tens of meters
└── Camera viewmatrix: adjusted to align with world

Key transform:
- world_to_splat = inverse(supersplat_transform)
```

### Quaternion Conventions

- **Genesis link.get_quat()**: (w,x,y,z) - verified
- **PLY file (rot_0~3)**: (w,x,y,z)
- **e3nn**: (w,x,y,z)
- **FK calculation**: (x,y,z,w) - converted with `np.roll(..., -1)`

### Spherical Harmonics

- **Layout**: (N, 16, 3) = [DC(1), rest(15)]
- **Rotation**: Only rotate rest (l=1,2,3), DC unchanged
- **Method**: Wigner-D matrices from e3nn

## Verification

### Step 1: Verify Alignment
Check that robot.ply aligns with Genesis rendering at initial pose.

### Step 2: Verify Segmentation
Run `visualize_segmentation.py` and check:
- Each link forms continuous color block
- No scattered isolated points
- Clear boundaries between links

### Step 3: Verify Dynamic Rendering
Run a short trajectory and check:
- Smooth robot motion, no flickering
- No gaps at joint connections
- Proper alignment during manipulation

## Troubleshooting

**Robot appears offset or misaligned:**
- Check world_to_splat transformation matrix
- Verify supersplat_transform parameters match background PLY generation

**Joints torn or disconnected:**
- Check FK calculation and quaternion format
- Verify relative transformation formula

**SH colors incorrect:**
- Ensure DC/rest separation is correct
- Verify Wigner-D calculation in transform_shs

**Flickering in video:**
- Check backup/restore logic in RobotGaussianModel.update()
- Ensure Gaussians aren't being reallocated every frame
