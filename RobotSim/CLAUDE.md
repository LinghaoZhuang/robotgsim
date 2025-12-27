# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RobotSim is a robotic manipulation data collection framework combining Genesis physics simulation with Gaussian Splatting (3DGS) for photorealistic rendering. It generates training data for imitation learning by simulating SO-100 6-DOF robot arm manipulation tasks.

## Build & Run Commands

### Environment Setup

```bash
# Install core dependencies
pip install genesis torch torchvision
pip install diff-gaussian-rasterization  # CUDA Gaussian rasterizer
pip install e3nn scikit-learn einops plyfile trimesh open3d pyntcloud natsort
pip install opencv-python pillow scipy
```

### Running Tasks

```bash
# Basic task execution (pick and place banana)
python tasks/pick_banana.py

# With command-line options
python tasks/pick_banana.py --start 0 --num_steps 100 --use_gs --data_augmentation --save_dir collected_data

# Pure Gaussian Splatting rendering (experimental)
python tasks/pick_banana.py --use_robot_gs

# Available tasks: pick_banana.py, close_box.py, stack_cubes.py, close_drawer.py,
#                  move_bottle.py, pick_toy.py, stand_bottle.py, wiping.py
```

### Robot Gaussian Segmentation (one-time setup for pure GS mode)

```bash
# Generate link segmentation labels
python scripts/segment_robot.py

# Visualize segmentation results
python scripts/visualize_segmentation.py
```

## Architecture

```
DataCollector (tasks/base_task.py)
├── Genesis Scene (gs.Scene)
│   ├── Robot arm (SO-100 via MJCF)
│   ├── Task objects (mesh/URDF entities)
│   └── Cameras (single or stereo)
├── Gaussian Renderer (Gaussians/render.py)
│   ├── Background PLY (scene without robot)
│   ├── Robot Gaussian Model (pure GS mode)
│   └── Object Gaussians (task objects)
└── Data Logging
    ├── Composited images (GS background + simulator foreground)
    ├── actions.npy, qpos.npy
    └── Video generation
```

### Two Rendering Modes

1. **Overlay Mode** (default, `use_robot_gs=False`): Renders robot from Genesis simulator, composites onto 3DGS background using color keying

2. **Pure GS Mode** (`use_robot_gs=True`): Renders entire scene (robot + objects + background) using Gaussian Splatting with FK-based robot Gaussian transformation

### Task Implementation Pattern

New tasks extend `DataCollector` and override:
- `init_3d_scene()`: Define scene objects, robot pose, cameras
- `get_data()`: Implement task-specific manipulation trajectory
- `reset()`: Randomize object positions for data augmentation
- `_init_object_gs()` (optional): Setup object Gaussians for pure GS mode

### Robot Gaussian Module (`robot_gaussian/`)

Handles FK-based Gaussian transformation for pure GS rendering:
- `segmentation.py`: KNN segmentation to assign Gaussians to robot links
- `forward_kinematics.py`: FK calculation from joint states
- `gaussian_transform.py`: Apply FK transforms to Gaussian positions
- `sh_rotation.py`: Spherical harmonics rotation using Wigner-D matrices
- `robot_gaussian_model.py`: Main model coordinating all components

## Key Data Structures

**Robot DOF mapping**:
- Motors: indices 0-4 (5 arm joints)
- Gripper: index 5

**GaussianDataCUDA** (Gaussians/render.py):
- `xyz`, `rot` (wxyz quaternion), `scale`, `opacity`, `sh` (spherical harmonics)

**Raster settings**: Contains viewmatrix, projmatrix, FOV, resolution for 3DGS rendering

## Coordinate Systems

```
Genesis World (robot physics, ~0.4m range)
├── Robot arm: euler=(0, 0, 90) at origin
├── link.get_vverts(): world coordinates
└── FK transforms: computed in world

COLMAP/NeRF (3DGS PLY files, ~tens of meters)
├── Background PLY: supersplat-transformed
├── Robot PLY: ICP-aligned to Genesis*0.8
└── Camera viewmatrix: aligned via transforms.json
```

**Key transforms** (see `base_task.py` lines 1481-1515):
- `transform_matrix3()`: Build supersplat transform (translation, rotation, scale around center)
- `gl_to_cv()` / `cv_to_gl()`: Convert between OpenGL and OpenCV conventions

## Important Paths

- `assets/so100/urdf/`: Robot URDF/MJCF definitions
- `exports/`: Pre-trained Gaussian splat `.ply` files
- `exports/mult-view-scene/`: Stereo camera scene PLYs
- `collected_data/`: Output directory for trajectories
- `data/labels/`: Segmentation labels and genesis_center
- `outputs/*/splatfacto/*/dataparser_transforms.json`: Nerfstudio camera transforms

## Quaternion Conventions

- Genesis `link.get_quat()`: (w, x, y, z)
- PLY file (`rot_0~3`): (w, x, y, z)
- e3nn: (w, x, y, z)
- scipy Rotation: (x, y, z, w) - requires conversion with `np.roll(..., -1)`
