# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RobotSim is a robotic manipulation data collection framework that combines:
- **Genesis physics simulator** for realistic robot control
- **Gaussian Splatting (3DGS)** for photorealistic scene rendering
- **Task-specific data collectors** for generating training datasets

The system simulates a SO-100 6-DOF robot arm performing manipulation tasks, renders observations using pre-trained Gaussian splats, and saves action/state data for imitation learning.

## Architecture

### Core Components

```
DataCollector (tasks/base_task.py)
├── Genesis physics simulation (gs.Scene, gs.Entity)
├── Robot control (SO-100 arm with 5 motors + 1 gripper DOF)
├── Camera system (single-view or stereo left/right)
├── Gaussian rendering (Renderer class)
└── Data logging (actions, qpos, videos)

Task Subclasses (tasks/*.py)
└── Override init_3d_scene() and get_data() for task-specific behavior
```

### Key Files

- `tasks/base_task.py`: Core `DataCollector` class - handles physics setup, rendering, camera transforms, observation compositing, and data saving
- `Gaussians/render.py`: `Renderer` class wrapping diff_gaussian_rasterization for 3DGS rendering
- `Gaussians/util_gau.py`: `GaussianData` dataclass and PLY file loading (`load_ply`)
- `utils/utils.py`: Helper functions including `RGB2SH`, `SH2RGB`, video creation utilities

### Task Implementation Pattern

New tasks extend `DataCollector` and override:
1. `init_3d_scene()` - Define scene objects, robot initial pose, camera setup
2. `get_data()` - Implement task-specific manipulation trajectory

Example tasks: `close_box.py`, `pick_banana.py`, `stack_cubes.py`, etc.

### Rendering Pipeline

1. Genesis physics step produces robot/object state
2. Camera transforms convert Genesis coordinates to OpenGL/NeRF Studio frame
3. Gaussian rasterizer renders scene background from pre-trained `.ply` files
4. Robot mask from physics render composited onto 3DGS background
5. Final observation saved with action data

## Key Data Structures

**GaussianDataCUDA** (Gaussians/render.py):
- `xyz`, `rot` (quaternion), `scale`, `opacity`, `sh` (spherical harmonics)

**Robot DOF indices**:
- Motors: indices 0-4 (5 arm joints)
- Gripper: index 5

**Camera raster settings**: Contains viewmatrix, projmatrix, FOV, resolution for 3DGS rendering

## Dependencies

Core packages (no requirements.txt exists):
- `genesis` - Physics simulation
- `torch` - Deep learning, CUDA tensors
- `diff_gaussian_rasterization` - CUDA Gaussian rasterizer
- `opencv-python` (cv2), `PIL`, `numpy`, `scipy`
- `plyfile`, `trimesh`, `open3d`, `pyntcloud`

## Running Tasks

Tasks are instantiated and run directly:
```python
from tasks.close_box import CloseBox

collector = CloseBox(
    task='close_box_demo',
    data_augmentation=True,
    use_gs=True,  # Use Gaussian Splatting rendering
    single_view=False  # Stereo views
)
collector.run()  # or collector.get_data()
```

## Important Paths

- `assets/so100/urdf/` - Robot URDF/MJCF definitions
- `exports/` - Pre-trained Gaussian splat `.ply` files
- `collected_data/` - Output directory for collected trajectories
