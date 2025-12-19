# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains two related robotics simulation projects:

### SplatSim
Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting. Combines PyBullet physics simulation with Gaussian Splatting (3DGS) for photorealistic rendering. Primarily uses UR5 robot with Robotiq gripper.

### RobotSim
Genesis-based physics simulation with Gaussian Splatting rendering for SO-100 6-DOF robot arm manipulation tasks.

## Project Structure

```
SplatSim/
├── splatsim/           # Core library
│   ├── robots/         # Robot servers and simulation classes
│   ├── agents/         # Teleoperation and policy agents
│   └── utils/          # Rendering and transform utilities
├── scripts/            # Entry points (launch_nodes.py, run_env*.py)
├── configs/            # YAML configs (object transforms, paths)
└── submodules/         # Git submodules (gaussian-splatting, gello, pybullet)

RobotSim/
├── Gaussians/          # 3DGS rendering (render.py, util_gau.py)
├── tasks/              # Task implementations extending DataCollector
├── assets/             # Robot URDFs and meshes
└── exports/            # Pre-trained Gaussian splat .ply files
```

## Build & Run Commands

### SplatSim Setup
```bash
conda create -n splatsim python=3.12 && conda activate splatsim
cd SplatSim
git submodule update --init --recursive
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install submodules/gaussian-splatting-wrapper/gaussian_splatting/submodules/diff-gaussian-rasterization \
    submodules/pybullet-URDF-models submodules/pybullet-playground-wrapper submodules/ghalton
pip install -e submodules/gaussian-splatting-wrapper -e submodules/gello_software
pip install -r submodules/gello_software/requirements.txt
pip install -e submodules/gello_software/third_party/DynamixelSDK/python
pip install submodules/simple-knn/ --no-build-isolation
pip install -r requirements.txt && pip install -e .
```

### Running SplatSim (Two Terminals)
```bash
# Terminal 1: Launch simulation server
python scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name robot_iphone

# Terminal 2: Run agent client
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

### Robot Calibration Pipeline
```bash
python scripts/articulated_robot_pipeline.py --robot_name your_robot_name
```

### RobotSim Tasks
```bash
python RobotSim/tasks/stack_cubes.py
```

## Architecture

### SplatSim: Server-Client Model (ZMQ)

**Server** (`scripts/launch_nodes.py`):
- Instantiates robot server from `splatsim/robots/sim_robot_pybullet_*.py`
- Manages PyBullet physics simulation
- Renders Gaussian splats with robot overlay
- Serves joint states and rendered images via ZMQ

**Client** (`scripts/run_env_sim.py`):
- Connects agent from `splatsim/agents/`
- Sends actions (joint commands) to server
- Receives observations (images, states)

**Key Robot Server Classes** (`splatsim/robots/`):
- `sim_robot_pybullet_base.py`: Base class with PyBullet setup and 3DGS rendering
- `sim_robot_pybullet_object_on_plate.py`: Template for object manipulation tasks
- `sim_robot_pybullet_splat.py`: Full splat rendering pipeline

**Agents** (`splatsim/agents/`):
- `gello_agent.py`: GELLO teleoperation hardware
- `replay_trajectory_agent.py`: Replay recorded demos
- `policy_agent.py`: Run trained policies

### RobotSim: DataCollector Pattern

New tasks extend `DataCollector` (tasks/base_task.py) and override:
1. `init_3d_scene()` - Scene objects, robot pose, cameras
2. `get_data()` - Task-specific manipulation trajectory

## Configuration

### SplatSim Configs (`SplatSim/configs/`)

**objects.yaml**: Robot/object definitions with:
- `transformation.matrix`: 4x4 splat-to-sim alignment transform
- `aabb.bounding_box`: Axis-aligned bounding box for segmentation
- `joint_states`: Robot joint angles when splat was captured
- `source_path`/`model_path`: Colmap and trained splat paths
- `ply_path`: Object Gaussian splat file

**folder_configs.yaml**: Trajectory storage path

### Adding New Robots/Objects
1. Train Gaussian splat (colmap + gaussian-splatting train)
2. Add entry to `configs/object_configs/objects.yaml`
3. Use CloudCompare to align splat to sim, copy transform matrix to config
4. Run `articulated_robot_pipeline.py` to verify alignment

## Key Dependencies

- PyTorch with CUDA
- diff-gaussian-rasterization (CUDA Gaussian rasterizer)
- pybullet (physics simulation)
- genesis (RobotSim physics)
- open3d, plyfile, trimesh (point cloud processing)
- gello_software (teleoperation hardware)

## Coding Conventions

- Python 3.12+, PEP 8 style
- Environment-specific paths in YAML configs, not hardcoded
- Robot servers use `SERVE_MODES` enum for demo generation vs interactive mode
