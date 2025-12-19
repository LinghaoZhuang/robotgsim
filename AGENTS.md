# Repository Guidelines

## Project Structure & Module Organization

- `SplatSim/`: primary Python project (Gaussian Splatting + PyBullet)
  - `SplatSim/splatsim/`: library code (`robots/`, `agents/`, `utils/`)
  - `SplatSim/scripts/`: runnable entrypoints (server/client, pipelines)
  - `SplatSim/configs/`: YAML configuration (robot models, paths, transforms)
  - `SplatSim/submodules/`: git submodules (install per `SplatSim/README.md`)
  - `SplatSim/to_delete/`: archived experiments (avoid new changes here)
- `RobotSim/`: Genesis-based simulation with Gaussian rendering
  - `RobotSim/Gaussians/`: rendering + geometry utilities
  - `RobotSim/tasks/`: runnable manipulation tasks
  - `RobotSim/assets/`, `RobotSim/exports/`: meshes/point clouds and outputs
- `CLAUDE.md`: high-level architecture notes and common workflows.

## Build, Test, and Development Commands

SplatSim setup (typical):
```bash
conda create -n splatsim python=3.12
conda activate splatsim
cd SplatSim
git submodule update --init --recursive
# Install PyTorch per pytorch.org (example CUDA 12.6):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e submodules/gaussian-splatting-wrapper -e submodules/gello_software
pip install -r requirements.txt
pip install -e .
```

Run SplatSim server + client (two terminals):
```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name robot_iphone
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

Run a RobotSim task (requires `genesis` installed and assets present):
```bash
python RobotSim/tasks/stack_cubes.py
```

## Coding Style & Naming Conventions

- Python: 4-space indentation, PEP 8 naming (`snake_case` funcs/vars, `CapWords` classes).
- Prefer explicit types for new/edited APIs (type hints and small docstrings on non-trivial functions).
- Keep environment-specific paths out of code; put them in `SplatSim/configs/*.yaml`.
- Avoid committing generated files: `__pycache__/`, `*.pyc`, `.DS_Store`, `*.egg-info/`, large binaries under `data/`/`exports/`.

## Testing Guidelines

- No formal test runner is configured; validation is primarily via runnable scripts.
- Smoke-test examples:
  - `python SplatSim/scripts/quick_run.py`
  - `python RobotSim/utils/test_supersplat.py`
- If adding automated tests, use `pytest` and place them under `SplatSim/tests/` or `RobotSim/tests/`.

## Commit & Pull Request Guidelines

- Follow the existing style in `SplatSim` history: short, imperative subjects (optionally `(#issue)`), e.g., `Refactoring and documentation`.
- Keep commits focused; avoid mixing refactors with behavior changes.
- PRs should include: what changed, how to run/verify (commands + config updates), and screenshots/GIFs for rendering/UI changes. Call out any submodule bumps and new large assets.

## Architecture Overview (Quick)

SplatSim typically runs a simulation/rendering server (ZMQ) and one or more clients that send actions and receive images; see `SplatSim/scripts/launch_nodes.py` and `SplatSim/scripts/run_env*.py`.
