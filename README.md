# robot-perception-mot

Camera-LiDAR 3D perception and multi-object tracking for the KITTI Tracking
dataset.

## Current Stage

This repository currently contains only the project scaffold: Python package
namespaces, shared dataclasses, YAML configuration, and pytest test structure.
Detection, tracking algorithms, calibration, point cloud processing, ROS2
nodes, evaluation, and visualization are intentionally not implemented yet.

## Planned Layout

```text
configs/                 YAML runtime configuration
src/robot_perception/    Reusable perception package
tests/                   pytest checks
docs/                    Architecture, algorithm, and experiment notes
outputs/                 Generated logs, metrics, and visualizations
```

## Environment

The target environment is Ubuntu 22.04, Python 3.10, PyTorch 2.1.2 with CUDA
11.8, and ROS2 Humble. See `readme.md` for the existing environment setup
notes. Do not treat this scaffold as a runnable perception pipeline.

## Verification

When pytest is available in the configured project environment, run:

```bash
pytest
```

## Status

No performance metrics are available. Experiments and reported metrics must be
added only after real KITTI runs.
