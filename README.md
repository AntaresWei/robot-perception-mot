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

### 环境配置

Ubuntu: 22.04
CUDA: 11.8
GPU: NVIDIA GeForce RTX 4090

```
conda create -n robot-perception-mot python=3.10 -y

conda activate robot-perception-mot

pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118

pip install \
    numpy==1.26.4 \
    scipy==1.11.4 \
    opencv-python \
    open3d \
    pyyaml \
    tqdm \
    matplotlib \
    scikit-learn


pip install \
    filterpy \
    lap \
    motmetrics

pip install pytest
```

## Verification

When pytest is available in the configured project environment, run:

```bash
pytest
```

## KITTI Tracking Data

`KittiTrackingDataset` reads KITTI Tracking images, LiDAR, calibration, and
`label_02` annotations. It supports both the conventional `training/` layout
and the split `data_tracking_*` download layout found under this repository's
`data/` directory. The default dataset configuration targets the local data:

```python
from robot_perception.dataset import KittiTrackingDataset

dataset = KittiTrackingDataset("data", sequences=["0000"])
frame = dataset[0]
```

`frame.image` is OpenCV BGR, `frame.lidar_points` is `float32` with shape
`(N, 4)`, and annotations retain KITTI Tracking IDs and camera-frame boxes.

## Status

No performance metrics are available. Experiments and reported metrics must be
added only after real KITTI runs.
