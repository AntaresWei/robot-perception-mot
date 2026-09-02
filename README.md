# robot-perception-mot

Camera-LiDAR 3D perception and multi-object tracking for the KITTI Tracking
dataset.

## Current Stage

The repository now contains the KITTI loader, calibration/projection utilities,
point-cloud preprocessing, Kalman/association primitives, and detector
adapters. `GTDetector` and `NoisyGTDetector` are available for tracker
validation. A real PointPillars adapter is provided for MMDetection3D when its
compatible runtime and checkpoint are supplied.

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

## Detection Adapters

`GTDetector` and `NoisyGTDetector` convert KITTI `label_02` annotations into
the common `Detection3D` contract. `MMDet3DPointPillarsAdapter` converts
MMDetection3D `boxes_3d`, `scores_3d`, and `labels_3d` into the same contract.
The GT adapters are controlled inputs for tracker validation. **GT-based
tracking experiments validate only the tracking implementation and must not be
reported as detector performance.**

The current `robot-perception-mot` environment does not include MMDetection3D,
MMEngine, MMCV, or a PointPillars checkpoint, so real-model inference remains
pending those external prerequisites.

When the official MMDetection3D PointPillars config and checkpoint are
available, set `detector.real.config` and `detector.real.checkpoint` in
`configs/detector.yaml`, then run:

```bash
python scripts/run_detection.py --detector real --sequence 0000 --frame 100
```

## Status

No performance metrics are available. Experiments and reported metrics must be
added only after real KITTI runs.
