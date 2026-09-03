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
11.8, and ROS2 Humble. The legacy PointPillars model uses the separate
`mmdet3d-kitti` environment documented below.

### 主项目环境

Ubuntu 22.04、Python 3.10、PyTorch 2.1.2+cu118 和 ROS2 Humble 用于运行本项目
的 Dataset、Point Cloud、Kalman、Association 和测试模块：

```bash
conda create -n robot-perception-mot python=3.10 -y
conda activate robot-perception-mot
pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 scipy==1.11.4 opencv-python open3d pyyaml \
  tqdm matplotlib scikit-learn filterpy lap motmetrics pytest
```

## PointPillars 环境

官方 `mmdetection3d-1.0.0rc6` PointPillars checkpoint 属于旧版 OpenMMLab
技术栈，不能与主项目的 PyTorch 2.1.2 环境混装。使用独立环境可重复运行，
不会影响 `robot-perception-mot`：

以下步骤已经在本机验证过。不要把这些依赖安装到 `robot-perception-mot`，
因为官方 `mmdetection3d-1.0.0rc6` 依赖旧版 PyTorch/MMCV。

```bash
source /data/weijiaman/miniconda3/etc/profile.d/conda.sh
conda create -n mmdet3d-kitti python=3.8 -y
conda activate mmdet3d-kitti

# 网络较慢时可在执行安装前设置代理，并在完成后 unset。
export HTTP_PROXY=http://192.168.109.107:7890
export HTTPS_PROXY=http://192.168.109.107:7890

pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy==1.20.3 Pillow==9.5.0
pip install mmcv-full==1.6.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install mmdet==2.28.2 mmsegmentation==0.30.0
pip install -v -e ./mmdetection3d-1.0

unset HTTP_PROXY HTTPS_PROXY
```

安装完成后可保存环境快照，便于以后重建：

```bash
conda env export --from-history > /tmp/mmdet3d-kitti-conda-history.yml
pip freeze > /tmp/mmdet3d-kitti-pip-freeze.txt
```

版本必须保持如下组合：

```text
Python 3.8
PyTorch 1.10.0+cu113
MMCV 1.6.0
MMDetection 2.28.2
MMDetection3D 1.0.0rc6
NumPy 1.20.3
Pillow 9.5.0
```

验证版本：

```bash
python -c "import torch,mmcv,mmdet,mmdet3d,numpy,PIL; print(\
torch.__version__, torch.version.cuda, mmcv.__version__, mmdet.__version__,\
mmdet3d.__version__, numpy.__version__, PIL.__version__)"
```

预期版本为 `1.10.0+cu113 1.6.0 2.28.2 1.0.0rc6 1.20.3 9.5.0`。

### PointPillars 文件

将官方配置和 checkpoint 放在仓库约定位置，或通过 `configs/detector.yaml`
修改路径：

```text
mmdetection3d-1.0/configs/pointpillars/
└── hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py

hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth
```

权重文件较大且不应提交到 Git。运行前确认两个文件都存在：

```bash
test -f mmdetection3d-1.0/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py
test -f hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth
```

该 checkpoint 的官方类别顺序是 `Pedestrian, Cyclist, Car`，项目映射为：

```text
model label 0 -> Pedestrian -> project class_id 2
model label 1 -> Cyclist    -> project class_id 3
model label 2 -> Car        -> project class_id 1
```

### 环境验证与单帧推理

先确认驱动和 GPU 可见。建议先选择空闲卡；例如设置
`CUDA_VISIBLE_DEVICES=6` 后，进程内的 `cuda:0` 就是物理 GPU 6：

```bash
nvidia-smi -L
CUDA_VISIBLE_DEVICES=6 python -c \
  "import torch; print(torch.__version__, torch.version.cuda, \
  torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

如果 `torch.cuda.is_available()` 为 `False`，先检查 `nvidia-smi` 和当前
容器/终端是否能访问 NVIDIA 驱动，不要修改模型代码。

验证官方 demo：

```bash
REPO_ROOT=$(pwd)
MMDET3D_ROOT="$REPO_ROOT/mmdetection3d-1.0"
CUDA_VISIBLE_DEVICES=6 python "$MMDET3D_ROOT/demo/pcd_demo.py" \
  "$REPO_ROOT/data/data_tracking_velodyne/training/velodyne/0000/000100.bin" \
  "$MMDET3D_ROOT/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py" \
  "$REPO_ROOT/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth" \
  --device cuda:0 --score-thr 0.3 --out-dir /tmp/pointpillars_demo
```

成功时会生成：

```text
/tmp/pointpillars_demo/000100/000100_pred.obj
/tmp/pointpillars_demo/000100/000100_points.obj
```

项目 adapter 推理：

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/run_detection.py \
  --detector real --sequence 0000 --frame 100
```

已验证的 `0000/100` 结果为 12 条 `Detection3D`：Car 7、Pedestrian 2、
Cyclist 3，score 范围约为 `0.3206-0.8429`。官方原始输出为 13 个 boxes，
阈值 `0.3` 后保留 12 个。项目 adapter 不修改 Tracker，只负责将
`boxes_3d/scores_3d/labels_3d` 转为统一 `Detection3D`。

### 常见问题

- `ModuleNotFoundError: mmdet`：确认已激活 `mmdet3d-kitti`，并从仓库根目录执行；检查 `pip show mmdet`。
- `numba._internal failed`：确认 `numpy==1.20.3`，不要升级到主项目的 NumPy 版本。
- `CUDA out of memory`：先用 `nvidia-smi` 选择空闲 GPU，再设置 `CUDA_VISIBLE_DEVICES`；不要直接终止其他任务。
- 不要在 `robot-perception-mot` 中安装 `mmcv-full 1.6.0`；它面向 PyTorch 1.10/cu113，和主环境不兼容。
- `CUDA version mismatches`：不要用系统 CUDA 12.x 编译旧版 MMCV；优先使用上面的 `cu113/torch1.10` 预编译 wheel。

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

## Status

PointPillars `0000/100` 单帧推理已在 GPU 6 验证。MOT 指标仍需在后续 Tracker
流水线完成后，通过真实 KITTI 序列运行获得；禁止使用 GT 结果代替 detector
性能指标。
