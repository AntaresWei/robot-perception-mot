# robot-perception-mot 项目规格与开发规范

> 版本：v1.0  
> 项目目标：在 1 周内完成一个可运行、可评估、可展示、可面试讲解的机器人感知项目，并用于机器人感知算法工程师岗位投递。  
> 开发方式：Codex VSCode 插件辅助编程；人负责方案决策、验证与最终代码审查。  
> 数据集：KITTI Tracking Dataset。  
> 主环境：Ubuntu 22.04 / Python 3.10 / PyTorch 2.1.2+cu118 / CUDA 11.8 / ROS2 Humble / RTX 4090 24GB。

---

## 1. 项目定位

### 1.1 项目名称

`robot-perception-mot`

### 1.2 项目一句话描述

构建一个面向移动机器人场景的 **Camera-LiDAR 3D 感知与多目标跟踪系统**，覆盖点云预处理、地面分割、3D 检测结果解析、跨传感器坐标变换、Kalman Filter 状态估计、Mahalanobis Gating、Hungarian 数据关联、Track 生命周期管理、MOT 评估和 ROS2 可视化。

### 1.3 项目核心目标

项目不是为了复现某篇论文，也不是为了训练一个 SOTA 3D Detector。

核心目标是证明以下工程能力：

1. 能解析机器人传感器数据。
2. 理解 Camera-LiDAR 坐标系与标定。
3. 能完成点云预处理和地面提取。
4. 能接入已有 3D Detection 结果。
5. 能独立实现 Kalman Filter。
6. 能独立实现数据关联。
7. 能完成多目标 Track 生命周期管理。
8. 能进行 Camera-LiDAR 跨模态结果验证。
9. 能将算法模块 ROS2 化。
10. 能进行单元测试、集成测试和指标评估。

---

# 2. 与目标岗位 JD 的对应关系

| JD 要求 | 本项目对应模块 |
|---|---|
| 相机、激光雷达传感器 | KITTI Camera + Velodyne LiDAR |
| 多传感器联合标定 | KITTI Calibration、LiDAR→Camera Projection |
| 点云预处理 | Voxel Downsampling、ROI、Outlier Filtering |
| 地面提取 | RANSAC Ground Plane |
| 障碍物检测 | 3D Detector Adapter |
| 障碍物跟踪 | 3D Multi-Object Tracker |
| 占用栅格 | Phase 2 可选扩展，不影响 MVP |
| 贝叶斯估计 | Kalman Filter 概率状态估计 |
| 卡尔曼滤波 | 自研 Constant Velocity KF |
| 粒子滤波 | 理论储备，本项目不作为 MVP |
| Python/C++ | Python 主实现；ROS2 接口设计保持可 C++ 化 |
| PCL/OpenCV | Open3D/OpenCV；后续可增加 PCL/C++ adapter |
| 单元测试 | pytest |
| 集成测试 | pipeline integration test |
| 仿真测试 | KITTI sequence replay |
| 性能指标 | Detection + MOT metrics |
| 机器人感知系统 | ROS2 Node + TF + RViz2 |

---

# 3. 时间约束与开发原则

## 3.1 硬约束

项目必须在约 7 天内形成第一版可投递成果。

因此：

- 不训练自己的 CenterPoint。
- 不完整复现 BEVFusion。
- 不完整复现 LIO-SAM。
- 不完整部署 Autoware。
- 不为了追求 SOTA 引入大量复杂依赖。
- 不允许因为某个研究仓库环境问题阻塞核心 Tracking 开发。
- 不允许先花大量时间写 README 而没有可运行代码。

## 3.2 核心原则

### 原则 A：先跑通，再优化

MVP pipeline：

```text
KITTI
  ↓
Dataset Loader
  ↓
Point Cloud Preprocessing
  ↓
3D Detection Input
  ↓
Kalman Filter
  ↓
Mahalanobis Gating
  ↓
Hungarian Association
  ↓
Track Management
  ↓
Evaluation
  ↓
Visualization
```

### 原则 B：模块解耦

Detector、Tracker、Dataset、Fusion、ROS2 必须通过明确的数据结构通信。

禁止 Tracker 直接依赖某一个具体 Detector 的内部实现。

### 原则 C：真实实验优先

README 中的所有指标必须来自实际运行结果。

禁止编造：

- Accuracy
- MOTA
- IDF1
- IDS
- FPS
- latency
- GPU memory
- precision
- recall

### 原则 D：每个阶段都必须可验证

每完成一个模块，都必须有：

- 单元测试，或
- 可执行 demo，或
- 可视化结果。

---

# 4. 技术栈

## 4.1 基础环境

```text
OS: Ubuntu 22.04
Python: 3.10
PyTorch: 2.1.2+cu118
CUDA: 11.8
GPU: NVIDIA RTX 4090 24GB
ROS2: Humble
```

## 4.2 Python 依赖

核心：

```text
numpy
scipy
opencv-python
open3d
pyyaml
tqdm
matplotlib
```

Tracking：

```text
filterpy
lap
motmetrics
```

Testing：

```text
pytest
```

原则：

- 依赖版本记录在 `requirements.txt`。
- 环境完整配置记录在 `environment.yml`。
- 不随意升级核心依赖。
- 若增加依赖，必须说明原因。

---

# 5. 数据集

## 5.1 主数据集

KITTI Tracking Dataset。

只使用：

```text
image_02
velodyne
calib
label_02
```

第一阶段只开发：

```text
sequence 0000
sequence 0001
sequence 0002
```

不要一开始处理全部训练序列。

## 5.2 数据含义

每个 frame 需要统一转换成：

```python
FrameData(
    sequence_id: str,
    frame_id: int,
    image: Optional[np.ndarray],
    lidar_points: np.ndarray,
    calibration: Calibration,
    annotations: Optional[list],
    timestamp: Optional[float],
)
```

LiDAR 点云默认：

```text
N x 4
[x, y, z, intensity]
```

## 5.3 坐标系

必须明确：

```text
Velodyne / LiDAR frame
Camera frame
Image pixel frame
World / ego frame（如需要）
```

禁止在代码中隐式假设坐标系。

所有坐标变换必须通过明确函数完成。

---

# 6. 推荐项目目录

```text
robot-perception-mot/
│
├── README.md
├── PROJECT_SPEC.md
├── DEVELOPMENT.md
├── CHANGELOG.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/
│   ├── dataset.yaml
│   ├── tracker.yaml
│   └── pipeline.yaml
│
├── data/
│   └── README.md
│
├── src/
│   └── robot_perception/
│       ├── __init__.py
│       │
│       ├── common/
│       │   ├── types.py
│       │   ├── geometry.py
│       │   └── logger.py
│       │
│       ├── dataset/
│       │   ├── kitti.py
│       │   └── calibration.py
│       │
│       ├── pointcloud/
│       │   ├── preprocess.py
│       │   └── ground.py
│       │
│       ├── detection/
│       │   ├── base.py
│       │   └── adapter.py
│       │
│       ├── tracking/
│       │   ├── kalman.py
│       │   ├── gating.py
│       │   ├── association.py
│       │   ├── track.py
│       │   └── tracker.py
│       │
│       ├── fusion/
│       │   └── camera_lidar.py
│       │
│       ├── evaluation/
│       │   ├── detection.py
│       │   └── tracking.py
│       │
│       └── visualization/
│           ├── camera.py
│           ├── pointcloud.py
│           └── rviz.py
│
├── scripts/
│   ├── inspect_kitti.py
│   ├── project_lidar_to_image.py
│   ├── run_detection.py
│   ├── run_tracking.py
│   └── evaluate.py
│
├── tests/
│   ├── test_calibration.py
│   ├── test_kalman.py
│   ├── test_gating.py
│   ├── test_association.py
│   ├── test_track.py
│   └── test_tracker.py
│
├── ros2_ws/
│   └── src/
│       └── perception_tracker/
│
├── outputs/
│   ├── visualizations/
│   ├── metrics/
│   └── logs/
│
└── docs/
    ├── architecture.md
    ├── algorithms.md
    └── experiments.md
```

---

# 7. 核心数据结构

所有模块优先使用明确的数据类。

## 7.1 Detection3D

```python
@dataclass
class Detection3D:
    position: np.ndarray      # [x, y, z]
    dimensions: np.ndarray    # [l, w, h]
    yaw: float
    score: float
    class_id: int
    velocity: Optional[np.ndarray] = None
```

## 7.2 TrackState

```python
@dataclass
class TrackState:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    age: int
    hits: int
    time_since_update: int
    status: str
```

## 7.3 推荐状态向量

第一版：

```text
x = [px, py, pz, vx, vy, vz]^T
```

采用 Constant Velocity Model。

第一版 measurement：

```text
z = [px, py, pz]^T
```

不要一开始把尺寸、yaw、加速度全部塞入状态。

---

# 8. Point Cloud Pipeline

## 8.1 处理顺序

```text
Raw PointCloud
    ↓
NaN / Inf filtering
    ↓
ROI filtering
    ↓
Voxel Downsampling
    ↓
Outlier Filtering
    ↓
Ground Removal
    ↓
Non-ground PointCloud
```

## 8.2 Ground Removal

MVP：

**RANSAC Plane Segmentation**

输出：

```text
ground_points
non_ground_points
plane_model
```

必须记录：

- distance threshold
- max iterations
- 最终 plane coefficients
- ground point ratio

## 8.3 不做的事情

MVP 阶段不实现：

- CSF
- Patchwork
- 深度学习 Ground Segmentation
- 复杂动态地面模型

除非核心 pipeline 已经稳定。

---

# 9. Camera-LiDAR Calibration

## 9.1 核心变换

必须实现：

```text
LiDAR
 ↓
Tr_velo_to_cam
 ↓
Camera 3D
 ↓
R0_rect
 ↓
Rectified Camera
 ↓
P2
 ↓
Image Pixel
```

## 9.2 必须提供函数

```python
lidar_to_camera(points)
camera_to_image(points)
lidar_to_image(points)
```

## 9.3 验收

随机选择 KITTI frame：

1. 加载 LiDAR。
2. 投影到 Camera。
3. 过滤相机视野外点。
4. 将点绘制到图像。
5. 生成保存图片。

验收标准：

- 点云投影位置合理。
- 点不会整体偏移。
- 深度变化与图像位置基本一致。

---

# 10. 3D Detection

## 10.1 设计原则

Detector 必须采用 Adapter Pattern。

```python
class BaseDetector(ABC):
    @abstractmethod
    def predict(self, frame: FrameData) -> list[Detection3D]:
        ...
```

这样 Tracker 不依赖具体模型。

## 10.2 一周 MVP

优先使用：

- 已有公开 3D detection inference
- 或 KITTI GT 作为临时 detection 输入
- 或 GT + controlled noise 作为 Tracking baseline

不训练新模型。

## 10.3 Tracking 开发允许使用 GT

为了先验证 Tracker，可以：

```text
GT 3D boxes
   ↓
Add controlled noise
   ↓
Tracker
```

但是 README 必须明确写：

> GT-based tracking experiment is used only for validating the tracking implementation and is not reported as detector performance.

---

# 11. Kalman Filter

## 11.1 模型

Constant Velocity Model。

状态：

```text
x = [px, py, pz, vx, vy, vz]
```

Prediction：

```text
x_k|k-1 = F x_k-1|k-1
P_k|k-1 = F P F^T + Q
```

Measurement：

```text
z = Hx + v
```

Update：

```text
y = z - Hx
S = HPH^T + R
K = PH^T S^-1
x = x + Ky
P = (I - KH)P
```

## 11.2 工程要求

KF 模块必须：

- 不依赖 Detector。
- 支持可配置 `Q`。
- 支持可配置 `R`。
- 支持 variable `dt`。
- 输入输出 shape 有断言。
- 数值异常时给出明确错误。
- 提供单元测试。

---

# 12. Mahalanobis Gating

不要直接用欧氏距离作为唯一关联标准。

定义 innovation：

```text
y = z - Hx
```

innovation covariance：

```text
S = HPH^T + R
```

Mahalanobis distance：

```text
d² = y^T S^-1 y
```

根据 configurable threshold 判断：

```text
d² <= threshold → valid candidate
d² > threshold  → invalid candidate
```

## 12.1 工程要求

必须支持：

```yaml
gating:
  enabled: true
  threshold: 9.21
```

阈值不能散落在源码。

---

# 13. Hungarian Data Association

流程：

```text
Tracks
  +
Detections
  ↓
Prediction
  ↓
Cost Matrix
  ↓
Mahalanobis Gating
  ↓
Invalid pair = large cost
  ↓
Hungarian Assignment
  ↓
Matched
Unmatched Tracks
Unmatched Detections
```

必须处理：

1. 0 track。
2. 0 detection。
3. track 数 ≠ detection 数。
4. 所有 pair 都被 gating。
5. 新目标出现。
6. 目标消失。

---

# 14. Track Lifecycle

推荐状态：

```text
Tentative
   ↓ min_hits
Confirmed
   ↓ missed
Lost
   ↓ max_age
Deleted
```

配置：

```yaml
tracker:
  min_hits: 3
  max_age: 5
  max_distance: 9.21
```

Track 必须维护：

```text
track_id
age
hits
time_since_update
state
covariance
status
```

ID 分配必须单调递增。

删除 Track 后不得立即复用旧 ID。

---

# 15. Tracker Pipeline

最终：

```text
Frame
 ↓
Detections
 ↓
Predict all Tracks
 ↓
Build Cost Matrix
 ↓
Mahalanobis Gating
 ↓
Hungarian Assignment
 ↓
Update matched Tracks
 ↓
Create unmatched Detection Tracks
 ↓
Age unmatched Tracks
 ↓
Delete expired Tracks
 ↓
Output Confirmed Tracks
```

---

# 16. Camera-LiDAR Fusion

MVP 不做复杂 neural fusion。

实现 Object-level fusion：

```text
LiDAR 3D Detection
       ↓
Project to Camera
       ↓
Camera 2D Detection
       ↓
2D IoU / center distance
       ↓
Cross-modal Association
       ↓
Fused Object
```

如果 Camera Detector 无法及时接入，则第一阶段至少实现：

```text
LiDAR 3D Box
       ↓
Camera Projection
       ↓
Visualization / Consistency Check
```

但 README 必须明确区分：

- Projection
- Cross-modal association
- True sensor fusion

不能把 projection 写成 fusion。

---

# 17. Evaluation

## 17.1 Detection

如使用真实 detector，记录：

```text
Precision
Recall
mAP
```

## 17.2 Tracking

优先记录：

```text
MOTA
IDF1
IDS
FP
FN
```

同时记录：

```text
FPS
latency/frame
GPU memory
```

## 17.3 必须做 baseline

至少比较：

```text
Baseline A:
Nearest Neighbor + no KF

Baseline B:
KF + Euclidean Association

Baseline C:
KF + Mahalanobis + Hungarian
```

这样可以证明算法设计带来的收益。

---

# 18. 实验设计

## Experiment 1：KF 状态估计

输入：

```text
GT position + Gaussian noise
```

比较：

```text
No filtering
vs
Kalman Filter
```

输出：

```text
position RMSE
velocity RMSE
```

## Experiment 2：Association

比较：

```text
Euclidean
vs
Mahalanobis
vs
Mahalanobis + Hungarian
```

记录：

```text
ID switches
MOTA
IDF1
```

## Experiment 3：Track Lifecycle

测试：

```text
目标突然出现
目标短时间消失
目标长时间消失
两个目标交叉
```

验证：

- ID 是否稳定。
- Track 是否正确删除。
- 是否产生大量 ID Switch。

## Experiment 4：Camera-LiDAR

输出：

```text
Camera image
+
Projected LiDAR
+
3D Bounding Box
+
Track ID
```

---

# 19. ROS2

## 19.1 节点

最小节点：

```text
pointcloud_node
detector_node
tracker_node
visualization_node
```

如果开发时间不足，可以先合并：

```text
perception_tracker_node
```

后续拆分。

## 19.2 Topic

推荐：

```text
/sensors/lidar/points
/sensors/camera/image
/perception/detections
/perception/tracks
```

## 19.3 ROS2 消息

优先使用标准消息或明确自定义消息。

必须携带：

```text
header.stamp
frame_id
```

不要忽略 timestamp。

## 19.4 TF

至少明确：

```text
base_link
lidar
camera
```

坐标变换不能通过 magic number 实现。

---

# 20. 测试规范

## 20.1 Unit Test

至少：

```text
test_calibration.py
test_kalman.py
test_gating.py
test_association.py
test_track.py
test_tracker.py
```

## 20.2 必须测试的边界

```text
empty input
single object
multiple objects
missing detection
new detection
large dt
invalid NaN
singular covariance
```

## 20.3 Integration Test

至少：

```text
KITTI frame
 ↓
Dataset Loader
 ↓
Detection
 ↓
Tracker
 ↓
Output
```

必须可以一条命令运行。

---

# 21. 配置管理

所有重要超参数放 YAML。

例如：

```yaml
dataset:
  root: /path/to/kitti/tracking
  sequences:
    - "0000"
    - "0001"
    - "0002"

pointcloud:
  voxel_size: 0.10
  roi:
    x_min: 0.0
    x_max: 70.0
    y_min: -40.0
    y_max: 40.0
    z_min: -3.0
    z_max: 2.0

ground:
  method: ransac
  distance_threshold: 0.15
  ransac_n: 3
  num_iterations: 100

tracker:
  min_hits: 3
  max_age: 5

gating:
  enabled: true
  threshold: 9.21
```

禁止在代码中散落 magic numbers。

---

# 22. Logging

所有运行实验必须记录：

```text
timestamp
git commit
config path
dataset sequence
frame range
metrics
runtime
errors
```

推荐：

```text
outputs/logs/
outputs/metrics/
outputs/visualizations/
```

---

# 23. Git 规范

Commit 必须小而明确。

推荐：

```text
feat: add KITTI tracking dataset loader
feat: implement lidar to camera projection
feat: add point cloud preprocessing
feat: implement kalman filter
feat: add mahalanobis gating
feat: implement hungarian association
feat: add track lifecycle management
feat: add MOT evaluation
feat: integrate ROS2 tracker node
test: add tracker unit tests
docs: add architecture and experiment report
```

禁止：

```text
update
fix
test
aaa
final
final2
final_final
```

---

# 24. Codex 开发规范

Codex 不是“自动写完整项目”的黑盒。

每次任务必须遵循：

```text
1. Read relevant files
2. Understand current architecture
3. State implementation plan
4. Implement smallest change
5. Run tests
6. Inspect output
7. Fix failures
8. Summarize changes
```

## 24.1 Codex 禁止事项

禁止：

- 未阅读现有代码就大规模重写。
- 擅自修改项目架构。
- 擅自升级核心依赖。
- 为了通过测试而修改测试预期。
- 删除失败测试。
- 使用 hard-coded absolute paths。
- 伪造实验结果。
- 未验证代码就声称完成。
- 一次性生成大量互相耦合代码。

## 24.2 每次 Codex Task 必须有验收标准

例如：

```text
Task:
Implement Kalman Filter.

Acceptance:
1. pytest tests/test_kalman.py passes.
2. predict() supports variable dt.
3. update() accepts [px, py, pz].
4. state shape is validated.
5. covariance remains symmetric within tolerance.
```

---

# 25. 一周开发计划

## Day 1

目标：

**Tracking Core**

完成：

```text
KF
Mahalanobis
Hungarian
Track
Tracker
Unit Tests
```

验收：

```bash
pytest tests/
```

通过。

---

## Day 2

目标：

**KITTI Dataset + Point Cloud**

完成：

```text
KITTI Loader
Calibration Parser
PointCloud Preprocessing
Ground Removal
```

验收：

```bash
python scripts/inspect_kitti.py --sequence 0000 --frame 100
```

输出 image / point cloud / calibration / GT。

---

## Day 3

目标：

**3D Detection Adapter**

完成：

```text
BaseDetector
Detector Adapter
GT/noisy GT baseline
真实 detector 接口
```

Tracker 不允许依赖 detector 内部代码。

---

## Day 4

目标：

**Camera-LiDAR Projection + Fusion**

完成：

```text
LiDAR → Camera
3D → 2D
Cross-modal association（若时间允许）
Visualization
```

---

## Day 5

目标：

**ROS2**

完成：

```text
ROS2 package
Node
Topics
TF
RViz
```

至少可以 replay KITTI sequence。

---

## Day 6

目标：

**Evaluation + Benchmark**

完成：

```text
MOTA
IDF1
IDS
FPS
Latency
Baseline comparison
```

---

## Day 7

目标：

**交付**

完成：

```text
README
Architecture diagram
Demo GIF/video
Experiment table
Code cleanup
Unit tests
Git tags
Resume project description
Interview notes
```

禁止 Day 7 添加大型新算法。

---

# 26. MVP 与扩展功能

## MVP：必须完成

```text
[x] KITTI loader
[x] Calibration
[x] LiDAR preprocessing
[x] Ground segmentation
[x] 3D detection interface
[x] Kalman Filter
[x] Mahalanobis gating
[x] Hungarian association
[x] Track lifecycle
[x] MOT evaluation
[x] Camera-LiDAR projection
[x] Unit tests
[x] README
```

## P1：时间允许再做

```text
[ ] Camera detection fusion
[ ] ROS2
[ ] RViz
[ ] Velocity estimation improvement
[ ] Better track visualization
```

## P2：本次不做

```text
[ ] BEVFusion training
[ ] CenterPoint training
[ ] LIO-SAM reproduction
[ ] Particle Filter
[ ] JPDA
[ ] DeepSORT appearance branch
[ ] Full Autoware integration
[ ] Radar fusion
[ ] Neural occupancy
```

---

# 27. 面试必须能解释

项目完成后，必须能够脱离代码解释：

### Tracking

1. 为什么需要 Kalman Filter？
2. Constant Velocity Model 假设是什么？
3. Q 和 R 分别代表什么？
4. covariance 有什么作用？
5. 为什么用 Mahalanobis Distance？
6. Hungarian Algorithm 解决什么问题？
7. Track 如何初始化？
8. Track 什么时候 Confirm？
9. Track 什么时候删除？
10. 遮挡情况下怎么办？

### Sensor Fusion

11. Camera 和 LiDAR 如何标定？
12. 外参矩阵表示什么？
13. 时间同步为什么重要？
14. Camera-LiDAR 如何进行 object-level fusion？
15. projection 和 fusion 有什么区别？

### Point Cloud

16. 为什么做 voxel downsampling？
17. RANSAC 地面分割原理？
18. 地面分割失败的情况？
19. 点云坐标系如何定义？
20. 如何处理 outlier？

### 工程

21. 为什么 Tracker 和 Detector 要解耦？
22. 如何进行单元测试？
23. 如何评估 Tracking？
24. IDF1 和 MOTA 有什么区别？
25. 如果 FPS 不够怎么办？

---

# 28. README 必须包含

```text
1. Project Overview
2. Motivation
3. System Architecture
4. Dataset
5. Environment
6. Installation
7. Dataset Preparation
8. Quick Start
9. Algorithm Details
10. Tracking Design
11. Sensor Fusion
12. ROS2
13. Evaluation
14. Experimental Results
15. Visualization
16. Limitations
17. Future Work
18. References
```

README 首页必须直接看到：

```text
Architecture Figure
Demo GIF
Main Metrics
Environment
Quick Start
```

---

# 29. 最终 Demo

最终 Demo 至少展示：

```text
Camera View
 ├── 3D projected box
 ├── Track ID
 └── class

LiDAR / RViz
 ├── Point Cloud
 ├── 3D Box
 └── Track ID
```

一个连续 sequence 中，同一目标应保持稳定 Track ID。

---

# 30. 质量标准

项目完成不等于“代码能运行”。

最终必须满足：

### Correctness

- 坐标变换正确。
- KF 数学实现正确。
- Association 处理边界情况。
- Tracking ID 稳定。

### Reproducibility

- 新环境可安装。
- 配置可复现。
- 命令可运行。
- 实验结果可重新生成。

### Engineering

- 模块解耦。
- 类型明确。
- 日志完整。
- 测试存在。
- 无硬编码绝对路径。

### Presentation

- README 清晰。
- 有架构图。
- 有 Demo。
- 有 benchmark。
- 能解释设计取舍。

---

# 31. 最终简历项目描述方向

项目完成后，可以根据真实实现情况写：

> **多传感器 3D 感知与多目标跟踪系统**
>
> - 基于 KITTI Camera-LiDAR 数据构建机器人感知 Pipeline，实现点云预处理、Voxel Downsampling、RANSAC 地面分割及传感器坐标变换。
> - 设计 Detector Adapter 接口接入 3D Detection 结果，自研 Constant-Velocity Kalman Filter，实现三维目标位置与速度状态估计。
> - 基于 Mahalanobis Distance 进行运动状态 Gating，并采用 Hungarian Algorithm 完成 Detection-to-Track 数据关联，实现 Track 初始化、确认、丢失和删除等生命周期管理。
> - 实现 Camera-LiDAR 3D→2D 投影及跨模态目标一致性验证，并通过可视化展示目标 Bounding Box 与 Track ID。
> - 使用 pytest 构建 KF、数据关联及 Tracker 单元测试，并基于 MOTA、IDF1、IDS、FPS 等指标评估系统性能。

所有数字指标必须替换为真实实验结果。

---

# 32. 开发完成定义

只有满足以下条件，项目才算“完成”：

```text
1. 一条命令可以加载 KITTI sequence。
2. 可以读取 Camera + LiDAR + Calibration。
3. LiDAR 可以正确投影到 Camera。
4. 点云预处理可运行。
5. Ground Removal 可运行。
6. Tracker 可以处理连续 frames。
7. Track ID 能稳定保持。
8. Unit tests 全部通过。
9. 至少有一个 baseline 对比。
10. 有真实 MOT metrics。
11. 有可视化 Demo。
12. README 可以让别人复现。
13. Git commit 历史清晰。
14. 项目中的每个核心算法都能解释。
```

---

# 33. Codex 第一阶段启动任务

不要一次要求 Codex 编完整项目。

第一条任务只做：

```text
阅读 PROJECT_SPEC.md。

当前只允许完成：
1. 创建项目目录结构。
2. 创建 Python package。
3. 创建基础 dataclass。
4. 创建 configs。
5. 创建 pytest 基础结构。
6. 创建 README skeleton。
7. 不实现 Detector。
8. 不实现 ROS2。
9. 不实现复杂算法。
10. 不安装或修改环境。

完成后：
- 运行 pytest。
- 检查 import。
- 输出目录树。
- 汇报完成项和未完成项。
```

第二条任务再开始：

```text
Implement Kalman Filter according to PROJECT_SPEC.md.
```

第三条：

```text
Implement Mahalanobis gating and Hungarian association.
```

依次推进。

**不要让 Codex 一口气生成整个 repo。**

---

# 34. 项目核心思想

本项目最终要证明的不是：

> “我会调用一个 3D Detection 模型。”

而是：

```text
Sensor Data
    ↓
Coordinate System
    ↓
Point Cloud Processing
    ↓
Detection
    ↓
State Estimation
    ↓
Data Association
    ↓
Tracking
    ↓
Sensor Fusion
    ↓
Evaluation
    ↓
ROS2 Integration
```

这条链才是本项目与机器人感知算法工程师岗位的核心匹配点。
