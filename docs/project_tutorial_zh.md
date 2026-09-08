# robot-perception-mot 中文项目教程

这是一份面向初学者的项目教程。建议先通读流程，再打开源码逐个函数对照。教程
描述的是当前仓库真实实现，不把计划中的功能写成已经完成的功能。

## 1. 先建立整体认识

这个项目不是“一个模型输入一张图片然后输出框”，而是一条多传感器时间序列：

```text
KITTI image + LiDAR + calibration + label
        ↓
      FrameData
        ↓
Camera 2D detector       LiDAR 3D detector
YOLO26 ONNX              PointPillars
        ↓                       ↓
Camera2DDetection[]      Detection3D[]
                                ↓
                     Kalman + Gating + Hungarian
                                ↓
                           TrackState[]
                                ↓
              3D box projection + 2D/3D association
                                ↓
                         FusedDetection[]
                                ↓
                    PNG / MP4 / JSON / ROS2 topic
```

最重要的工程边界是：

- Dataset 负责读文件，不负责检测和跟踪；
- Detector 负责把模型输出转换成统一数据类；
- Tracker 只接收 `Detection3D`，不导入 YOLO、PointPillars 或 ROS2；
- Fusion 负责投影、跨模态匹配和融合结果，不修改 Tracker 核心数学；
- ROS2 只是适配层，不把 `rclpy` 塞进算法模块。

## 2. 一个 frame 到底包含什么

`FrameData` 定义在 `src/robot_perception/common/types.py`：

```python
FrameData(
    sequence_id="0000",
    frame_id=100,
    image=image,              # H x W x 3，OpenCV BGR
    lidar_points=points,      # N x 4，[x, y, z, intensity]
    calibration=calibration,
    annotations=annotations,
)
```

KITTI Tracking 的标签字段与普通目标检测不完全一样。3D `location` 是 camera
坐标系下的 bottom-center，尺寸原始顺序是 `h, w, l`。项目内部把尺寸统一成
`[l, w, h]`，并在需要作为 box center 时减去一半高度。这是非常容易造成“框看起来
整体上下偏移”的地方。

数据读取基准是 LiDAR 文件，而不是图片或标签：如果某个图像缺失，Loader 会在读取
该 frame 时报错；如果标签为空，`annotations` 是空列表；如果 `.bin` 不是 4 的倍数，
或者包含 NaN/Inf，会直接拒绝。这样做是为了尽早暴露数据问题，而不是把坏数据传到
后面的矩阵运算中。

## 3. 从哪里开始运行

主项目环境使用 Python 3.10。先从仓库根目录检查一帧：

```bash
conda run -n robot-perception-mot python scripts/inspect_kitti.py \
  --sequence 0000 --frame 100
```

典型输出包含：

```text
Image shape: (375, 1242, 3)
LiDAR shape: (122624, 4)
Number of annotations: 7
```

这里的 7 个标签可能包含 `DontCare`，不能直接当作 7 个可跟踪目标。

## 4. 点云预处理：为什么有多个版本

`preprocess_point_cloud()` 的顺序是：

```text
raw
 ↓ remove_invalid_points
valid
 ↓ crop_roi
roi
 ↓ voxel_downsample
voxel
 ↓ statistical/radius outlier filter
outlier_filtered
 ↓ segment_ground_ransac
ground + non_ground
```

所有参数来自 `configs/pipeline.yaml`。`PointCloudPreprocessResult` 保留各阶段数组，
所以可以回答“到底是哪一步把点数从 122624 降到了 15349”。

一个关键实践判断：**预处理可以存在，不代表必须插入每个 Detector。** 当前官方
PointPillars 预训练模型接收 KITTI 原始 `.bin`，模型内部已经有自己的点云范围和
pillar 编码逻辑。外部先做 ground removal 可能改变训练分布，甚至把目标下半部分删掉。
因此当前项目：

- PointPillars 真实推理使用原始点云；
- `non_ground` 主要用于可视化和独立预处理实验；
- 若要把预处理接到 Detector 前，必须重新做检测精度和运行速度对比，不能凭感觉接入。

## 5. Camera 2D 检测：YOLO26 到统一接口

原始 `yolo26s.pt` 是 COCO 80 类预训练模型，不是 KITTI 三类模型。项目先使用
`scripts/prepare_kitti_yolo.py` 把 KITTI 标签转成 YOLO 格式，再训练三类 head：

```text
YOLO label 0 -> Car
YOLO label 1 -> Pedestrian
YOLO label 2 -> Cyclist
```

`Van`、`Truck` 按当前项目规则归入 `Car`，`DontCare` 被忽略。训练数据按 sequence
切分，避免同一段连续视频的相邻帧同时出现在 train 和 val。

推理时，`Yolo26OnnxCameraDetector` 做四件容易被忽略的事：

1. 对原图做 centered letterbox，而不是简单拉伸；
2. ONNX 前向输出 `[x1, y1, x2, y2, score, class]`；
3. 把 padding 和缩放逆变换回原始图像像素；
4. 按类别执行 NMS，再映射成 `Camera2DDetection`。

因此不能直接把 ONNX 的框坐标当作原图坐标。最常见的错误是忘记去除上下 padding，
导致框整体偏移。

## 6. LiDAR 3D 检测：PointPillars 的职责

PointPillars 来自官方 MMDetection3D，模型输入是 LiDAR `.bin`，输出是：

```text
boxes_3d
scores_3d
labels_3d
```

项目的 `MMDet3DPointPillarsAdapter` 只做输出适配和阈值过滤，不重新实现神经网络：

```text
model label 0 -> Pedestrian -> project class_id 2
model label 1 -> Cyclist    -> project class_id 3
model label 2 -> Car        -> project class_id 1
```

真实模型使用旧版 PyTorch/MMCV，因此与主环境隔离在 `mmdet3d-kitti`。环境问题不要
用“升级所有包”解决；旧版 MMCV 很容易因为 CUDA、NumPy 或 PyTorch 版本不匹配而失效。

## 7. 坐标系：框为什么会错位

KITTI 中至少要区分：

```text
LiDAR frame
Camera frame
Rectified camera frame
Image pixel frame
```

点投影是：

```text
lidar xyz
  --Tr_velo_to_cam--> camera xyz
  --R0_rect-------> rectified camera xyz
  --P2------------> pixel (u, v)
```

代码先把点变成 homogeneous `[x, y, z, 1]`，再做矩阵乘法。投影前必须过滤：

- NaN/Inf；
- camera z <= 0 的后方点；
- 图像边界之外的像素。

检查坐标问题的实践顺序：

1. 用一个 identity calibration 写单元测试；
2. 用一个已知平移矩阵验证坐标变化；
3. 先画 raw LiDAR 投影，不要先画 non-ground；
4. 再检查 box 的 camera/lidar 坐标声明是否正确；
5. 最后才调 IoU 阈值。

很多所谓“融合算法不好”的问题，实际是 camera bottom-center 没有还原成 box center，
或者把 LiDAR box 当成 camera box 旋转。

## 8. Tracker 每一帧做什么

生产 Tracker 是 `MultiObjectTracker`，每帧大致如下：

```text
active Track
    ↓ predict(dt)
predicted state/covariance
    ↓ Mahalanobis gating
valid cost matrix
    ↓ Hungarian
matched pairs
    ├─ matched: Track.update(detection)
    ├─ unmatched track: Lost / Deleted
    └─ unmatched detection: create new Track
```

状态是：

```text
[px, py, pz, vx, vy, vz]
```

`min_hits=3` 的实际含义是：一个新目标至少连续/累计得到足够多次有效更新后才会被
输出为 Confirmed。于是第一帧可能有 12 个检测，但 confirmed track 是 0，这不是 bug，
而是生命周期设计。做单帧 smoke 时可以用 `--min-hits 1`，正式实验应使用配置值。

### 8.1 漏检后能不能找回

可以，但只在 `max_age` 窗口内找回。一次没有检测到目标时，Tracker 不会马上删除
Track，而是：

```text
Confirmed
   ↓ 本帧没有匹配
Lost（继续 predict）
   ↓ 在 max_age 帧内重新出现，并通过 gating
原 Track.update()
   ↓
恢复为 Confirmed，保留原 track_id
```

`time_since_update` 会在 `predict()` 时增加，在 `Track.update()` 时归零。当
`time_since_update >= max_age` 时，Track 变为 `Deleted`。Deleted Track 不会再被
恢复；目标下一次出现会创建新 Track 和新 ID。

未启用 FastReID 时，长时间遮挡后只能重新建轨迹；启用后，能否找回还取决于特征库
有效期、相似度阈值和当前 active ID 排除。能否通过运动恢复短期目标取决于三个因素：

- 目标消失的帧数是否小于 `max_age`；
- 重新出现的位置是否落在 Kalman 预测的马氏门内；
- 期间是否有其他 detection 抢占了这个 Track 的匹配。

这也是为什么 `max_age` 不能随便调大：太小会频繁换 ID，太大则可能把错误检测
吸附到一条已经消失的轨迹上。

### 8.2 马氏门控到底做什么

马氏门控不是“选最近的 detection”，而是先判断 pair 是否统计上合理：

```text
y = detection_position - predicted_position
S = H P H^T + R
d² = y^T S^-1 y
```

其中 `P` 来自 Kalman 的预测协方差，`R` 是 measurement noise。当前实现优先用
线性方程求解而不是显式矩阵求逆。若 `d² > gating_threshold`，这个 track/detection
pair 会被标记为无效，Hungarian 不允许它参与匹配。

它解决的是“几何距离相同但可信度不同”的问题：

```text
协方差小：模型很确定，允许的偏差小
协方差大：模型不确定，允许的偏差大
```

当前默认 `gating_threshold=9.21` 是三维位置观测常用的卡方阈值量级。门控负责
删掉不可能的边，Hungarian 才负责在剩余边中做全局一对一选择；二者不是同一个步骤。

### 8.3 ID 如何管理

Tracker 内部维护 `_next_track_id`，新建 Track 时取当前值后加一：

```text
_next_track_id = 1
new Track -> 1
new Track -> 2
new Track -> 3
```

ID 单调递增且不复用。Lost Track 恢复时使用原 ID，Deleted Track 再出现时一定获得
新 ID。它与 KITTI annotation 的 `track_id`、YOLO 检测序号、PointPillars 检测序号
都不是同一个概念：

```text
KITTI track_id     = 数据集真值身份
Tracker track_id   = 本次运行分配的预测身份
detection index    = 当前帧数组下标，不能跨帧使用
```

真实评估时，预测 `track_id` 要通过位置/类别匹配与 KITTI 真值身份比较，不能直接
拿两个整数相等来判断跟踪正确。

### 8.4 2D 检测当前是否也做 Kalman

当前没有。YOLO26 输出的 `Camera2DDetection` 只经过 score threshold、letterbox
逆变换和 NMS，不进入项目的 3D Kalman Filter。当前数据流是：

```text
PointPillars 3D Detection -> 3D Kalman/Tracker
YOLO26 2D Detection       -> Projection 后的跨模态 Association
```

原因是当前 Kalman 状态是 `[px, py, pz, vx, vy, vz]`，而 2D bbox 是像素坐标
`[x1, y1, x2, y2]`，没有直接深度。若要对 2D 做时间滤波，可以单独实现
`[cx, cy, w, h, vx, vy, vw, vh]` 的 2D tracker；若要用 2D 更新 3D，则需要
相机观测模型和 EKF/UKF，目前都不在本项目实现范围内。

### 置信度不等于 Track 可靠性

Detector score 只表示模型对当前检测框的信心，不等于轨迹身份稳定。工程上通常会
同时考虑：

- detection score；
- Mahalanobis distance；
- track age/hits；
- `time_since_update`；
- 目标类别和尺寸变化。

低分框不一定立刻删除，高分框也不一定一定正确。实际系统经常采用“检测分数、运动
一致性、轨迹生命周期”联合决策。

## 9. ReID 应该什么时候用

当前项目已提供可选的 FastReID 外观适配器，但它不改变 3D Tracker 的主状态。ReID 是
外观特征，用于回答：

> 这个新检测是否仍然属于之前的那个目标？

实际工程中不建议每一对 track/detection 都强制跑 ReID，原因是耗时、光照变化和相似
外观会制造新的误匹配。常见策略是：

```text
高置信度检测 + 运动距离小
    -> 直接接受几何/运动关联，不调用 ReID

低置信度、遮挡恢复、运动 gating 边界、轨迹交叉
    -> 才使用 ReID 作为二级证据

ReID 与运动都不一致
    -> 不匹配，等待新轨迹或保持 Lost
```

ReID 应当是 gated fallback，而不是替代 Kalman/Hungarian 的全局关联。当前实现由
`FastReIDEncoder`、`AppearanceGallery` 和 `ReIDIdentityManager` 组成：FastReID 负责
crop -> embedding，gallery 负责 EMA 特征记录，manager 负责长时间消失后的 public ID
恢复；Tracker 本身不 import FastReID。

本项目使用 VehicleID 的 `vehicleid_bot_R50-ibn.pth`，官方配置为
`/tmp/fast-reid/configs/VehicleID/bagtricks_R50-ibn.yml`。运行环境需要 FastReID 源码
和 `yacs`、`faiss-cpu` 等可选依赖；主项目环境不强制安装它们。启用方式：

```bash
PYTHONPATH=/tmp/fast-reid:$PYTHONPATH \
  /data/weijiaman/miniconda3/envs/mmdet3d-kitti/bin/python \
  scripts/run_end_to_end.py --sequence 0000 --enable-reid \
  --fastreid-root /tmp/fast-reid
```

FastReID 输出 2048 维 L2-normalized vehicle embedding。当前只在可靠的 2D/3D
`FusedDetection` 上更新 gallery；没有可靠 crop 时不更新，避免外观库被误检污染。

## 10. Camera-LiDAR Fusion 到底融合了什么

必须区分三个阶段：

1. **Projection**：把 3D box 投影成图像 2D box；
2. **Cross-modal association**：用类别、IoU、中心距离判断 2D/3D 是否对应；
3. **Sensor fusion**：生成 `FusedDetection`，保留两侧观测和匹配置信度。

当前不是把 2D bbox 和 3D position 简单平均。原因是 2D bbox 没有直接深度，不能可靠
地修正 3D 位置。当前 Tracker 的 3D state 由 LiDAR 3D detection 更新，YOLO26 主要
提供图像侧目标和一致性证据。

如果未来使用 2D bbox 更新 3D state，需要定义 camera measurement model、Jacobian 或
UKF sigma points、观测噪声，以及误匹配保护。这才属于 EKF/UKF 状态级融合。

当前单帧 `run_fusion_demo.py` 仍可使用 `SyntheticCameraDetector` 做几何流程 mock；
真实端到端脚本使用 YOLO26 ONNX。FastReID 是可选的车辆外观身份恢复适配器，不把
2D 观测直接写入 3D Kalman state，也不是 neural sensor fusion。

### 10.1 当前 2D/3D 跨模态关联的具体步骤

在真实端到端脚本中，先取 Tracker 当前帧已 Confirmed 的 3D track，临时包装成
`Detection3D`，再调用 `associate_camera_lidar_detections()`：

```text
TrackState（LiDAR 坐标）
    ↓ project_detection_to_image()
projected 2D bbox（像素）
    + YOLO26 Camera2DDetection（像素）
    ↓ 类别一致性检查
    ↓ 计算 IoU 和 bbox 中心距离
    ↓ IoU 或中心距离通过阈值
    ↓ IoU 优先、中心距离次优的贪心一对一匹配
FusedDetection
```

这里有一个容易混淆的细节：MOT Tracker 内部的 detection-to-track 关联使用
Mahalanobis + Hungarian；Camera-LiDAR Fusion 当前使用的是 IoU 优先的贪心一对一
匹配。Fusion 不会把 YOLO 检测直接写回 Kalman state。

### 10.2 中心距离为什么还要存在

中心距离是两个图像框中心的像素欧氏距离：

```text
|| center(projected_3d_bbox) - center(camera_2d_bbox) ||₂
```

它用于补充 IoU：投影框可能因为 3D 尺寸、遮挡或截断导致面积重叠很小，但两个
中心仍然接近。当前默认阈值是 `80.0`，单位是像素，不是米；匹配条件是：

```text
class_id 相同 且 (IoU >= iou_threshold 或 center_distance <= threshold)
```

中心距离阈值太小会漏掉有标定误差的正确 pair，太大会把相邻目标错误连起来。它
不能单独证明两个框是同一个物体，因此必须结合类别、IoU 和一对一约束；当前可选的
FastReID 只在可靠关联后提供车辆外观相似度，作为长时间身份恢复的二级证据。

### 10.3 未匹配的 2D 和 3D 怎么处理

- 只有 LiDAR 3D：可以继续进入 Tracker，表示 Camera 未检出、遮挡或不在 FOV；
- 只有 Camera 2D：当前只保留为 Camera-side observation，不凭空生成 3D 深度；
- 两者匹配：生成 `FusedDetection`，保存两侧 detection、projected bbox、confidence
  和真实 Tracker ID。

这是一种保守策略：宁可暂时不融合，也不把错误的 2D 框硬塞给 3D 状态估计。

## 11. 统一端到端脚本

真实端到端入口是：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data/weijiaman/miniconda3/envs/mmdet3d-kitti/bin/python \
scripts/run_end_to_end.py --sequence 0000 --device cuda:0
```

脚本内部的实际顺序是：

```text
load FrameData
  ↓
PointPillars.predict(frame)
  ↓
Yolo26OnnxCameraDetector.predict(frame)
  ↓
MultiObjectTracker.update(lidar_detections)
  ↓
把 confirmed Track 转成临时 Detection3D
  ↓
project_detection_to_image()
  ↓
associate_camera_lidar_detections()
  ↓
绘制原始框、投影框、融合框
  ↓
VideoWriter + summary JSON
```

当前真实 `0000` 全序列输出曾记录为：154 帧、PointPillars 1,594 个 3D 框、YOLO26
727 个 2D 框、确认轨迹 1,214 个、融合结果 482 个。运行时间受 GPU、模型缓存、视频
编码和机器负载影响，不能把一次 FPS 当成固定理论性能。

在另一段 KITTI 场景 `0001` 上启用 FastReID 并完整处理 443 帧时，实际记录了 29 次
public ID reactivation：PointPillars 4,687 个 3D 框、YOLO26 3,118 个 2D 框、
确认轨迹 4,111 个、融合结果 2,349 个，端到端 4.36 FPS。该结果说明 ReID 分支确实
参与了跨帧身份管理，但 reactivation 次数不是准确率，仍需专门的 GT identity 评估。

## 12. 真实 MOT 评估与 GT 验证的区别

`evaluate.py` 使用 GT/NoisyGT，是为了隔离验证 Tracker。它回答：

```text
给 Tracker 理想或可控噪声检测，生命周期和关联逻辑是否正确？
```

`evaluate_real_mot.py` 才使用真实 PointPillars 输出，并将 KITTI camera GT 转到 LiDAR
坐标后评估。它回答：

```text
真实误检、漏检和定位误差进入后，Tracker 的表现怎样？
```

当前真实 PointPillars `0000` 全序列的旧版逐帧记录：

```text
GT 711, matches 448, FP 766, FN 263, IDS 8
MOTA -0.4585, IDF1 0.4655, FPS 25.10
```

该结果中的 IDF1 是逐帧位置匹配近似，不是全局 identity 指标。负 MOTA 不是应该隐藏的
结果，而是说明真实检测质量和 Tracker 输入质量仍是瓶颈。

### 12.1 基于 KITTI identity 的标准 MOT 指标

`scripts/evaluate_real_mot.py` 现在使用 KITTI `label_02` 的 `track_id` 作为 GT identity，
通过全局一对一 identity assignment 计算标准 MOTA、IDF1、IDP、IDR、IDS、FP 和 FN。
结果文件为：

```text
outputs/metrics/real_pointpillars_mot_0000_standard.json
outputs/metrics/real_pointpillars_mot_0001_standard.json
```

真实结果如下（2 m 三维中心距离阈值）：

```text
sequence 0000: MOTA -0.4585, IDF1 0.4010, IDP 0.3180, IDR 0.5429, IDS 8, FP 766, FN 263
sequence 0001: MOTA  0.1700, IDF1 0.6395, IDP 0.5527, IDR 0.7589, IDS 4, FP 1799, FN 682
```

当前旧版 MMDetection3D 环境没有安装 `motmetrics`，因此本次 JSON 的
`metric_backend` 为 `scipy_fallback`；该 fallback 使用同样的帧级距离匹配和全局
Hungarian identity assignment。`0000` 的旧逐帧 IDF1 `0.4655` 仍保留在
`legacy_framewise`，不能与标准 IDF1 `0.4010` 混用。

## 13. 常见实践坑

### 13.1 训练数据泄漏

KITTI Tracking 是连续视频。随机按图片切分会让几乎相同的相邻帧同时出现在 train
和 val，指标看起来很好但泛化不可信。应按 sequence 切分，最好把评估 sequence 完全
留作 hold-out。

### 13.2 低分框不是简单删除

低置信度检测可能是遮挡目标，也可能是背景误检。常见做法是先提高关联门槛、限制
生命周期影响，或让低分框只参与 tentative track，而不是完全丢弃。

### 13.3 漏检与轨迹删除不同

一次漏检不代表目标消失。`max_age` 允许短暂 Lost；过早删除会造成新 ID，过晚保留
会造成漂浮轨迹。这个参数必须通过漏检实验而不是凭经验设置。

### 13.4 视频编码会裁掉奇数尺寸

KITTI 图像高度 375 是奇数。某些 `mp4v` 编码器会静默改成 374，因此视频脚本在写
入前补一行 padding。padding 只用于视频，不改变任何投影坐标。

### 13.5 先检查环境再怀疑模型

PointPillars 使用旧环境，YOLO26 ONNX 使用 OpenCV DNN，主项目使用 Python 3.10。
先检查 `torch.cuda.is_available()`、模型路径、ONNX 输入输出 shape，再分析算法效果。

## 14. 面试回答模板

### 问：为什么 Tracker 不直接 import Detector？

答：Detector 输出先统一成 `Detection3D`，Tracker 只依赖这个数据契约。这样可以把
GT、PointPillars 或未来的 CenterPoint 替换掉，而不用修改跟踪核心。

### 问：为什么有 2D 框和 3D 框两个框？

答：它们来自不同传感器和坐标系。先把 3D 框投影成 2D，再用类别、IoU 和中心距离做
跨模态一对一关联，匹配后生成 `FusedDetection`。可视化保留两个原始框，是为了检查
标定和匹配质量，不是重复输出。

### 问：为什么不用 2D 框直接修正 3D 位置？

答：2D bbox 没有直接深度，简单平均会引入错误深度。需要明确的 camera measurement
model、观测噪声和 EKF/UKF 更新，目前项目采用更稳妥的 object-level fusion。

### 问：为什么有 MOTA=1.0，又有 MOTA=-0.4585？

答：前者是 GTDetector 驱动的 Tracker validation，后者是真实 PointPillars 输出驱动的
MOT 评估。两者输入不同，不能混为一个系统精度结论。

### 问：ReID 应该放在哪里？

答：作为独立 AppearanceAdapter，在运动 gating 后对低置信度、遮挡恢复和轨迹交叉等
困难 pair 提供二级 cost。高置信度且运动一致时不调用 ReID，以控制延迟和误匹配。
当前项目的 ReID 需要显式启用，并依赖外部 FastReID 源码及 VehicleID 权重。

### 问：为什么 PointPillars 前面不默认做 ground removal？

答：官方预训练模型已经假设 KITTI 原始输入分布。外部 ground removal 会改变输入并
可能删除目标点，所以当前只在可视化/独立实验使用；若要接入，必须做 ablation。

### 问：如何证明不是数据泄漏？

答：按 sequence 切分训练和验证，评估 sequence 不进入训练；同时把训练配置、Git hash、
帧范围和模型路径写入结果 JSON。

### 问：Track Lost 后重新出现，什么时候保留旧 ID？

答：只要 Track 尚未超过 `max_age` 变成 Deleted，且新 detection 通过马氏门控并被
关联到原 Track，就调用原 Track 的 `update()`，保留旧 ID。超过窗口或门控失败后，
只能新建 Track；启用 FastReID 后，才会在可靠 2D/3D 匹配的基础上尝试长时间身份恢复。

### 问：马氏门控和 Hungarian 为什么都需要？

答：马氏门控删除统计上不可能的边，Hungarian 在剩余候选上做全局一对一最优分配。
没有 gating 时，Hungarian 可能被迫选择一个很远的错误 pair；没有 Hungarian 时，
多个 Track 可能竞争同一个 detection。

### 问：为什么不用一个全局 ID 表直接管理目标身份？

答：Track ID 是运行时预测身份，必须由生命周期和关联维护；KITTI `track_id` 是离线
真值，不能泄漏给 Tracker。评估阶段才把两者通过空间/类别匹配比较。

### 问：为什么 Camera 2D 不进入当前 Kalman？

答：当前 KF 状态是三维位置和速度，2D bbox 没有直接深度。直接把像素坐标当三维位置
会造成量纲和观测模型错误。需要独立 2D KF，或使用 EKF/UKF 建立投影观测模型。

### 问：2D/3D 关联为什么同时使用 IoU 和中心距离？

答：IoU 对框大小和截断敏感，中心距离对小幅几何偏差更宽容。当前使用类别一致且
IoU 或中心距离满足阈值，再做一对一匹配；中心距离阈值是像素单位，不是三维米制距离。

### 问：Fusion 为什么不直接把两个框平均成一个框？

答：2D 框没有深度，不能和 3D position/dimensions 做同量纲平均。当前融合保留两侧
观测和置信度，3D 状态仍由 LiDAR Tracker 负责，避免错误 2D 深度把轨迹拉偏。

### 问：低置信度检测应该怎么处理？

答：不要只按一个 score 开关处理。可以让低分框只参与 tentative track，或要求更严格
的运动/几何一致性；在遮挡恢复、gating 边界、交叉目标等困难 pair 上再启用 ReID
作为二级证据。当前使用 FastReID，接口仍保持独立，Tracker 不依赖具体外观网络。

### 问：为什么真实 PointPillars MOT 的 MOTA 可能是负数？

答：MOTA 会同时惩罚 FP、FN 和 IDS。若检测器误检/漏检很多，错误数超过 GT 数量，
MOTA 就可能小于 0；这是真实 detector-driven 结果，不应改成 GT validation 的 1.0。

### 问：如何定位“框不对齐”是模型问题还是标定问题？

答：先用 identity/known transform 单测，再检查 camera bottom-center 到 box center、
`Tr_velo_to_cam`、`R0_rect`、`P2` 的顺序，最后用 raw 点云投影检查 FOV。不要一开始
调 IoU 或中心距离阈值，否则会用关联参数掩盖坐标错误。

### 问：为什么 PointPillars 前面不默认做 ground removal？

答：官方权重假设 KITTI 原始点云分布，外部地面移除可能删除目标点并改变输入分布。
当前预处理用于独立分析和可视化；若接入 Detector，必须做 raw/non-ground ablation。

### 问：端到端 FPS 为什么和单个模块 FPS 不一样？

答：端到端包含图像解码、PointPillars GPU 推理、YOLO ONNX CPU 推理、Tracker、投影、
融合和视频写入；单模块测速通常只覆盖其中一段。报告性能时必须说明计时边界。

## 15. 建议的学习顺序

```text
1. scripts/inspect_kitti.py
2. common/types.py
3. dataset/kitti.py
4. dataset/calibration.py + common/geometry.py
5. pointcloud/preprocess.py
6. detection/adapter.py
7. tracking/kalman.py
8. tracking/association.py + tracking/tracker.py
9. fusion/camera_lidar.py + fusion/yolo26.py
10. scripts/run_end_to_end.py
11. scripts/evaluate_real_mot.py
12. ros2_ws/src/perception_tracker/
```

每读完一个模块，都运行对应测试。工程学习不是只看数学公式，而是要同时追踪：

- 输入 shape；
- 坐标系；
- score/threshold；
- 空输入和异常输入；
- 哪个模块拥有状态；
- 哪个模块只做适配；
- 输出如何被下一模块消费。

## 16. 最后记住一句话

这个项目真正展示的不是“调用了两个模型”，而是把不同坐标系、不同时间特性和不同
可靠性的传感器输出，经过明确的数据契约、状态估计、关联和可视化，组织成一条可以
测试、复现、解释和承认局限的工程流水线。
