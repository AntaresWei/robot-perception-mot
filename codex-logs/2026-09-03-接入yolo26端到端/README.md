# 接入 YOLO26 端到端

- **日期**：2026-09-03
- **基线 commit**：`9e998e4`
- **规模**：13 个文件，+886 / -11
- **完整 diff**：[changes.patch](./changes.patch) ｜ [变更清单](./files.txt)

## 1. 需求

使用 YOLO26 负责 KITTI Camera 2D 三类检测，保留 MMDetection3D PointPillars 负责
LiDAR 3D box，并打通真实的：

```text
KITTI image/bin -> YOLO26 2D + PointPillars 3D -> Detection/Tracker
                 -> Calibration Projection -> Camera-LiDAR Fusion -> MP4/JSON
```

## 2. 改了什么

1. 将 `yolo26s.pt` 和 PointPillars `.pth` 统一放入 `weights/`，新增 YOLO26 KITTI
   三类数据转换、训练配置、训练脚本和 ONNX 导出脚本。
2. 将 KITTI `Car/Van/Truck` 映射到 Car，`Pedestrian/Person_sitting` 映射到 Pedestrian，
   `Cyclist` 保留为 Cyclist；数据按 sequence 切分为 5,628 train / 2,380 val 图像，
   共 45,198 个框。
3. 新增 `Yolo26OnnxCameraDetector`，在主项目环境用 OpenCV DNN 加载三类 ONNX，输出
   统一 `Camera2DDetection`。
4. 新增 `run_end_to_end.py`：PointPillars 3D + YOLO26 ONNX 2D + 项目 Tracker +
   calibration projection + cross-modal association + `FusedDetection`，输出视频和 JSON。
5. 增加相应 pytest；未修改 Kalman、Association、Tracker、Calibration 核心数学。

## 3. 为什么这么改

- **环境隔离**：主环境继续运行核心包和 OpenCV DNN；YOLO26 训练/导出在继承主环境
  CUDA 的 `/tmp/robot-perception-mot-yolo26` 临时环境；PointPillars 保持官方
  `mmdet3d-kitti` 环境，避免混装旧版 MMCV/PyTorch。
- **真实三类训练**：官方 `yolo26s.pt` 是 COCO 80 类，只做初始化；先用 KITTI
  `label_02` 训练 `Car/Pedestrian/Cyclist` 三类 head，再导出 ONNX，避免直接用 COCO
  类别映射冒充 KITTI 模型。
- **坐标边界**：PointPillars 输出 LiDAR 坐标 3D box，YOLO26 输出图像坐标 2D box；
  Tracker 只消费 LiDAR `Detection3D`，最后通过标定投影和融合模块跨坐标系关联。

## 4. 验证与结论

已实际完成：

```bash
# 三类数据准备
python scripts/prepare_kitti_yolo.py --output data/yolo26_kitti_trainset

# YOLO26 10 epoch GPU 训练（RTX 4090）
/tmp/robot-perception-mot-yolo26/bin/python scripts/train_yolo26_kitti.py --epochs 10 --device 0

# ONNX 导出
/tmp/robot-perception-mot-yolo26/bin/python scripts/export_yolo26_onnx.py

# 主环境 ONNX 单帧推理：类别已限制为项目三类
# 实际输出 0000/100：3 个高分框，class_id 仅为 1/3

# 真实全序列端到端
CUDA_VISIBLE_DEVICES=0 /data/weijiaman/miniconda3/envs/mmdet3d-kitti/bin/python \
  scripts/run_end_to_end.py --sequence 0000 --device cuda:0

# pytest
/data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python -m pytest tests/
# 103 passed
```

真实 `0000` 全 154 帧结果：PointPillars 1,594 个 3D detections，YOLO26 727 个 2D
detections，确认轨迹 1,214 个，融合结果 482 个；端到端耗时 16.49 秒、9.34 FPS、
94.18 ms/帧。输出为 `outputs/videos/end_to_end_0000.mp4` 和
`outputs/metrics/end_to_end_0000.json`。YOLO26 10 epoch 验证结果为 Precision 0.62777、
Recall 0.53965、mAP50 0.53000、mAP50-95 0.32233。当前没有标准 detector 全序列
mAP/recall 或真实 detector MOT ground-truth 指标，不能把以上 replay 统计解释为模型
精度；后续可继续训练和按官方协议评估。
