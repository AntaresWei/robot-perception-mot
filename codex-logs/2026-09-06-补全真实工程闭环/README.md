# 补全真实工程闭环

- **日期**：2026-09-06
- **基线 commit**：`f6791cf`
- **规模**：7 个文件，+288 / -21
- **完整 diff**：[changes.patch](./changes.patch) ｜ [变更清单](./files.txt)

## 1. 需求：要解决什么问题

把项目从“多个可运行模块”补成可解释、可复现、可审计的真实感知工程：YOLO26
相机三类检测、PointPillars LiDAR 3D 检测、Tracker、Calibration Projection、
Camera-LiDAR Fusion、真实 Detector MOT 评估和严格 hold-out 数据划分必须形成闭环。

## 2. 改了什么

1. 新增 `evaluate_real_mot.py`，将 KITTI camera GT box center 转换到 LiDAR 坐标，
   对真实 PointPillars -> Tracker 输出计算 MOTA、IDF1、IDS、FP、FN。
2. 给 YOLO 数据转换器增加显式 `--val-sequences`，支持把 sequence `0000` 完全留作
   hold-out；增加 hold-out 配置、模型和 ONNX 产物路径。
3. 端到端脚本增加 `--camera-model`，可以用 hold-out YOLO26 ONNX 运行无泄漏 replay。
4. 修正端到端 JSON 的实际帧范围、视频尺寸和 README 中重复/过期说明。
5. 增加真实评估坐标转换单测；全量 pytest 通过 104 项。

## 3. 为什么这么改

- **分离实验语义**：GT benchmark 只验证 Tracker；真实 PointPillars MOT 评估单独记录，
  不再用 GT 的 1.0 结果暗示真实 Detector 性能。
- **避免数据泄漏**：YOLO26 hold-out 训练不包含 sequence `0000`，可以明确区分训练内
  replay 和未见 sequence 泛化验证。
- **保留模块解耦**：真实评估和端到端脚本只负责装配 Detector、Tracker、Fusion；没有
  修改 Kalman、Association 或 Tracker 的核心数学。

## 4. 验证与可复述结论

全量测试：

```bash
/data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python -m pytest tests/
# 104 passed
```

真实 PointPillars `0000` 全 154 帧 MOT 结果：GT 711、matches 448、FP 766、FN 263、
IDS 8、MOTA `-0.4585`、IDF1 `0.4655`、FPS `25.10`。这揭示了真实预训练 detector
误检/漏检对 tracking 的影响，不能用 GT benchmark 替代。

YOLO26 全量训练模型的 10 epoch 验证结果为 mAP50 `0.53000`；严格 hold-out `0000`
模型实际训练完成 10 epochs，mAP50 `0.43773`、mAP50-95 `0.30819`，并已成功导出 ONNX。
真实端到端 replay 已处理 154 帧，生成 1,594 个 PointPillars 3D 框、727 个 YOLO26
2D 框和 482 个融合对象。剩余限制是：尚未按 KITTI 官方 detector 协议计算完整 mAP，
当前 Fusion 仍是 object-level association，不是 EKF/UKF 状态级融合。
