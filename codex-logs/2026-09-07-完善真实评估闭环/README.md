# 完善真实评估闭环

- **日期**：2026-09-07
- **基线 commit**：`f6791cf`
- **规模**：8 个文件，+316 / -21
- **完整 diff**：[changes.patch](./changes.patch) ｜ [变更清单](./files.txt)

## 1. 需求：要解决什么问题

将“真实 PointPillars + YOLO26 replay 已跑通”进一步补成可经受面试追问的实验闭环：
严格区分 GT tracking validation 与真实 detector MOT，避免 sequence 泄漏，并记录
真实 detector 驱动的 MOTA、IDF1、IDS、FP、FN。

## 2. 改了什么

1. 新增 `scripts/evaluate_real_mot.py`，把 KITTI camera GT bottom-center box 转到
   LiDAR 坐标，再用真实 PointPillars -> Tracker 输出计算 MOT 指标。
2. 增加 `--val-sequences`，生成不包含 sequence `0000` 的 YOLO26 hold-out 数据集和
   对应训练配置；端到端脚本增加 `--camera-model` 以使用 hold-out ONNX。
3. 增加真实评估坐标转换、YOLO 数据划分单元测试，并修正端到端实际帧范围记录。
4. README 补充真实 detector MOT 结果、hold-out 结果、实验限制和数据边界。

## 3. 为什么这么改

- **实验语义分离**：GTDetector benchmark 仍用于验证 Tracker；真实 PointPillars 指标
  单独生成，不能用 GT 的 1.0 结果包装真实系统。
- **坐标正确性**：KITTI GT 在 rectified camera，PointPillars 在 LiDAR；评估前显式
  做逆变换并将 bottom-center 修正为 box center。
- **防止数据泄漏**：`0000` 的 154 帧不进入 hold-out YOLO 训练，只用于验证和 replay。

## 4. 验证与可复述结论

```bash
/data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python -m pytest tests/
# 106 passed

CUDA_VISIBLE_DEVICES=0 /data/weijiaman/miniconda3/envs/mmdet3d-kitti/bin/python \
  scripts/evaluate_real_mot.py --sequence 0000 --device cuda:0
```

真实 PointPillars `0000` 全 154 帧结果：GT 711、matches 448、FP 766、FN 263、IDS 8、
MOTA `-0.4585`、IDF1 `0.4655`、FPS `25.10`。该负 MOTA 真实反映误检/漏检对 Tracker
的影响。hold-out YOLO26（0000 不参与训练）完成 10 epochs，mAP50 `0.43773`、
mAP50-95 `0.30819`；导出的 ONNX 已通过 10 帧真实端到端 replay smoke。工程现在同时
具备训练内 replay、严格 hold-out、真实 detector MOT 和端到端 fusion 四种可区分证据。
