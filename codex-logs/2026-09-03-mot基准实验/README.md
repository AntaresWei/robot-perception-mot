# MOT基准实验

- **日期**：2026-09-03
- **基线 commit**：`8c9be185d82818439b64cafea6f65dd727560b18`
- **规模**：5 个文件，+673 / -0
- **完整 diff**：[changes.patch](./changes.patch) ｜ [变更清单](./files.txt)

## 1. 需求：要解决什么问题

建立可重复的 KITTI Tracking benchmark，并在完全相同的数据集、sequence、帧范围
和 detection input 下对比三个方法：无 Kalman 的 Nearest Neighbor、Kalman +
Euclidean Association、以及 Kalman + Mahalanobis Gating + Hungarian。每次运行
必须自动产出 MOTA、IDF1、IDS、FP、FN、FPS、latency、配置、Git hash 和报告。

## 2. 改了什么

1. 在 `configs/pipeline.yaml` 声明 benchmark 输入、帧范围、阈值、输出路径和绘图开关。
2. 新增 `scripts/evaluate.py`，预计算一次 detection stream，保证三个方法的输入严格
   一致，并输出三个 JSON、统一柱状图和 `docs/experiments.md`。
3. Baseline A 是无 Kalman 的 Euclidean nearest-neighbor；Baseline B 复用项目
   `Track` 的 KF，但使用 Euclidean 关联；Proposed 直接适配生产
   `MultiObjectTracker`。
4. 增加指标和空输入的 pytest；不改生产 Tracker 或任何核心数学逻辑。

## 3. 为什么这么改

- **公平性**：Detector 会引入随机性和 I/O 开销，先预计算相同 detection stream，再
  独立计时各 tracker，使 FPS/latency 只反映 tracking/association。
- **指标实现**：按 frame 的类别一致、位置阈值一对一匹配计算 FP/FN/IDS/MOTA 与可复现
  的 IDF1 baseline；不把 Tracker 内部关联结果当评估结果。
- **取舍**：不新增大型 MOT 评估依赖。当前 IDF1 不是 MOTChallenge 的全局 ID 指标，
  已在报告中明确限制。
- **实验边界**：本次使用 GT detector，因此所有方法 MOTA/IDF1 为 1.0。结果只说明
  GT tracking pipeline 能跑通，不能作为真实 detector 的 mAP、recall 或端到端性能。

## 4. 验证与可复述结论

```bash
/data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python -m pytest tests/
# 95 passed

MPLCONFIGDIR=/tmp/matplotlib-cache \
  /data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python \
  scripts/evaluate.py --config configs/pipeline.yaml --sequence 0000
```

实际运行 sequence `0000` 的 154 帧、711 个 GT object：三个方法均得到 MOTA 1.0000、
IDF1 1.0000、IDS/FP/FN 为 0。运行时从实际进程计得：NN 6766.61 FPS、KF 1364.61 FPS、
Proposed 912.12 FPS；CPU 峰值 RSS 68.17 MB。GPU memory 为 **NOT RUN**，因为输入是
GT detector 而非 GPU detector。运行后生成了 `baseline_nn.json`、`baseline_kf.json`、
`proposed.json`、`mot_comparison.png` 和 `docs/experiments.md`。
