# MOT 实验报告

本报告由 `scripts/evaluate.py` 自动生成，指标来自实际运行结果，不手工填写。

## Configuration

- Git commit: `38c0a143f638422118a17a76063e16dfb085a9da`
- Dataset sequence: `0000`
- Frame range: `0..153`
- Timestamp (UTC): `2026-09-03T07:16:51.987255+00:00`
- Config: `{"pipeline": "/data/weijiaman/MyCode/2026/robot-perception-mot/configs/pipeline.yaml", "dataset": "/data/weijiaman/MyCode/2026/robot-perception-mot/configs/dataset.yaml", "detector": "/data/weijiaman/MyCode/2026/robot-perception-mot/configs/detector.yaml", "tracker": "/data/weijiaman/MyCode/2026/robot-perception-mot/configs/tracker.yaml"}`

## Metrics

| Method | MOTA | IDF1 | IDS | FP | FN | FPS | latency/frame (ms) | CPU peak RSS (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_nn | 1.0000 | 1.0000 | 0 | 0 | 0 | 6701.92 | 0.149 | 62.94 |
| baseline_kf | 1.0000 | 1.0000 | 0 | 0 | 0 | 1094.63 | 0.914 | 62.94 |
| proposed | 1.0000 | 1.0000 | 0 | 0 | 0 | 1034.87 | 0.966 | 62.94 |

## Runtime

每个方法在同一预计算 detection stream 上独立运行；FPS 和 latency 只统计 tracker/association，
不包含 KITTI 文件解码和 detector 推理。GPU memory: NOT RUN（本实验使用 GT detector）。

## Observations

- Baseline A 不使用 Kalman prediction，容易在漏检、交叉或临近目标场景中丢失轨迹。
- Baseline B 使用 Kalman，但 Euclidean cost 未使用协方差；Proposed 还包含 Mahalanobis gating 和 Hungarian 一对一关联。
- GT/NoisyGT detector 仅用于验证 tracking pipeline，不能解释为 detector accuracy、mAP 或 recall。

## Limitations

- 当前 IDF1 是按帧位置匹配得到的可复现实验基线，不替代完整 MOTChallenge 全局 ID 指标实现。
- 若配置真实 detector，脚本会拒绝运行并明确标记 NOT RUN，避免伪造结果。
