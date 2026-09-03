# 融合可视化与视频

- **日期**：2026-09-03
- **基线 commit**：`8c9be185d82818439b64cafea6f65dd727560b18`
- **覆盖提交**：`ce5fe12`、`38c0a14`
- **规模**：11 个文件，+1212 / -42
- **完整 diff**：[changes.patch](./changes.patch) ｜ [变更清单](./files.txt)

## 1. 需求：要解决什么问题

检查 fusion visualization 中投影点的真实来源，使 raw、non-ground 与无点云模式
可配置；将点云按 rectified camera depth 着色，输出 raw/non-ground 对比图；并把
KITTI sequence `0000` 的最终 non-ground fusion visual 离线渲染为可检查的 MP4。

这两次连续提交还包含先前未提交的 MOT benchmark 文件，因此本归档一并保留其可复现
配置、评估脚本、JSON 指标生成逻辑和自动生成的实验报告。

## 2. 改了什么

1. `configs/fusion.yaml` 新增 `visualization.lidar_points_mode`、深度色表和视频
   默认参数；raw 用于 calibration debug，non-ground 用于最终 demo。
2. `scripts/run_fusion_demo.py` 将 raw / `preprocess_point_cloud(...).non_ground_points`
   投影为与像素对齐的 rectified camera depth，并使用 TURBO 色表绘制。
3. 新增 `scripts/render_fusion_video.py`，复用单帧渲染函数并通过 `cv2.VideoWriter`
   写出 MP4；视频前对奇数高度补边，避免 mp4v 静默裁掉末行。
4. 补充 benchmark 配置、评估脚本、实验报告和 pytest，全部指标由实际运行生成。

## 3. 为什么这么改

- **可视化输入明确化**：此前绿色点直接来自 `frame.lidar_points`，没有经过 ROI、
  ground removal 或其他预处理。新配置让 calibration debug 与最终 demo 的点云语义
  可区分且可复现。
- **复用渲染路径**：视频调用 `render_fusion_modes()`，不通过截图或重复实现 fusion，
  因而与单帧 PNG 使用相同的 box、投影、深度颜色和预处理参数。
- **取舍**：MOT benchmark 以 GT input 验证跟踪链路，不能报告为真实 detector 性能；
  GPU memory 对 GT input 记为 NOT RUN。
- **风险与后续**：当前 Camera 侧仍为 SyntheticCameraDetector，视频仅验证实验性
  object-level fusion；真实 Camera detector 与端到端 PointPillars 视频仍待接入。

## 4. 验证与可复述结论

```bash
/data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python -m pytest tests/
# 101 passed

python scripts/render_fusion_video.py --sequence 0000
# 154 frames, 1242x376, 10 FPS
```

实际生成 `outputs/videos/fusion_0000_non_ground.mp4`：154 帧、15.4 秒、约 18 MB，
离线处理 35.73 秒（4.31 FPS）。单帧 `0000/100` 的 raw/non-ground 投影分别为
18,995 / 6,004 点。MOT benchmark 运行 `0000` 的 154 帧、711 个 GT object；由于
统一输入为 GT，三种方法 MOTA/IDF1 均为 1.0，IDS/FP/FN 为 0，此结果只证明 tracking
pipeline 可运行。
