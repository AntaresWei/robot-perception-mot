# 测试FastReID场景

- **日期**：2026-09-07 20:35
- **基线 commit**：`3e051e082a6bcf5c91b2d8ddef2f2f1264453e28`
- **规模**：1 个文件，+27 / -2
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

本次验证将真实端到端 replay 从已测试的 KITTI `0000` 切换到不同场景
`0001`，检查 PointPillars、YOLO26、3D Tracker、Camera-LiDAR fusion 和
FastReID public ID 管理是否能在另一段数据上连续运行。

**原始需求 / prompt：**

```text
测试换用数据集别的场景；使用 KITTI sequence 0001 完整运行 FastReID 端到端流程。
```

**验收标准：**

- [x] 完成 `0001` 全部 443 帧 replay，并保存 JSON 统计和 MP4 可视化。
- [x] 运行主项目 pytest，确认既有模块没有回归。
- [x] 明确 reactivation 统计不等价于身份恢复准确率。

---

## 2. 改了什么

> 只写关键改动，不要复述 diff。3～6 条为宜。
> 面试时这一节回答「具体怎么做的」。

1. **`README.md`**：补充 `0001` 场景的真实 FastReID 运行结果、输出路径和限制说明。
2. 复用现有 `scripts/run_end_to_end.py`，未修改 Tracker 核心；FastReID 仍作为外部
   appearance gallery，通过可靠的 2D/3D fusion 结果辅助 public ID 恢复。
3. 真实输出：`outputs/metrics/end_to_end_0001_reid.json` 和
   `outputs/videos/end_to_end_0001_reid.mp4`。

**改动前后对比（最关键的一处）：**

```diff
+在不同场景 `0001` 上启用 FastReID 后，完整 443 帧 replay 实际发生 29 次 public ID
+reactivation ...
```

**验证方式：**

```bash
/data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python -m pytest tests/
# 110 passed in 4.53s
```

`0001` 真实 replay 已在带 GPU 的 `mmdet3d-kitti` 环境完成：PointPillars 4687 个
3D detections、YOLO26 3118 个 2D detections、确认轨迹 4111 个、融合结果 2349 个；
总耗时 101.65 s，4.36 FPS，平均 148.26 ms/frame，FastReID public ID reactivation
计数为 29。

---

## 3. 为什么这么改

> 这一节是精华，决定面试能不能讲出深度。

- **候选方案**：继续只测 `0000`，无法检查场景泛化；切换到 `0001` 全序列，采用后者
  以获得独立场景证据。
- **权衡点**：保留 PointPillars 原始输入和现有模块边界，避免为测试改变 detector
  的训练分布；代价是单帧推理时间较长，且本次没有新增标准 ReID accuracy 评估。
- **风险与后续**：29 次 reactivation 只是管理器触发次数，不证明身份恢复正确。后续
  需要使用 KITTI track identity 做 TP/FP、ID switch 和 ReID top-1/Recall 评估。

---

## 4. 我学到了什么（面试可复述）

> 用「STAR + 技术判断」的口径写 2～4 条，每条都要能直接开口讲。
> 模板：当时遇到 X，我判断 Y，于是用 Z 解决，最终结果是 W。

1. **技术点**：我把 FastReID 放在 Tracker 外部，用 appearance gallery 映射内部
   3D track ID 到稳定 public ID，避免把网络依赖耦合进 Kalman/Hungarian 核心。
2. **工程判断**：只有 2D/3D 几何匹配可靠且融合置信度达阈值时才更新 gallery，长时间
   丢失后才尝试 ReID，控制误匹配和推理开销。
3. **一句话总结（面试开场用）**：我在 KITTI `0001` 的 443 帧真实 replay 中验证了
   PointPillars + YOLO26 + 3D MOT + FastReID 闭环，记录 29 次 public ID 恢复触发；
   同时明确该数字不是身份准确率，标准 identity evaluation 仍是后续工作。
