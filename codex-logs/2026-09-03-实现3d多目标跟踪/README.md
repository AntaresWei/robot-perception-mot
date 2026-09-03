# 实现3D多目标跟踪

- **日期**：2026-09-03 11:22
- **基线 commit**：`2e985a4ba63c0b66bb2aa47d3fcc8ef3aba44e75`
- **规模**：7 个文件，+664 / -0
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

> 把当时给 Codex 的原始提示词贴在这里（或简述背景）。
> 面试时这一节回答「你为什么做这件事」。

**任务目标：**基于已有 Kalman Filter、Mahalanobis Gating 和 Hungarian
Association，实现完整的 3D Multi-Object Tracker 和 Track 生命周期。

**验收标准：**

- [x] 支持 Tentative、Confirmed、Lost、Deleted 状态。
- [x] 支持 birth、confirmation、missed detection、deletion 和 crossing。
- [x] Track ID 单调递增且删除后不复用。
- [x] 输出 confirmed `TrackState`，提供 synthetic MOT demo 和 MOTA/IDF1/IDS。

---

## 2. 改了什么

> 只写关键改动，不要复述 diff。3～6 条为宜。
> 面试时这一节回答「具体怎么做的」。

1. **`src/robot_perception/tracking/track.py`**：封装单目标 KF、尺寸/类别
   最新观测和四阶段生命周期。
2. **`src/robot_perception/tracking/tracker.py`**：实现逐帧 predict、已有
   association API 调用、matched update、new birth、miss aging 和删除。
3. **`configs/tracker.yaml`**：增加 Tracker 数值参数及 synthetic demo 配置。
4. **`tests/test_track.py` / `tests/test_tracker.py`**：覆盖单 Track、多目标、
   新生、漏检恢复、删除、ID 唯一性、交叉轨迹和非法输入。
5. **`scripts/run_tracker_demo.py`**：生成 3 条带噪声/漏检轨迹，输出指标和图像。

**验证方式：**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot-perception-mot \
  python -m pytest tests/
```

结果：`90 passed in 0.54s`。

```bash
MPLCONFIGDIR=/tmp/robot-perception-mot-mpl \
  conda run -n robot-perception-mot python scripts/run_tracker_demo.py
```

结果：80 帧、3 个对象，`MOTA=0.7708333`、`IDF1=0.8965517`、`IDS=10`；
生成 `outputs/metrics/tracker_baseline.json` 和 `outputs/visualizations/tracker_demo.png`。

---

## 3. 为什么这么改

> 这一节是精华，决定面试能不能讲出深度。

- **职责边界**：Tracker 只接收 `Detection3D`，不导入任何具体 detector；
  KF 和 association 通过已有公共 API 调用。
- **生命周期**：每帧先预测并增加 age/time_since_update，匹配后更新并确认；
  未匹配 Track 进入 Lost，达到 `max_age` 后 Deleted；未匹配 detection 新生。
- **ID 管理**：Tracker 内部计数器只递增，即使删除 Track 也不会回收旧 ID。
- **指标策略**：demo 使用独立位置匹配统计 MOTA/IDF1/IDS，避免直接复用 Tracker
  内部关联结果造成指标自证。
- **风险与后续**：当前状态只估计中心位置/速度，尺寸和 yaw 沿用最近观测；
  crossing 在目标完全重合时仍可能发生 ID switch。

---

## 4. 我学到了什么（面试可复述）

> 用「STAR + 技术判断」的口径写 2～4 条，每条都要能直接开口讲。
> 模板：当时遇到 X，我判断 Y，于是用 Z 解决，最终结果是 W。

1. **技术点**：常速度 KF 的预测提供位置先验，Mahalanobis/Hungarian 再利用
   先验和协方差完成 detection-to-track 一对一匹配。
2. **工程判断**：将生命周期和算法状态放在 Track，将批量调度放在 Tracker，
   能在不改 KF/association 数学模块的前提下覆盖 birth、miss 和 delete。
3. **工程判断**：deleted Track 保留在历史集合中用于审计，但不再进入活动关联，
   同时 ID 计数器单调递增以避免身份复用。
4. **一句话总结（面试开场用）**：我基于已有 KF 和关联模块实现了可测试的 3D
   MOT 管理层，并用 80 帧、3 目标合成实验验证了漏检和交叉场景。
