# 生成标准MOT指标

- **日期**：2026-09-07 22:02
- **基线 commit**：`3e051e082a6bcf5c91b2d8ddef2f2f1264453e28`
- **规模**：5 个文件，+357 / -22
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

将已有的 PointPillars -> 3D Tracker 评估从逐帧 IDF1 近似升级为基于 KITTI
`label_02` `track_id` 的标准全局 identity MOT 指标，并在不同场景上实际运行。

**原始需求 / prompt：**

```text
生成基于 KITTI GT identity 的标准 MOT 指标。
```

**验收标准：**

- [x] 使用 KITTI GT `track_id` 计算全局 MOTA、IDF1、IDP、IDR、IDS、FP、FN。
- [x] 真实运行 PointPillars + Tracker 的 `0000`（154 帧）和 `0001`（443 帧）。
- [x] 保留旧逐帧结果并在 JSON 中区分，避免指标含义混淆。

---

## 2. 改了什么

> 只写关键改动，不要复述 diff。3～6 条为宜。
> 面试时这一节回答「具体怎么做的」。

1. **`scripts/evaluate_real_mot.py`**：加入 MOTAccumulator 标准评估；旧版 detector
   环境缺少 `motmetrics` 时使用 SciPy Hungarian global-ID fallback。
2. **`tests/test_real_evaluation.py`**：增加稳定 identity、identity switch 和类别门控测试。
3. **`README.md`、`docs/project_tutorial_zh.md`、`docs/experiments.md`**：补充标准
   指标定义、命令、结果和限制。
4. 生成真实输出：`outputs/metrics/real_pointpillars_mot_0000_standard.json` 与
   `outputs/metrics/real_pointpillars_mot_0001_standard.json`。

**改动前后对比（最关键的一处）：**

```diff
+"metric_definition": "MOTChallenge global identity metrics with global ID assignment",
+"legacy_framewise": { ... },
```

**验证方式：**

```bash
/data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python -m pytest tests/
# 112 passed in 3.11s

/data/weijiaman/miniconda3/envs/mmdet3d-kitti/bin/python scripts/evaluate_real_mot.py \
  --sequence 0000 --device cpu \
  --output outputs/metrics/real_pointpillars_mot_0000_standard.json

/data/weijiaman/miniconda3/envs/mmdet3d-kitti/bin/python scripts/evaluate_real_mot.py \
  --sequence 0001 --device cpu \
  --output outputs/metrics/real_pointpillars_mot_0001_standard.json
```

真实结果：`0000` MOTA `-0.4585`、标准 IDF1 `0.4010`、IDS `8`、FP `766`、FN `263`；
`0001` MOTA `0.1700`、标准 IDF1 `0.6395`、IDS `4`、FP `1799`、FN `682`。

---

## 3. 为什么这么改

> 这一节是精华，决定面试能不能讲出深度。

- **候选方案**：只保留逐帧计数实现简单但不能反映全局 ID 一致性；采用 MOTAccumulator
  定义，并为旧版 MMDetection3D 环境提供等价 SciPy fallback。
- **权衡点**：不改动 detector 环境或安装依赖，保证 PointPillars 可运行；fallback 的
  `num_fragmentations` 暂未统计，但 MOTA/IDF1/IDP/IDR/IDS/FP/FN 已按全局 identity
  assignment 生成。
- **风险与后续**：当前评估是三维中心距离匹配，不是 KITTI 官方 3D AP；还需补充按类别、
  遮挡级别和 fragmentation/HOTA 的分析，并恢复 GPU 后重新记录运行性能。

---

## 4. 我学到了什么（面试可复述）

> 用「STAR + 技术判断」的口径写 2～4 条，每条都要能直接开口讲。
> 模板：当时遇到 X，我判断 Y，于是用 Z 解决，最终结果是 W。

1. **技术点**：标准 IDF1 先进行帧级空间匹配，再基于所有帧的 GT/prediction ID 共现
   做全局一对一映射；因此同一目标发生 ID 切换会降低 IDF1，而不只是增加局部计数。
2. **工程判断**：将旧逐帧值保存在 `legacy_framewise`，顶层字段改为标准值，避免
   README 和 JSON 同时出现两个无法区分的 IDF1。
3. **一句话总结（面试开场用）**：我用 KITTI `track_id` 对真实 PointPillars 轨迹做了
   两个场景的标准 identity MOT 评估，得到 `0000` IDF1 `0.4010`、`0001` IDF1 `0.6395`，
   并保留了可复现配置、帧范围和运行环境信息。
