# PointPillars训练验证

- **日期**：2026-09-18 10:59
- **基线 commit**：`2202b54fe5cc85c5932d05fdd5e3ee6c9e5f1122`
- **规模**：19 个文件，+513 / -41
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

> 使用 KITTI 训练集对 PointPillars 做微调，并以 KITTI validation AP 选择最终权重；要求不修改 CUDA/PyTorch 环境、不伪造指标。

**原始需求 / prompt：**

```text
根据 `docs/KITTI_train.md` 准备 KITTI 数据、检查 GPU、增加微调 epoch，并验证训练准确率。
```

**验收标准：**

- [x] 生成训练/验证信息文件和 GT database。
- [x] 比较原始 checkpoint、普通微调和 DB sampler 微调的真实 AP。
- [x] 记录 FOV-reduced 点云基线，并保留更优权重。

---

## 2. 改了什么

> 只写关键改动，不要复述 diff。3～6 条为宜。
> 面试时这一节回答「具体怎么做的」。

1. **`scripts/prepare_kitti_reduced.py`**：按 KITTI 相机 FOV 生成 `velodyne_reduced`，不修改原始点云。
2. **`configs/pointpillars_kitti_*.py`**：提供普通微调、DB sampler、续训和 reduced-point 配置。
3. **`scripts/evaluate_pointpillars_kitti.py`**：复用 MMDetection3D KITTI evaluator，并用 Shapely CPU BEV IoU 绕过旧 Numba/CUDA 兼容问题。
4. **训练产物**：保留各 epoch checkpoint 与 validation 输出，原始 checkpoint 未被覆盖。

**改动前后对比（最关键的一处）：**

```diff
```text
原始 velodyne 3D AP40 moderate: 61.7238
FOV-reduced velodyne 3D AP40 moderate: 64.3576
普通微调最佳约: 58.29
DB sampler 微调最佳约: 53.62
```
```

**验证方式：**

```bash
```bash
python scripts/prepare_kitti_reduced.py
tools/test.py ... --out outputs/pointpillars_reduced_original_val.pkl
python scripts/evaluate_pointpillars_kitti.py \
  --config configs/pointpillars_kitti_reduced_dbsampler.py \
  --predictions outputs/pointpillars_reduced_original_val.pkl
```

结果：3769 个 KITTI validation 帧完成推理；reduced-point baseline AP40 moderate 为 64.3576%。归档时 GPU 驱动已不可通信，未继续启动新训练。
```

---

## 3. 为什么这么改

> 这一节是精华，决定面试能不能讲出深度。

- **候选方案**：直接继续增加 epoch、使用 DB sampler，或先对齐官方 reduced-point 输入。前两者均导致 AP 下降，因此采用 reduced-point 原始 checkpoint 作为生产基线。
- **权衡点**：评估脚本增加 CPU Shapely 计算，速度较慢，但避免旧 Numba 与主机 CUDA 12.2 的 NVVM 冲突，指标仍来自官方 KITTI 评估逻辑。
- **风险与后续**：当前没有在 GPU 恢复后重新训练 reduced + DB sampler；微调模型未超过 64.3576%，不可替换官方权重。

---

## 4. 我学到了什么（面试可复述）

> 用「STAR + 技术判断」的口径写 2～4 条，每条都要能直接开口讲。
> 模板：当时遇到 X，我判断 Y，于是用 Z 解决，最终结果是 W。

1. **技术点**：PointPillars 的输入分布包含相机 FOV reduction；仅用原始点云微调会把数据分布和官方 checkpoint 错开。
2. **工程判断**：训练是否“更好”必须由同一 validation split 的真实 AP 决定，不能因为 epoch 增加就替换基线。
3. **一句话总结（面试开场用）**：我复现了 KITTI PointPillars 验证链路，发现对齐 reduced 点云后 Overall 3D AP40 moderate 从 61.72% 提升到 64.36%，并据此保留可复现的最佳权重。
