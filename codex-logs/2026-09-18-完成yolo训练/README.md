# 完成YOLO训练

- **日期**：2026-09-18 14:55
- **基线 commit**：`2202b54fe5cc85c5932d05fdd5e3ee6c9e5f1122`
- **规模**：20 个文件，+516 / -43
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

> 使用 KITTI Tracking 图像和三类 2D 标注训练 YOLO26，并要求训练计算量达到至少 70 轮。

**原始需求 / prompt：**

```text
使用 `robot_2` 环境中的 Ultralytics，不修改该环境；指定空闲 GPU 1，训练 `Car / Pedestrian / Cyclist`。
```

**验收标准：**

- [x] KITTI 数据集完成转换并检查图像链接、类别和标注数量。
- [x] 完成第一阶段 46 轮和第二阶段 80 轮训练，累计 126 轮训练计算。
- [x] 根据验证集 mAP50-95 选择最终权重。

---

## 2. 改了什么

> 只写关键改动，不要复述 diff。3～6 条为宜。
> 面试时这一节回答「具体怎么做的」。

1. **`configs/yolo26_train.yaml`**：默认训练轮数设为 80，早停耐心值设为 20。
2. **训练数据**：重新生成迁移后有效的 KITTI 图像软链接，训练集 5,628 帧、验证集 2,380 帧，共 45,198 个三类框。
3. **训练方式**：`robot_2` 提供 Ultralytics；通过临时 `PYTHONPATH` 使用项目已有的 PyTorch 2.1.2+cu118，以兼容主机 CUDA 12.8 驱动，不改 conda 环境。
4. **权重选择**：比较第一阶段早停模型和第二阶段完整 80 轮模型，最终保留验证指标更高的第一阶段 best。

**改动前后对比（最关键的一处）：**

```diff
（从 changes.patch 里摘出最有代表性的 3～15 行）
```

**验证方式：**

```bash
```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/data/weijiaman/miniconda3/envs/robot-perception-mot/lib/python3.10/site-packages \
/data/weijiaman/miniconda3/envs/robot_2/bin/python \
scripts/train_yolo26_kitti.py --config configs/yolo26_train.yaml --epochs 80 --device 0
```

第一阶段最佳 epoch 26：mAP50 `0.60325`，mAP50-95 `0.36409`，Precision `0.68404`，Recall `0.54489`。
第二阶段完整 80 轮最佳 epoch 66：mAP50-95 `0.25958`。最终权重为 `weights/yolo26s_kitti3class.pt`。
```

---

## 3. 为什么这么改

> 这一节是精华，决定面试能不能讲出深度。

- **候选方案**：直接在 `robot_2` 的 cu130 PyTorch 上训练，或复用项目已有 cu118 PyTorch。前者与主机驱动不兼容，后者只通过环境变量临时注入，保持环境不变。
- **权衡点**：第二阶段达到完整 80 轮，但重置优化器后验证性能下降；因此按验证指标选模型，而不是按训练轮数或最后一个 checkpoint 选模型。
- **风险与后续**：当前验证按序列切分，可能存在场景分布差异；Cyclist 指标明显低于 Car/Pedestrian，后续应增加类别均衡和遮挡分层评估。

---

## 4. 我学到了什么（面试可复述）

> 用「STAR + 技术判断」的口径写 2～4 条，每条都要能直接开口讲。
> 模板：当时遇到 X，我判断 Y，于是用 Z 解决，最终结果是 W。

1. **技术点**：训练前先检查迁移后软链接，避免数据扫描成功但实际图像不可读。
2. **工程判断**：训练轮数和模型质量不是同一指标；完整 80 轮模型低于早停模型时，应保留验证集更优 checkpoint。
3. **一句话总结（面试开场用）**：我在不改动既有 conda 环境的前提下完成了 KITTI 三类 YOLO26 训练，累计 126 轮 GPU 计算，最终保留 mAP50 `0.60325`、mAP50-95 `0.36409` 的验证最优权重。
