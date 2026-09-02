# 实现数据关联

- **日期**：2026-09-02 11:58
- **基线 commit**：`1101f4927f00ff341c946239ae74cd6cf57bdcff`
- **规模**：5 个文件，+497 / -1
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

**任务目标：**实现 Detection-to-Track 的 Mahalanobis Gating、Cost Matrix
和 Hungarian 一对一关联，并保留 Euclidean nearest-neighbor baseline。

**验收标准：**

- [x] 使用稳定线性方程求解计算 `d² = y.T @ S^-1 @ y`。
- [x] 支持空输入、数量不等、全无效 pair 和 invalid pair 防误匹配。
- [x] API 返回 `matches`、`unmatched_tracks`、`unmatched_detections`。
- [x] 添加 gating、association 和 Euclidean baseline 测试。

---

## 2. 改了什么

1. **`src/robot_perception/tracking/gating.py`**：新增 Mahalanobis 距离、
   pairwise distance matrix 和 threshold mask，支持共享或逐 track 协方差。
2. **`src/robot_perception/tracking/association.py`**：新增带有效掩码的
   Hungarian association、Kalman state association API 和 Euclidean baseline。
3. **`tests/test_gating.py`**：覆盖近距离、远距离和 covariance effect。
4. **`tests/test_association.py`**：覆盖 perfect/ambiguous matching、尺寸不等、
   空输入、gating、全无效 pair 和 baseline。

**关键实现：**

```python
solved = np.linalg.solve(covariance, innovation)
distance = innovation @ solved
```

**验证方式：**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot-perception-mot \
  python -m pytest tests/
```

结果：`59 passed in 1.58s`。

```bash
conda run -n robot-perception-mot python -B -c "... associate_detections_to_tracks(...) ..."
```

结果：两条 track 与两条 detection 正确返回 `[[0, 0], [1, 1]]`，无未匹配项。

---

## 3. 为什么这么改

- **候选方案**：直接使用欧氏距离：保留为 baseline，但不作为主方法，因为
  它没有利用每条 track 的状态不确定性。
- **主方案**：先构造创新协方差，再用 `np.linalg.solve` 计算 Mahalanobis
  距离；避免显式矩阵求逆并保留统计意义。
- **invalid pair**：使用有效掩码和大代价送入 Hungarian，随后再次过滤分配
  结果；当所有 pair 被 gating 时直接返回全 unmatched。
- **风险与后续**：当前 association 接收已预测的 state/covariance，不负责
  调用 `predict()`，也不包含 Track lifecycle 或 ID 管理。

---

## 4. 我学到了什么（面试可复述）

1. **技术点**：Mahalanobis distance 的关键不是公式本身，而是正确构造
   `S = HPH.T + R` 并用线性求解替代 `inv(S)`。
2. **工程判断**：Hungarian 会强制分配方阵中的 pair，因此 invalid pair
   不能只设置大代价，还必须在结果中用有效掩码过滤。
3. **工程判断**：空 track、空 detection 和全 gating 场景应在进入优化器前
   直接返回，避免库函数对空矩阵或全无穷代价行为不一致。
4. **一句话总结（面试开场用）**：我实现了带 Mahalanobis Gating 的稳定
   Hungarian 一对一关联，并用 59 项测试覆盖了正常、歧义和边界输入。
