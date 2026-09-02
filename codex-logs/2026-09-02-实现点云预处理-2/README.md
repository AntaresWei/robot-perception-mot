# 实现点云预处理

- **日期**：2026-09-02 09:56
- **基线 commit**：`3daf42797d5a7bb900607dfcb977a9bbd99cdb1c`
- **规模**：6 个文件，+493 / -1
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

**任务目标：**实现稳定、可配置、可测试的 LiDAR 点云预处理流水线：
无效点清理、ROI、Voxel、离群点过滤和 RANSAC 地面分割。

**验收标准：**

- [x] 输出各阶段点云、shape 和 `plane_model`。
- [x] 支持 statistical/radius 两种离群点过滤。
- [x] 空点云安全处理，RANSAC 参数来自配置。
- [x] 提供合成点云 pytest 和真实 KITTI frame demo。

---

## 2. 改了什么

1. **`src/robot_perception/pointcloud/preprocess.py`**：实现 invalid removal、
   ROI、voxel averaging、statistical/radius outlier 和完整 pipeline。
2. **`src/robot_perception/pointcloud/ground.py`**：实现 NumPy RANSAC 平面
   拟合，返回 ground/non-ground 点和归一化平面方程。
3. **`scripts/run_pointcloud_preprocess.py`**：读取 YAML、指定 sequence/frame，
   输出点数、各阶段 shape、耗时和 plane model。
4. **`tests/test_pointcloud.py`**：覆盖无效点、ROI、voxel、两种 outlier、
   合成平面和空输入。

**验证方式：**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot-perception-mot \
  python -m pytest tests/
```

结果：`36 passed in 0.43s`。

```bash
conda run -n robot-perception-mot python scripts/run_pointcloud_preprocess.py \
  --sequence 0000 --frame 100
```

结果：Raw `122624`，Outlier 后 `25911`，Ground `10562`，Non-ground
`15349`，处理耗时 `228.05 ms`。

---

## 3. 为什么这么改

- **候选方案**：使用 Open3D 的完整处理链：放弃，因为会把核心逻辑绑定
  到第三方 API；采用 NumPy + SciPy `cKDTree`，便于测试和参数控制。
- **RANSAC**：采用固定 `random_seed` 的 NumPy 实现，保证合成测试与 demo
  可复现，并输出归一化 `[a, b, c, d]` 平面模型。
- **数据契约**：保留 raw、valid、ROI、voxel、outlier、ground 和
  non-ground 全部阶段，便于定位点数异常和评估延迟。
- **风险与后续**：当前地面模型为单平面 RANSAC，不适合复杂坡面；离群点
  参数仍需按不同序列调优。

---

## 4. 我学到了什么（面试可复述）

1. **技术点**：点云处理必须先做有限值过滤，再进入 voxel、邻域查询和
   RANSAC，否则 NaN/Inf 会污染空间索引和距离计算。
2. **工程判断**：每个阶段保留独立数组和 shape，使真实数据上出现点数骤降
   时可以直接定位是 ROI、voxel 还是 outlier 参数造成的。
3. **工程判断**：RANSAC 结果不能只返回 mask，还要返回平面模型和 ground /
   non-ground 数据，后续模块才能验证几何假设。
4. **一句话总结（面试开场用）**：我实现了参数全部来自 YAML、可复现且可观测
   的 LiDAR 预处理链，并在 KITTI `0000/100` 上完成了真实帧验证。
