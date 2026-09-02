# 实现常速度卡尔曼滤波

- **日期**：2026-09-02 11:35
- **基线 commit**：`8c38442f7ee8d8b35297879587dd63b5e653ccaa`
- **规模**：5 个文件，+396 / -1
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

**任务目标：**独立实现 3D Constant Velocity Kalman Filter，估计
`[px, py, pz, vx, vy, vz]`，支持 variable `dt`、可配置 Q/R、数值稳定
协方差和合成轨迹评估。

**验收标准：**

- [x] 实现 `predict()` 和 `update()`，不依赖 filterpy 核心逻辑。
- [x] 检查矩阵/向量 shape，并明确拒绝 NaN/Inf 和非法 `dt`。
- [x] 使用 Joseph covariance update 并保持协方差对称正定。
- [x] 添加单元测试和 `kalman_baseline.json` 合成实验输出。

---

## 2. 改了什么

1. **`src/robot_perception/tracking/kalman.py`**：新增自研 6D 状态/3D
   观测 Kalman Filter，动态构造 `F(dt)`，使用 `np.linalg.solve` 更新。
2. **`configs/kalman.yaml`**：集中配置 Q、R、初始协方差和合成轨迹参数。
3. **`tests/test_kalman.py`**：覆盖初始化、预测、更新、variable `dt`、
   covariance、噪声 RMSE 和非法输入。
4. **`scripts/run_kalman_demo.py`**：生成 constant-velocity truth 和高斯
   噪声 measurement，比较 raw 与 filtered RMSE 并写入 JSON。

**验证方式：**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot-perception-mot \
  python -m pytest tests/
```

结果：`47 passed in 0.61s`。

```bash
conda run -n robot-perception-mot python scripts/run_kalman_demo.py
```

结果：`raw_measurement_rmse=0.4658344540`，
`kalman_filtered_rmse=0.3706396668`，改善 `0.0951947873`，输出文件为
`outputs/metrics/kalman_baseline.json`。

---

## 3. 为什么这么改

- **候选方案**：直接依赖 filterpy：放弃，因为本项目需要拥有可解释、可
  单测的核心数学逻辑；最终采用纯 NumPy 实现。
- **数值稳定性**：更新使用 Joseph 形式，之后强制对称化并将协方差特征值
  截断到 `covariance_floor`，降低浮点误差导致的非正定风险。
- **调试过程**：首次测试发现 `solve(S, PH^T)` 维度方向错误；修正为对
  `(PH^T).T` 求解后转置，全部测试和 demo 通过。
- **风险与后续**：当前过程噪声 Q 使用固定矩阵，后续可按 `dt` 构造连续
  白噪声模型；本任务不包含关联或 Track 生命周期。

---

## 4. 我学到了什么

1. **技术点**：Kalman 增益满足 `K = P H^T S^-1`；实现时通过解线性方程
   代替显式求逆，并注意右端矩阵方向以满足 NumPy shape。
2. **工程判断**：variable `dt` 必须每次预测动态更新状态转移矩阵，否则
   不规则传感器时间间隔会直接造成位置预测偏差。
3. **工程判断**：协方差不仅要数值有限，还要保持对称正定；Joseph 形式和
   特征值 floor 同时覆盖模型误差与浮点舍入风险。
4. **一句话总结（面试开场用）**：我实现了不依赖第三方 KF 核心逻辑的 3D
   常速度滤波器，并用 100 步合成轨迹将 RMSE 从 0.4658 降到 0.3706。
