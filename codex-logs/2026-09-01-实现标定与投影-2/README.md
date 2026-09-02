# 实现标定与投影

- **日期**：2026-09-01 17:51
- **基线 commit**：`da7bfde5d30c31905913d9ec08fb4e3ae8eb01d4`
- **规模**：4 个文件，+437 / -0
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

**任务目标：**解析 KITTI calibration 文件并实现 LiDAR 到相机图像的
向量化投影，输出真实序列的点云投影可视化。

**验收标准：**

- [x] 解析 `P2`、`R0_rect`、`Tr_velo_to_cam`。
- [x] 实现 LiDAR -> Camera -> Rectified Camera -> Image 坐标变换。
- [x] 过滤非有限点、相机后方点和图像 FOV 外点。
- [x] 覆盖矩阵 shape、齐次坐标和图像边界测试。
- [x] 生成 KITTI `0000/100` 投影图片。

---

## 2. 改了什么

1. **`src/robot_perception/dataset/calibration.py`**：新增最小
   `KittiCalibration` 和 parser，仅读取本任务需要的三个矩阵并验证 shape。
2. **`src/robot_perception/common/geometry.py`**：新增 NumPy 向量化的
   `lidar_to_camera`、`camera_to_image`、`lidar_to_image`。
3. **`scripts/project_lidar_to_image.py`**：从 YAML 或 `--root` 获取数据根
   目录，读取指定帧并生成绿色点云叠加图。
4. **`tests/test_calibration.py`**：新增 11 项 parser、变换、过滤与异常测试。

**关键实现：**

```python
camera_points = homogeneous_lidar_points @ Tr_velo_to_cam.T
rectified_points = camera_points @ R0_rect.T
projected = homogeneous_rectified_points @ P2.T
pixels = projected[:, :2] / projected[:, 2:3]
```

**验证方式：**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot-perception-mot \
  python -m pytest tests/
```

结果：`30 passed in 1.12s`。

```bash
conda run -n robot-perception-mot python \
  scripts/project_lidar_to_image.py --sequence 0000 --frame 100
```

结果：生成 `outputs/visualizations/kitti_0000_000100.png`，尺寸为
`1242 x 375`，投影了 18,995 个 LiDAR 点。

---

## 3. 为什么这么改

- **候选方案**：
  - 在 Dataset Loader 内解析 calibration：放弃，因为会把原始数据读取和
    坐标变换职责耦合，且超出最小 Loader 的任务边界。
  - 独立 parser 和 geometry 模块：采用，便于单独测试矩阵和后续复用。
- **齐次坐标**：将 `N x 3` 坐标扩展为 `N x 4` 后右乘 `3 x 4` 变换矩阵的
  转置；所有点一次性处理，不使用逐点 Python 循环。
- **过滤策略**：投影前剔除非有限输入，使用 rectified camera depth 过滤
  `z <= 0` 点，再按 `[0, width)`、`[0, height)` 过滤像素坐标。
- **风险与后续**：可视化绘制阶段因 OpenCV API 需要逐点调用 `circle`；
  数学变换仍保持向量化。直接运行 pytest 会加载系统 ROS 插件并因缺少
  `lark` 失败，未修改环境，故使用禁用外部插件自动加载的命令验证项目测试。

---

## 4. 我学到了什么（面试可复述）

1. **技术点**：LiDAR 到 KITTI 图像不是单一外参变换，必须依次经过
   `Tr_velo_to_cam`、`R0_rect`、`P2`，每一步都使用齐次坐标。
2. **工程判断**：将标定解析独立于 Dataset Loader，可避免数据 I/O 模块
   依赖几何逻辑，并让矩阵 shape 与投影边界更容易进行单元测试。
3. **工程判断**：正深度和 FOV 过滤要在投影链路中显式完成；否则相机后方
   或无穷远点会得到无意义像素坐标并污染可视化。
4. **一句话总结（面试开场用）**：我实现了 KITTI LiDAR 到相机图像的完整
   向量化投影链路，并在真实 `0000/100` 序列中投影 18,995 个有效点进行验证。
