# 实现KITTI数据加载器

- **日期**：2026-09-01 14:48
- **基线 commit**：`8323bafab64e4bcb5fac1b30e2cb1e48e37aef1e`
- **规模**：10 个文件，+465 / -45
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

**任务目标：**参考本地 `JMODT-main` 的 Dataset Loader，为仓库实现对
`data/` 中 KITTI Tracking 图像、LiDAR、标定和标签的读取。

**验收标准：**

- [x] 支持本地 `data_tracking_*` 分拆下载布局。
- [x] 读取训练集首帧的图像、`N x 4` 点云、标定和 `label_02` 标签。
- [x] 支持无标签的 testing split。
- [x] 提供 pytest 单元测试和本地数据集成测试。
- [ ] pytest 实际运行；当前 conda 环境未安装该模块。

---

## 2. 改了什么

1. **`src/robot_perception/dataset/kitti.py`**：新增无 PyTorch 依赖的
   `KittiTrackingDataset`，支持标准 KITTI 和本地分拆下载目录布局。
2. **`src/robot_perception/common/types.py`**：补充完整标定矩阵字段和
   `KittiTrackingAnnotation`，保留 Tracking ID、2D 框、相机坐标与 yaw。
3. **`configs/dataset.yaml`**：将默认根目录配置为本地 `data/`，并指定
   training split、前三个序列及图像/标签加载开关。
4. **`tests/test_kitti.py`**：覆盖分拆布局、无标签测试集和本地 `0000`
   序列的真实读取。
5. **`README.md`**：说明 Loader 的输入布局与输出数据约定；环境说明已
   合并到大写 README，原 `readme.md` 随之删除。

**关键实现：**

```python
points = np.fromfile(lidar_path, dtype=np.float32)
if points.size % 4 != 0:
    raise ValueError("LiDAR file does not contain N x 4 float32 values")
return points.reshape(-1, 4)
```

**验证方式：**

```bash
conda run -n robot-perception-mot python -B -c "... KittiTrackingDataset('data', sequences=['0000']) ..."
```

结果：训练集 `0000` 共 154 帧；首帧图像为 `(375, 1242, 3)`，点云为
`(123397, 4)`，包含 5 条原始标签。testing `0000` 在无标签目录下返回
`annotations=None`。

```bash
conda run -n robot-perception-mot python -B -c "... importlib.import_module(...) ..."
```

结果：11 个项目模块导入通过。

```bash
conda run -n robot-perception-mot python -m pytest
```

结果：失败，`robot-perception-mot` 环境缺少 `pytest`；未安装依赖。

---

## 3. 为什么这么改

- **候选方案**：
  - 直接复用 JMODT 的 `KittiDataset`：放弃，因为它依赖 PyTorch、训练
    配置和训练样本处理，无法作为独立的工程数据边界。
  - 独立 Loader：采用，保留 JMODT 的 LiDAR `float32`/`N x 4` 语义和
    标定读取方式，但只返回原始帧数据。
- **标定解析**：不按文件行号读取。以键名解析 `P0`-`P3`、`R_rect`/
  `R0_rect`、`Tr_velo_cam`/`Tr_velo_to_cam`，兼容本地 KITTI Tracking
  变体，避免格式顺序变化导致静默错误。
- **标签策略**：保留 `DontCare` 和所有原始 Tracking 标签，不在 Loader
  中过滤类别；Detector 或 Tracker 可在后续模块按任务需求筛选。
- **风险与后续**：图像依赖 OpenCV；`data/` 在其他环境通常不存在，真实
  数据集成测试会跳过；尚未实现坐标变换、预处理和时间戳读取。

---

## 4. 我学到了什么

1. **技术点**：KITTI Tracking 的点云文件是连续 `float32` 值；读取后
   必须验证元素数量可被 4 整除，才能安全 reshape 为 `[x, y, z, intensity]`。
2. **工程判断**：本地数据来自多个官方压缩包，目录与单体 KITTI 根目录
   不同，因此 Loader 在输入边界兼容两种布局，避免复制大型数据集。
3. **工程判断**：testing split 缺少标签是数据集语义而非异常，故返回
   `annotations=None`；training split 请求标签但缺目录时才抛出明确错误。
4. **一句话总结（面试开场用）**：我将 JMODT 的数据读取语义抽离为无训练
   框架依赖的 KITTI Tracking Loader，并在真实序列上验证了图像、点云、
   标定和 Tracking 标签的完整读取链路。
