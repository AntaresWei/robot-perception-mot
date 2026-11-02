# 迁移跟踪数据目录

- **日期**：2026-09-09 10:00
- **基线 commit**：`2202b54fe5cc85c5932d05fdd5e3ee6c9e5f1122`
- **规模**：12 个文件，+55 / -41
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

将 KITTI Tracking 的四个分拆下载目录集中到 `data/track/`，让配置、Loader、
工具脚本、测试和文档使用同一相对路径，并移除文档中的个人绝对路径。

**原始需求 / prompt：**

```text
将 data_tracking_{calib,image_2,label_2,velodyne} 移入 data/track，更新路径，
尽量使用相对路径，不能出现包含用户名的完整路径。
```

**验收标准：**

- [x] 四个 Tracking 数据目录已从 `data/` 移到 `data/track/`。
- [x] `dataset.root: data`、CLI `--root data` 和直接传入 `data/track` 均能读取数据。
- [x] 文档和命令示例不再包含个人绝对路径。

---

## 2. 改了什么

> 只写关键改动，不要复述 diff。3～6 条为宜。
> 面试时这一节回答「具体怎么做的」。

1. **`src/robot_perception/dataset/kitti.py`**：分拆 Tracking 布局优先从
   `root/track/data_tracking_*` 查找；直接将 root 传为 `data/track` 也保持可用。
2. **`scripts/*.py`**：标定和 YOLO 数据准备入口同步查找新的 `track` 子目录。
3. **`tests/test_kitti.py`**：合成 fixture 改为新的目录层级，并覆盖真实本地 frame 读取。
4. **文档**：数据目录树、PointPillars demo 和 conda 命令改为相对路径。

**改动前后对比（最关键的一处）：**

```diff
+for tracking_root in (root / "track", root):
+    image_dir = tracking_root / "data_tracking_image_2" / split / "image_02"
+    lidar_dir = tracking_root / "data_tracking_velodyne" / split / "velodyne"
```

**验证方式：**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n robot-perception-mot python -m pytest tests/
# 112 passed in 48.34s

conda run -n robot-perception-mot python \
  scripts/inspect_kitti.py --sequence 0000 --frame 100
# image (375, 1242, 3), LiDAR (122624, 4), 7 annotations
```

`scripts/project_lidar_to_image.py --sequence 0000 --frame 100` 成功输出
`/tmp/kitti_0000_000100_projection.png`，投影点数 18995。`git diff --check` 通过。

---

## 3. 为什么这么改

> 这一节是精华，决定面试能不能讲出深度。

- **候选方案**：把 `dataset.root` 改成 `data/track`，或保持 `data` 作为稳定根目录并让
  Loader 查找 `track` 子目录；采用后者，避免现有配置和命令调用方全部需要改根目录。
- **权衡点**：保留 CLI 直接传入 `data/track` 的兼容性；分拆布局会有两个候选根，但标准
  KITTI `training/` 布局仍优先匹配。
- **风险与后续**：本次只迁移 Tracking 数据；`data/train/` 下的 KITTI Object Detection
  训练数据保持独立，不能与 Tracking loader 混用。默认 pytest 因环境自动加载 ROS 插件
  缺少 `lark` 失败，验证使用禁用外部插件自动加载的方式完成，未修改环境。

---

## 4. 我学到了什么（面试可复述）

> 用「STAR + 技术判断」的口径写 2～4 条，每条都要能直接开口讲。
> 模板：当时遇到 X，我判断 Y，于是用 Z 解决，最终结果是 W。

1. **技术点**：数据路径应由配置根目录和目录布局推导，而不是把机器绝对路径写进脚本。
2. **工程判断**：目录调整后先用真实帧、标定投影和完整 pytest 分层验证，避免只靠
   单元 fixture 掩盖真实数据路径错误。
3. **一句话总结（面试开场用）**：我将 50 GB 级 KITTI Tracking 分拆数据重组到
   `data/track`，保持标准 KITTI 与 CLI 使用方式兼容，并用真实 frame 和 112 项测试
   验证了感知数据读取链路。
