# ROS2感知跟踪集成

- **日期**：2026-09-03
- **基线 commit**：`3b26b739eba1fbf671953ea7eb5761c4ab682ee8`
- **规模**：14 个文件，+613 / -34
- **完整 diff**：[changes.patch](./changes.patch) ｜ [变更清单](./files.txt)

## 1. 需求：要解决什么问题

将已验证的 KITTI Dataset、Detector Adapter 和 3D Multi-Object Tracker 封装为
ROS2 Humble replay node。要求发布 LiDAR、Image、Detection、Track topic，使用
TF2 与 `base_link`/`lidar`/`camera` frame，保持 Tracker 对 ROS 无依赖，并提供
launch、KITTI 单帧 replay、RViz 和 smoke test。

## 2. 改了什么

1. 新建 `ros2_ws/src/perception_tracker` 的 `ament_python` 包和
   `perception_tracker_node` 入口。
2. 节点从 KITTI replay 帧发布 `PointCloud2`、`Image` 和带 Header 的
   `MarkerArray`；Detection/Track box 与 Track ID 可直接在 RViz 显示。
3. ROS adapter 调用 `GTDetector`/`NoisyGTDetector` 与 `MultiObjectTracker`，核心
   `robot_perception` 包未导入 `rclpy`。
4. 使用 TF2 listener 在 `require_tf=true` 时验证 `base_link <- lidar` 和
   `base_link <- camera`；不写入固定外参或伪造 transform。
5. 添加 RViz 配置、launch 参数、可执行 smoke test 和 pytest 结构检查。
6. 修复归档工具在只读 Git 对象库下遗漏未跟踪文件的问题，并忽略 colcon `log/`。

## 3. 为什么这么改

- **选择 replay-oriented node**：KITTI 没有实时驱动 topic；先复用稳定 Dataset
  Loader 可以验证消息适配、时间戳和可视化，不扩大到传感器驱动重构。
- **选择标准 ROS 消息**：点云和图像使用 `sensor_msgs`，3D box/ID 使用
  `visualization_msgs/MarkerArray`，避免为 MVP 引入接口生成与额外依赖。
- **坐标系取舍**：GT label 在 camera 坐标系，默认 `detection_frame=camera`；
  LiDAR detector 需显式设为 `lidar`。节点不隐式变换对象数据，避免错误可视化。
- **已知限制**：当前 detector 仅支持 GT/NoisyGT replay；真实 PointPillars 可在
  后续作为独立 adapter 传入。容器禁止 FastDDS 创建 UDP socket，因此无法在此环境
  验证跨进程 topic 通信；节点仍完成本进程单帧 replay。

## 4. 验证与可复述结论

```bash
/data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python -m pytest tests/
# 92 passed

source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths ros2_ws
# 1 package finished

/usr/bin/python3 scripts/ros2_smoke_test.py
# perception_tracker package smoke test passed
```

使用 `0000/100` 执行 `ros2 run`，节点输出 `KITTI replay ready` 和
`KITTI replay complete`，并正常退出。面试中可概括为：我把 ROS 与感知核心算法
隔离，先以 KITTI replay 验证可视化和消息边界，再由 TF2 管理真实系统坐标系。
