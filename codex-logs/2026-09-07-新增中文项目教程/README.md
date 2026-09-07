# 新增中文项目教程

- **日期**：2026-09-07
- **基线 commit**：`f6791cf`
- **规模**：9 个文件，+751 / -21
- **完整 diff**：[changes.patch](./changes.patch) ｜ [变更清单](./files.txt)

## 1. 需求

为项目增加一份小白可逐步阅读的中文教程，解释从 KITTI 输入到 Detector、Tracker、
Projection、Fusion、视频和指标输出的完整流程，并覆盖坐标系、置信度、数据泄漏、
环境隔离、常见工程坑、ReID 使用时机和面试官可能追问的问题。

## 2. 改了什么

1. 新增 `docs/project_tutorial_zh.md`，共约 433 行，按模块和学习顺序讲解真实代码。
2. 文档明确区分 GT tracking validation、真实 PointPillars MOT、YOLO26 2D 检测、
   Camera-LiDAR projection、object-level fusion 和未来的 EKF/UKF 状态级融合。
3. 加入逐帧数据结构、坐标变换、Track 生命周期、NMS/letterbox、ReID gated fallback、
   视频奇数尺寸、阈值和数据泄漏等实践细节。
4. 增加 2D/3D 框、MOTA 解释、常见问题和面试回答模板；README 增加教程入口。

## 3. 为什么这么改

- 让读者同时看到“数学公式、代码调用链、输入输出 shape 和故障排查”，避免只理解
  算法名而不会运行系统。
- 对尚未实现的 ReID、EKF/UKF、实时 ROS2 做明确边界，防止教程造成能力夸大。
- 示例数字全部来自已生成的实际 JSON/日志，保留真实 PointPillars 的负 MOTA 结果。

## 4. 验证与可复述结论

```bash
/data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python -m pytest tests/
# 106 passed
git diff --check
```

README 与教程中的本地链接均已检查通过，引用的模型、视频和 metrics 文件存在。
教程结论是：一个可靠感知项目必须同时管理数据契约、坐标系、置信度、生命周期、
实验隔离和可复现输出；ReID 等复杂模块应在运动 gating 后作为独立二级证据接入，
而不是破坏 Detector/Tracker/Fusion 解耦。
