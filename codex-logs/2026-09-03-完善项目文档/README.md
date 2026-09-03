# 完善项目文档

- **日期**：2026-09-03
- **基线 commit**：`38c0a143f638422118a17a76063e16dfb085a9da`
- **规模**：2 个文件，+337 / -235
- **完整 diff**：[changes.patch](./changes.patch) ｜ [变更清单](./files.txt)

## 1. 需求：要解决什么问题

将 README 整理为中文贡献/使用文档，覆盖项目概览、动机、架构、数据集、环境安装、
快速开始、所有已实现算法模块、ROS2、benchmark、结果、可视化、限制和后续工作。
文档必须使用真实 `outputs/metrics` 数值，并明确区分 GT tracking validation、真实
PointPillars 单帧实验、Camera-LiDAR projection 和 object-level fusion。

## 2. 改了什么

1. 完整重写 `README.md`，按 21 个章节组织中文说明。
2. 增加 Mermaid 系统架构图、端到端 pipeline 图和点云处理流程图。
3. 补充 KITTI 两种目录布局、主项目/PointPillars 隔离环境、checkpoint 下载与版本
   验证命令、ROS2 launch、benchmark 与 MP4 可视化命令。
4. 从 `outputs/metrics/baseline_*.json` 同步 0000 全 154 帧实际 MOTA、IDF1、IDS、
   FP、FN、FPS、latency 和 RSS，并引用已生成图像/视频。
5. 说明当前真实 detector 只有 PointPillars 单帧接入验证；没有虚构全序列 detector
   或 GPU 指标。

## 3. 为什么这么改

- **文档结构**：使用用户指定的 21 个标题，方便复现、面试讲解和定位限制。
- **结果边界**：benchmark 使用 GT input，结果只证明 tracking pipeline；真实
  PointPillars 与 SyntheticCameraDetector 的能力分别单独标注，避免夸大。
- **可复现性**：命令统一从仓库根目录运行，并保留主环境与旧版 MMDetection3D 的
  精确版本组合；输出目录明确为本地生成物，不提交数据/权重。

## 4. 验证与可复述结论

已实际执行 README Quick Start：

```bash
python scripts/inspect_kitti.py --sequence 0000 --frame 100
python scripts/run_pointcloud_preprocess.py --sequence 0000 --frame 100
python scripts/run_fusion_demo.py --sequence 0000 --frame 100
python scripts/evaluate.py --config configs/pipeline.yaml --sequence 0000
python -m pytest tests/
# 101 passed
```

另外检查了 PointPillars 配置/checkpoint 文件、ROS2 Humble/RViz2 可用性、README
章节和链接。核心算法未修改。文档的关键结论是：当前工程已形成可复现的 KITTI
tracking、几何投影、实验性融合和 ROS2 replay 链路，但真实 Detector 全序列指标、
真实 Camera detector 与实时传感器集成仍属于后续工作。
