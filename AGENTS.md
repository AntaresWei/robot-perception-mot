# Repository Guidelines

## 开始前

先阅读 `PROJECT_SPEC.md`，再检查当前实现、配置和 `git status`。只实现当前任务涉及的模块，不提前开发 Detector、Tracker、Fusion、ROS2 或其他后续功能。不得自行执行 Git commit；提交由开发者 review 后完成。

## 目录与数据

- `src/robot_perception/`：主 Python 包。`common` 放跨模块数据类；`dataset`、`pointcloud`、`detection`、`tracking`、`fusion`、`evaluation`、`visualization` 保持职责分离。
- `configs/`：YAML 配置。数据集根目录、序列、ROI、阈值和协方差等可调参数必须在此声明，禁止在源码中散落 magic number。
- `tests/`：pytest 测试，命名为 `test_<module>.py`。`data/` 和 `outputs/` 分别存放本地数据与生成结果，均不得提交。
- `data/` 当前采用 KITTI 分拆下载布局，如 `track/data_tracking_velodyne/training/velodyne/0000/`；`KittiTrackingDataset` 同时兼容标准 KITTI `training/` 布局。`JMODT-main/` 仅作本地参考，不可作为运行时依赖。

## 环境与代码

使用 conda 环境 `robot-perception-mot`，例如：

```bash
conda run -n robot-perception-mot python -m pytest
```

不要擅自升级 Python、PyTorch、CUDA 或核心依赖，也不要新增依赖。Python 使用 4 空格缩进、`snake_case` 的模块/函数/变量、`PascalCase` 的类，并为公开 API 添加类型标注。跨模块通信使用 dataclass，例如 `FrameData`、`Calibration`、`Detection3D` 和 `TrackState`。坐标系、单位和时间戳必须在 API 中明确，不能隐式转换。

## 测试与验证

新增功能必须同时新增 pytest。Dataset 改动应覆盖合成文件布局和本地 KITTI 小序列；Tracker 改动还应覆盖空输入、漏检、`dt` 异常和生命周期边界。完成后运行相关测试、检查包导入、执行必要的真实数据 demo，并运行 `git diff --check`。测试无法执行时，报告失败原因、当前状态和下一步，不得伪造结果或为通过测试改预期。

## 归档与汇报

完成有实际工程意义的功能、算法、数据流程或实验后，在验证完成后使用：

```bash
bash scripts/codex-log.sh "中文任务标题"
```

标题不超过 20 个中文字符。若变更已提交，参数必须位于标题之前，例如 `bash scripts/codex-log.sh --base HEAD~1 "实现数据加载器"`。补全生成归档的 `README.md`，如实记录需求、关键改动、取舍、验证、限制和可复述结论；不要手改 `changes.patch` 或 `files.txt`。

最终汇报应列出修改文件、实现内容、测试命令及结果、发现的问题和未完成事项。Commit 消息建议使用 `feat:`、`fix:`、`test:` 或 `docs:`，但不要代替开发者执行提交。

## GPU
环境中存在gpu，使用nvidia-smi查看。
