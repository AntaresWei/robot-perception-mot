# Repository Guidelines

## 项目结构

本仓库实现基于 KITTI Tracking 数据集的 Camera-LiDAR 三维感知与多目标跟踪系统。目前仓库以 `PROJECT_SPEC.md` 中的项目规格和 `readme.md` 中的环境说明为主；后续代码应遵循以下目录约定：

- `src/robot_perception/`：可复用的 Python 包，按 `dataset`、`pointcloud`、`detection`、`tracking`、`fusion`、`evaluation` 和 `visualization` 分层。
- `scripts/`：可执行入口，例如 `run_tracking.py`、`evaluate.py`。
- `tests/`：pytest 单元测试和流水线集成测试。
- `configs/`：数据集、预处理和 Tracker 参数的 YAML 配置。
- `data/` 与 `outputs/`：本地 KITTI 数据、日志、指标和可视化结果，禁止提交。

Detector Adapter 不得依赖 Tracker 内部实现。模块间使用明确的数据类通信，例如 `FrameData`、`Detection3D`、`TrackState`。

## 环境与命令

目标环境为 Ubuntu 22.04、Python 3.10、PyTorch 2.1.2 + CUDA 11.8；涉及 ROS 节点时使用 ROS2 Humble。按 `readme.md` 中的命令创建环境。新增依赖文件后，使用 `pip install -r requirements.txt` 或 `conda env create -f environment.yml` 安装。

在仓库根目录运行以下计划中的检查命令：

```bash
pytest
pytest tests/test_tracker.py
python scripts/run_tracking.py --config configs/tracker.yaml
```

随着脚本落地，最后一条命令必须保持为可复现的端到端执行入口。

## 代码风格与配置

Python 使用 4 空格缩进；模块、函数和变量使用 `snake_case`；类使用 `PascalCase`；公开 API 应提供类型标注。模块应小而职责单一，NumPy 数组应校验 shape。坐标系必须在命名和 API 中明确表达，不得使用 magic number 隐式完成 LiDAR-to-Camera 变换或时间处理。

阈值、协方差、ROI 边界和数据集路径必须放在 `configs/` 下的 YAML 文件中，不要散落在源码里。

## 测试要求

使用 pytest，测试文件命名为 `test_<module>.py`。覆盖正常流程和边界情况：空输入、未匹配目标、漏检、大 `dt`、NaN 和奇异协方差。修改 Tracker 时，应测试 Kalman Filter 预测与更新、gating、Hungarian association、生命周期状态转换和单调递增的 track ID。修改流水线边界时，补充小规模 KITTI frame 集成测试。

## 提交与 Pull Request

遵循现有 Conventional Commit 风格，例如 `feat: implement kalman filter`、`test: add tracker unit tests`、`docs: add experiment report`。提交应小而聚焦，使用清晰的祈使式说明；禁止 `update`、`final` 等模糊标题。

Pull Request 需说明影响的流水线阶段、所用配置和数据序列、验证命令及实际指标。涉及投影或跟踪输出时附可视化结果，并关联相关 issue 或规格章节。基于 GT 的跟踪实验仅用于验证 Tracker 实现，不得表述为 Detector 性能。


---




# AI Coding Change Log

本项目使用 `codex-logs/` 记录具有实际工程意义的 AI 辅助开发任务。

目标不是记录所有代码 diff，而是沉淀：

- 技术决策
- 算法实现
- 验证过程
- 失败原因
- 方案取舍
- 实验结果
- 面试可复述知识

---

## 1. 什么时候必须归档

当本轮工作完成以下任意一项时，必须创建一条 Change Log：

- 新增一个功能模块
- 修改核心算法
- 修改数据处理流程
- 修改 Tracking 策略
- 修改 Sensor Fusion 逻辑
- 修复影响功能的 Bug
- 完成一个可独立验收的工程任务
- 完成一次具有意义的实验

例如：

```text
Implement Kalman Filter
Implement Hungarian Association
Add KITTI Calibration
Add Ground Segmentation
Fix coordinate transformation
Compare Euclidean vs Mahalanobis Association
````

---

## 2. 以下情况不需要归档

以下修改不单独创建日志：

- 拼写错误
- 格式化
- import 排序
- 注释修改
- 简单变量重命名
- 单纯依赖版本调整
- 同一个任务中的连续小修复

这些修改应该归属于当前正在进行的任务。

---

## 3. 完成任务后的归档流程

在确认代码修改完成并通过验证后执行：

```
bash scripts/codex-log.sh "中文任务标题"
```

标题：

- 不超过 20 个中文字符
- 描述实际完成的工作
- 不使用 `final`、`update`、`fix` 等无意义名称

例如：

```
bash scripts/codex-log.sh "实现卡尔曼滤波器"
```

---

## 4. 日志目录

每条日志：

```
codex-logs/
└── YYYYMMDD-HHMM-任务标题/
    ├── README.md
    ├── changes.patch
    └── files.txt
```

其中：

### README.md

记录：

1. 需求
2. 改动
3. 技术决策
4. 验证
5. 实验结果
6. 学习总结

### changes.patch

自动生成。

禁止手动修改。

### files.txt

自动生成。

禁止手动修改。

---

# 5. README.md 内容规范

## 5.1 需求

不要机械复制整个 Codex prompt。

只记录：

```
任务目标：
验收标准：
相关模块：
```

如果需要保留原始需求，可以记录到：

```
docs/tasks/
```

而不是每个 Change Log 都复制一遍。

---

## 5.2 改了什么

列出 3～6 个关键改动。

必须注明：

```
文件路径
核心变化
```

例如：

```
- src/robot_perception/tracking/kalman.py
  实现 Constant Velocity Kalman Filter。

- tests/test_kalman.py
  增加 predict/update 数值验证。

- configs/tracker.yaml
  增加 Q/R 参数配置。
```

不要简单复制 diff。

---

## 5.3 技术决策

必须回答：

```
为什么这么做？
为什么没有选择其他方案？
这个方案的优缺点是什么？
```

至少记录一个被放弃的方案。

例如：

```
候选方案：
直接使用欧氏距离进行 Detection-Track Association。

放弃原因：
欧氏距离无法利用 Kalman covariance 表示的不确定性，
在不同目标状态不确定程度下缺乏统一统计意义。

最终方案：
Mahalanobis Distance + Hungarian Assignment。
```

---

## 5.4 验证

必须记录实际执行过的命令。

例如：

```
pytest tests/test_kalman.py -v
```

结果：

```
12 passed
```

禁止编造。

如果没有验证：

```
状态：待验证
```

---

## 5.5 实验结果

如果任务涉及算法实验，必须记录：

```
Dataset
Sequence
Config
Baseline
Method
Metric
Result
```

例如：

|Method|MOTA|IDF1|IDS|
|---|---|---|---|
|Euclidean|...|...|...|
|Mahalanobis|...|...|...|

没有实际数据时禁止填写数字。

---

## 5.6 已知问题

必须记录：

```
Known Limitations
TODO
Potential Risks
```

例如：

```
- 当前状态模型采用 Constant Velocity。
- 尚未处理长时间遮挡。
- 当前 Fusion 仍属于 object-level association。
```

---

## 5.7 我学到了什么

写 2～4 条真正可以在面试中复述的内容。

推荐格式：

> 遇到 X → 判断 Y → 使用 Z → 得到 W

例如：

> 点云投影存在系统性偏移 → 判断坐标变换链存在问题 → 检查 LiDAR→Camera 外参与 rectification 顺序 → 修正后投影与目标位置一致。

最后增加：

```
面试总结：
一句话描述这个任务中最重要的技术决策。
```

---

# 6. Codex 工作要求

完成一个具有实际意义的任务后：

1. 运行相关测试。
2. 检查 git diff。
3. 检查是否存在未预期修改。
4. 运行必要的 integration test / demo。
5. 生成 Change Log。
6. 更新实验记录。
7. 最后才向用户汇报。

如果测试失败：

```
不得生成“已完成”的总结。
```

必须明确：

```
失败原因
当前状态
下一步建议
```

---

# 7. Git 与 Change Log 的关系

Change Log 不替代 Git。

Git 负责：

```
版本控制
回滚
分支
代码历史
```

Change Log 负责：

```
为什么改
怎么验证
为什么选择这个方案
学到了什么
```

一个完成的任务应该尽量形成：

```
Code
+
Test
+
Git Commit
+
Change Log
```

---

# 8. Commit 规范

推荐：

```
feat: implement kalman filter
feat: add kitti calibration
feat: add mahalanobis gating
feat: implement hungarian association
fix: correct lidar camera transform
test: add tracker lifecycle tests
docs: add tracking experiment results
```

禁止：

```
update
test
fix
aaa
final
final2
final_final
```

---

# 9. 实验日志

具有比较意义的实验必须独立记录。

例如：

```
docs/experiments/
├── 001_kalman_filter.md
├── 002_association.md
└── 003_tracker_ablation.md
```

实验日志至少包含：

```
Hypothesis
Dataset
Configuration
Baseline
Method
Metrics
Results
Analysis
Conclusion
```

---

# 10. AI Coding 原则

Codex 是开发辅助工具，不是项目决策者。

Codex：

- 可以实现代码。
- 可以运行测试。
- 可以分析错误。
- 可以提出方案。

但最终：

- 算法选择
- 实验设计
- 指标解释
- 技术取舍

必须由开发者确认。

禁止：

- 编造测试结果。
- 编造 benchmark。
- 未运行代码就声称成功。
- 修改测试来掩盖错误。
- 删除失败实验。
- 大规模无理由重构。
- 擅自修改核心依赖。
- 使用硬编码绝对路径。