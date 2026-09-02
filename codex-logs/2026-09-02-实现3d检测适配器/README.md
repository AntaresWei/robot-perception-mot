# 实现3D检测适配器

- **日期**：2026-09-02 14:07
- **基线 commit**：`8a8f86a295a491133eb67fddb4b25a864093460e`
- **规模**：7 个文件，+439 / -1
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

**任务目标：**定义与 Tracker 解耦的 `BaseDetector`，将 KITTI GT 和
可控噪声 GT 转换成统一的 `Detection3D`，为后续 Tracking pipeline 提供输入。

**验收标准：**

- [x] 定义 `predict(frame: FrameData) -> list[Detection3D]` 接口。
- [x] 实现 class filtering、score threshold、GT 与 NoisyGT adapter。
- [x] 提供单帧检测统计 CLI 与 6 项 adapter 测试。
- [x] 在 README 明确 GT 实验只验证 Tracker，不报告 detector performance。

---

## 2. 改了什么

1. **`src/robot_perception/detection/base.py`**：新增抽象接口和 KITTI
   类别到稳定 class ID 的映射。
2. **`src/robot_perception/detection/adapter.py`**：新增 `GTDetector` 与
   `NoisyGTDetector`，输出统一 `position/dimensions/yaw/score/class_id`。
3. **`configs/detector.yaml`**：配置 detector 类型、过滤阈值、类别和噪声。
4. **`scripts/run_detection.py`**：读取指定 frame，输出检测数量、类别分布和
   score 统计。
5. **`tests/test_detector_adapter.py`**：覆盖接口、类别过滤、阈值、无标签和
   噪声可复现性。

**验证方式：**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot-perception-mot \
  python -m pytest tests/
```

结果：`65 passed in 5.60s`。

```bash
conda run -n robot-perception-mot python scripts/run_detection.py \
  --sequence 0000 --frame 100
```

结果：`0000/000100` 输出 3 条检测，类别为 2 个 Car、1 个 Cyclist，GT score
均为 1.0。

---

## 3. 为什么这么改

- **候选方案**：直接让 Tracker 读取 `KittiTrackingAnnotation`：放弃，因为
  会把 Tracker 与 Dataset、KITTI 标签格式耦合。
- **最终方案**：以 `BaseDetector` 和 `Detection3D` 作为边界；GT、噪声 GT
  或后续真实 detector 都只需实现同一 `predict` 方法。
- **NoisyGT**：使用固定 seed 的随机生成器，将位置、尺寸、yaw、score 的
  噪声放在 YAML 中；尺寸强制为正，score 限制为 `[0, 1]`。
- **未实现 RealDetectorAdapter**：当前环境没有已验证、可稳定运行的公开
  3D Detector 接口。按任务限制保留 GT/NoisyGT，未投入研究仓库环境适配。
- **GT 限制**：GT-based tracking experiment is used only for validating the
  tracking implementation and is not reported as detector performance.

---

## 4. 我学到了什么（面试可复述）

1. **技术点**：Adapter 的价值是把不同 detector 输出规约为同一个
   `Detection3D`，使 Kalman、gating 和 association 无需依赖模型格式。
2. **工程判断**：噪声应可配置且可复现，才能让 Tracking 基线实验可重复，
   并且不把随机性隐藏在算法内部。
3. **工程判断**：将 `DontCare` 和未知类别挡在 detector adapter 边界外，
   可避免其进入 Tracker 后生成没有语义的轨迹。
4. **一句话总结（面试开场用）**：我定义了独立的 Detector Adapter 边界，
   用 KITTI GT 与可控噪声 GT 向 Tracker 提供统一 3D 检测输入，同时避免将
   GT 验证误表述为 detector 性能。
