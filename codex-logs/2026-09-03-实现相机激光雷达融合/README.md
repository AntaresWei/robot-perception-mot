# 实现相机激光雷达融合

- **日期**：2026-09-03 10:48
- **基线 commit**：`026884fa37250462305a80094452fb2fca4e7016`
- **规模**：6 个文件，+713 / -7
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

---

## 1. 需求：要解决什么问题

**任务目标：**实现 object-level Camera-LiDAR fusion，明确拆分 3D box
projection、cross-modal association 和最终 sensor fusion。

**验收标准：**

- [x] 实现 3D box corners -> 2D bbox 和 `project_detection_to_image()`。
- [x] 支持 class、IoU、center distance 的跨模态一对一匹配。
- [x] 输出保留双侧 detection、`fusion_confidence` 和 `track_id` 的 `FusedDetection`。
- [x] 提供 synthetic Camera adapter、单元测试和真实 KITTI 可视化 demo。

---

## 2. 改了什么

1. **`src/robot_perception/fusion/camera_lidar.py`**：新增 3D box 投影、
   Camera 2D 数据类、IoU/中心距离关联、`FusedDetection` 和 mock Camera adapter。
2. **`configs/fusion.yaml`**：配置 IoU、中心距离、mock Camera score 和 bbox 噪声。
3. **`tests/test_fusion.py`**：覆盖 perfect overlap、no overlap、invalid projection、
   FOV filtering 和 mock track ID 保留。
4. **`scripts/run_fusion_demo.py`**：读取真实 KITTI `0000/100`，叠加 LiDAR 点、
   3D 投影框、2D 框和 Track ID，输出 PNG。
5. **`README.md`**：明确 projection、association、sensor fusion 的区别和
   SyntheticCameraDetector 的实验限制。

**验证方式：**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot-perception-mot \
  python -m pytest tests/
```

结果：`77 passed in 2.28s`。

```bash
conda run -n robot-perception-mot python scripts/run_fusion_demo.py \
  --sequence 0000 --frame 100
```

结果：生成 `outputs/visualizations/fusion_0000_000100.png`，图像尺寸
`1242 x 375`，投影 18,995 个 LiDAR 点，融合 3 个对象，无 unmatched LiDAR
或 Camera detection。

---

## 3. 为什么这么改

- **职责边界**：projection 只将 3D box 转成图像 bbox；association 只计算
  几何相似度并一对一匹配；只有两侧检测被匹配后才生成 FusedDetection。
- **匹配策略**：class ID 必须一致，IoU 或中心距离任一 gate 通过即可；候选按
  IoU 降序、距离升序选择，避免同一对象重复使用。
- **Camera 输入**：当前没有稳定的 Camera detector，采用 KITTI 2D 标注的
  `SyntheticCameraDetector`，明确仅用于实验和流程验证。
- **风险与后续**：当前 confidence 是几何相似度与两侧 score 的组合，不是经过
  校准的概率；真实 Camera detector 接入后需重新评估阈值和置信度定义。

---

## 4. 我学到了什么（面试可复述）

1. **技术点**：KITTI/PointPillars 的 3D box 需要先按坐标系绕 z 或 y 轴
   构造 8 个 corners，再通过 calibration 和 P2 得到 2D bbox。
2. **工程判断**：完全在图像 FOV 外的 box 必须返回 None，不能简单 clip 到
   图像边界，否则会制造虚假的跨模态匹配。
3. **工程判断**：将 mock Camera detector 放在独立接口后，可以验证融合管线，
   同时避免把 GT 结果误报为 Camera detector 性能。
4. **一句话总结（面试开场用）**：我实现了 projection、cross-modal association
   和 sensor fusion 分层的 object-level 融合，并在真实 KITTI 帧上完成可视化验证。
