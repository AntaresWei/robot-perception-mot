# 补充跟踪融合问答

- **日期**：2026-09-07
- **基线 commit**：`f6791cf`
- **规模**：1 个文件，+632 / -0
- **完整 diff**：[changes.patch](./changes.patch) ｜ [变更清单](./files.txt)

## 1. 需求

把跟踪和 Camera-LiDAR Fusion 中容易被面试官追问的机制补进中文教程：Lost Track
恢复、Mahalanobis gating、ID 管理、2D 是否做 Kalman、2D/3D 跨模态关联和中心距离
作用；同时增加更多阈值、ReID、数据泄漏、坐标、实时性和故障定位问答。

## 2. 改了什么

更新 `docs/project_tutorial_zh.md`：

1. 增加 Track 的 Lost -> Confirmed 恢复条件、`max_age`、`time_since_update` 和
   原 ID 保留规则。
2. 解释马氏距离中的 innovation/covariance、gating 与 Hungarian 的职责边界。
3. 解释 Tracker ID、KITTI GT track_id、检测数组下标的区别和不可复用规则。
4. 明确当前 2D bbox 不进入 3D Kalman；补充 2D/3D projection、IoU、center distance、
   一对一 Fusion 关联的具体实现。
5. 增加未匹配目标处理、低置信度策略、ReID gated fallback 和十余个面试问答。

## 3. 为什么这么改

- 教程不只讲公式，还回答“为什么第一帧没有 confirmed track”“为什么两个框不能平均”
  等实际运行中最容易产生误解的问题。
- 明确 MOT 内部使用 Mahalanobis + Hungarian，而当前 Fusion 使用 IoU 优先贪心匹配，
  防止读者把两个关联阶段混为一谈。
- 对 ReID、2D tracker、EKF/UKF 状态级融合明确标注为未实现，保持项目陈述准确。

## 4. 验证

```bash
/data/weijiaman/miniconda3/envs/robot-perception-mot/bin/python -m pytest tests/
# 106 passed
git diff --check
```

教程约 632 行，内容引用当前 Tracker/Fusion 行为、阈值和真实实验边界。没有修改核心
算法；归档 patch 已通过反向检查。
