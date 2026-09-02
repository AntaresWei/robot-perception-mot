"""Detector adapters that normalize KITTI labels into ``Detection3D``."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from robot_perception.common.types import Detection3D, FrameData, KittiTrackingAnnotation
from robot_perception.detection.base import BaseDetector, KITTI_CLASS_IDS


def _as_non_negative_std(value: float | Iterable[float], size: int, name: str) -> np.ndarray:
    # 配置允许简写为标量，也允许按 x/y/z 维度分别设置噪声标准差。
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"{name} must be a finite non-negative scalar or a ({size},) vector")
    return array


class GTDetector(BaseDetector):
    """Adapt KITTI ground-truth annotations into normalized 3D detections.

    This adapter validates tracking and association behavior only. It must not
    be used to claim detector accuracy, recall, mAP, or real detector metrics.
    """

    def __init__(
        self,
        *,
        score_threshold: float = 0.0,
        class_ids: Iterable[int] | None = None,
        default_score: float = 1.0,
    ) -> None:
        if not np.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be finite and within [0, 1]")
        if not np.isfinite(default_score) or not 0.0 <= default_score <= 1.0:
            raise ValueError("default_score must be finite and within [0, 1]")
        allowed_ids = None if class_ids is None else frozenset(int(class_id) for class_id in class_ids)
        if allowed_ids is not None and not allowed_ids.issubset(set(KITTI_CLASS_IDS.values())):
            raise ValueError("class_ids contains an unsupported KITTI class ID")
        self.score_threshold = float(score_threshold)
        self.class_ids = allowed_ids
        self.default_score = float(default_score)

    def _annotation_to_detection(
        self,
        annotation: KittiTrackingAnnotation,
        *,
        apply_score_threshold: bool = True,
    ) -> Detection3D | None:
        class_id = KITTI_CLASS_IDS.get(annotation.class_name)
        # DontCare 与未知类别不能进入 Tracker，否则会生成无意义轨迹。
        if class_id is None or (self.class_ids is not None and class_id not in self.class_ids):
            return None
        # KITTI GT 通常没有 detector score，统一赋 default_score 以满足 Detection3D 契约。
        score = self.default_score if annotation.score is None else float(annotation.score)
        if not np.isfinite(score) or (apply_score_threshold and score < self.score_threshold):
            return None
        if not np.isfinite(annotation.location).all() or not np.isfinite(annotation.dimensions).all():
            return None
        if not np.isfinite(annotation.yaw):
            return None
        return Detection3D(
            position=annotation.location.astype(np.float64, copy=True),
            dimensions=annotation.dimensions.astype(np.float64, copy=True),
            yaw=float(annotation.yaw),
            score=score,
            class_id=class_id,
        )

    def predict(self, frame: FrameData) -> list[Detection3D]:
        """Convert the frame's optional KITTI labels without mutating them."""

        if frame.annotations is None:
            return []
        detections: list[Detection3D] = []
        for annotation in frame.annotations:
            detection = self._annotation_to_detection(annotation)
            if detection is not None:
                detections.append(detection)
        return detections


class NoisyGTDetector(GTDetector):
    """Generate controlled noisy observations from GT for tracker validation."""

    def __init__(
        self,
        *,
        position_std: float | Iterable[float] = 0.0,
        dimensions_std: float | Iterable[float] = 0.0,
        yaw_std: float = 0.0,
        score_std: float = 0.0,
        random_seed: int | None = None,
        score_threshold: float = 0.0,
        class_ids: Iterable[int] | None = None,
        default_score: float = 1.0,
    ) -> None:
        super().__init__(
            score_threshold=score_threshold,
            class_ids=class_ids,
            default_score=default_score,
        )
        self.position_std = _as_non_negative_std(position_std, 3, "position_std")
        self.dimensions_std = _as_non_negative_std(dimensions_std, 3, "dimensions_std")
        if not np.isfinite(yaw_std) or yaw_std < 0.0:
            raise ValueError("yaw_std must be finite and non-negative")
        if not np.isfinite(score_std) or score_std < 0.0:
            raise ValueError("score_std must be finite and non-negative")
        self.yaw_std = float(yaw_std)
        self.score_std = float(score_std)
        self._random_generator = np.random.default_rng(random_seed)

    def predict(self, frame: FrameData) -> list[Detection3D]:
        """Perturb GT observations while keeping dimensions and scores valid."""

        noisy_detections: list[Detection3D] = []
        if frame.annotations is None:
            return []
        for annotation in frame.annotations:
            # 先保留低分 GT，才能让 threshold 对加噪后的 score 生效。
            detection = self._annotation_to_detection(annotation, apply_score_threshold=False)
            if detection is None:
                continue
            # 单个 Generator 由固定 seed 驱动，使同一实验配置可以复现实验输入。
            position = detection.position + self._random_generator.normal(0.0, self.position_std)
            # 长宽高必须为正；截断避免噪声使 3D box 退化。
            dimensions = np.maximum(
                detection.dimensions + self._random_generator.normal(0.0, self.dimensions_std),
                np.finfo(np.float64).eps,
            )
            yaw = float(detection.yaw + self._random_generator.normal(0.0, self.yaw_std))
            # 分数限制在 [0, 1]，再应用 threshold，模拟 detector 输出后的筛选顺序。
            score = float(np.clip(detection.score + self._random_generator.normal(0.0, self.score_std), 0.0, 1.0))
            if score >= self.score_threshold:
                noisy_detections.append(
                    Detection3D(
                        position=position,
                        dimensions=dimensions,
                        yaw=yaw,
                        score=score,
                        class_id=detection.class_id,
                    )
                )
        return noisy_detections
