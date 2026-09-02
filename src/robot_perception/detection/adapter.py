"""Detector adapters that normalize KITTI labels into ``Detection3D``."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

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


class MMDet3DPointPillarsAdapter(BaseDetector):
    """Adapt MMDetection3D PointPillars KITTI outputs to ``Detection3D``.

    The official MMDetection3D inference API consumes a point-cloud file path,
    so the adapter writes the current frame to a temporary `.bin` file. Tests
    can inject ``model`` and ``inference_fn`` to exercise output conversion
    without installing MMDetection3D or downloading a checkpoint.
    """

    # 官方 KITTI 三分类配置顺序为 Pedestrian、Cyclist、Car。
    DEFAULT_LABEL_MAPPING = {0: 2, 1: 3, 2: 1}

    def __init__(
        self,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        *,
        device: str = "cuda:0",
        score_threshold: float = 0.0,
        class_mapping: dict[int, int] | None = None,
        model: Any | None = None,
        inference_fn: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        if not np.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be finite and within [0, 1]")
        self.score_threshold = float(score_threshold)
        self.device = device
        self.class_mapping = dict(self.DEFAULT_LABEL_MAPPING if class_mapping is None else class_mapping)
        if any(int(key) < 0 or int(value) <= 0 for key, value in self.class_mapping.items()):
            raise ValueError("class_mapping must map non-negative model labels to positive project IDs")
        self._inference_fn = inference_fn
        self.model = model
        self._uses_path_input = False
        if self.model is None:
            if config_path is None or checkpoint_path is None:
                raise ValueError("config_path and checkpoint_path are required for the real MMDetection3D adapter")
            self.model, self._inference_fn = self._load_model(
                Path(config_path), Path(checkpoint_path), device
            )
            self._uses_path_input = True
        elif self._inference_fn is None:
            raise ValueError("inference_fn is required when injecting a model for testing")

    @staticmethod
    def _load_model(
        config_path: Path,
        checkpoint_path: Path,
        device: str,
    ) -> tuple[Any, Callable[[Any, str], Any]]:
        if not config_path.is_file():
            raise FileNotFoundError(f"MMDetection3D config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PointPillars checkpoint not found: {checkpoint_path}")
        try:
            from mmdet3d.apis import inference_detector, init_model
        except ImportError as error:
            raise RuntimeError(
                "MMDetection3D stack is unavailable or incompatible "
                f"({error}); install the compatible official stack before using the real adapter"
            ) from error
        return init_model(str(config_path), str(checkpoint_path), device=device), inference_detector

    @staticmethod
    def _to_numpy(value: Any, name: str) -> np.ndarray:
        # 兼容 torch.Tensor、NumPy 数组和 MMDetection3D 包装对象。
        tensor = getattr(value, "tensor", value)
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        array = np.asarray(tensor)
        if not np.isfinite(array).all():
            raise ValueError(f"MMDetection3D output {name} contains NaN or Inf values")
        return array

    @classmethod
    def _extract_output(cls, output: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample = output
        if isinstance(sample, (list, tuple)):
            if len(sample) == 1:
                sample = sample[0]
            elif len(sample) == 2 and isinstance(sample[0], (list, tuple, dict)):
                # MMDetection3D v1.0 的 inference_detector 返回 (result, data)。
                sample = sample[0]
                if isinstance(sample, (list, tuple)):
                    if len(sample) != 1:
                        raise ValueError("MMDetection3D result list must contain exactly one frame")
                    sample = sample[0]
            else:
                raise ValueError("MMDetection3D output must contain one frame result")
        if isinstance(sample, dict):
            sample = sample.get("pred_instances_3d", sample)
            boxes = sample.get("bboxes_3d", sample.get("boxes_3d"))
            scores = sample.get("scores_3d")
            labels = sample.get("labels_3d")
        else:
            sample = getattr(sample, "pred_instances_3d", sample)
            boxes = getattr(sample, "bboxes_3d", getattr(sample, "boxes_3d", None))
            scores = getattr(sample, "scores_3d", None)
            labels = getattr(sample, "labels_3d", None)
        if boxes is None or scores is None or labels is None:
            raise ValueError("MMDetection3D output must contain boxes_3d, scores_3d, and labels_3d")
        box_array = cls._to_numpy(boxes, "boxes_3d")
        score_array = cls._to_numpy(scores, "scores_3d").reshape(-1)
        label_array = cls._to_numpy(labels, "labels_3d").reshape(-1).astype(np.int64)
        if box_array.ndim != 2 or box_array.shape[1] < 7:
            raise ValueError(f"boxes_3d must have shape (N, >=7), got {box_array.shape}")
        if len(box_array) != len(score_array) or len(box_array) != len(label_array):
            raise ValueError("boxes_3d, scores_3d, and labels_3d must have matching lengths")
        return box_array[:, :7], score_array, label_array

    def _run_inference(self, points: np.ndarray) -> Any:
        if self._inference_fn is not None and not self._uses_path_input:
            return self._inference_fn(self.model, points)
        # 官方 API 以路径为输入；临时文件生命周期覆盖整个推理调用。
        with NamedTemporaryFile(suffix=".bin") as point_file:
            np.asarray(points, dtype=np.float32).tofile(point_file)
            point_file.flush()
            return self._inference_fn(self.model, point_file.name)

    def predict(self, frame: FrameData) -> list[Detection3D]:
        """Run PointPillars and return project-level ``Detection3D`` objects."""

        points = np.asarray(frame.lidar_points)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(f"frame.lidar_points must have shape (N, 4), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("frame.lidar_points contains NaN or Inf values")
        boxes, scores, labels = self._extract_output(self._run_inference(points))
        detections: list[Detection3D] = []
        for box, score, label in zip(boxes, scores, labels):
            class_id = self.class_mapping.get(int(label))
            if class_id is None or float(score) < self.score_threshold:
                continue
            # MMDetection3D LiDAR box 默认顺序为 x,y,z,w,l,h,yaw；项目统一为 l,w,h。
            dimensions = box[[4, 3, 5]].astype(np.float64, copy=True)
            if not np.isfinite(dimensions).all() or (dimensions <= 0.0).any():
                continue
            detections.append(
                Detection3D(
                    position=box[:3].astype(np.float64, copy=True),
                    dimensions=dimensions,
                    yaw=float(box[6]),
                    score=float(score),
                    class_id=class_id,
                )
            )
        return detections
