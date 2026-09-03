"""Object-level Camera-LiDAR projection, association, and fusion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_perception.common.geometry import lidar_to_camera
from robot_perception.common.types import Detection3D, FrameData
from robot_perception.detection.base import KITTI_CLASS_IDS


@dataclass(frozen=True)
class Camera2DDetection:
    """Camera detection represented by pixel ``[x1, y1, x2, y2]``."""

    bbox: np.ndarray
    score: float
    class_id: int
    track_id: int | None = None

    def __post_init__(self) -> None:
        bbox = np.asarray(self.bbox, dtype=np.float64)
        if bbox.shape != (4,):
            raise ValueError(f"bbox must have shape (4,), got {bbox.shape}")
        if not np.isfinite(bbox).all() or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError("bbox must be finite and ordered as x1 <= x2, y1 <= y2")
        if not np.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and within [0, 1]")
        if int(self.class_id) <= 0:
            raise ValueError("class_id must be positive")
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "class_id", int(self.class_id))


@dataclass(frozen=True)
class FusedDetection:
    """A matched object retaining both sensor observations and confidence."""

    lidar_detection: Detection3D
    camera_detection: Camera2DDetection
    projected_bbox: np.ndarray
    fusion_confidence: float
    track_id: int | None = None

    def __post_init__(self) -> None:
        bbox = np.asarray(self.projected_bbox, dtype=np.float64)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError("projected_bbox must be a finite vector with shape (4,)")
        if not np.isfinite(self.fusion_confidence) or not 0.0 <= self.fusion_confidence <= 1.0:
            raise ValueError("fusion_confidence must be finite and within [0, 1]")
        object.__setattr__(self, "projected_bbox", bbox)


@dataclass(frozen=True)
class FusionResult:
    """Cross-modal association output using original sensor indices."""

    fused_detections: tuple[FusedDetection, ...]
    unmatched_lidar: np.ndarray
    unmatched_camera: np.ndarray
    iou_matrix: np.ndarray
    center_distance_matrix: np.ndarray


class BaseCameraDetector(ABC):
    """Interface for camera 2D detectors used by object-level fusion."""

    @abstractmethod
    def predict(self, frame: FrameData) -> list[Camera2DDetection]:
        """Return camera detections in image pixel coordinates."""


def box_corners_3d(detection: Detection3D, *, coordinate_frame: str = "lidar") -> np.ndarray:
    """Construct eight corners for a center-based ``[l, w, h]`` 3D box.

    LiDAR boxes rotate around z. Camera boxes rotate around y. This function
    only constructs corners; projection is performed separately below.
    """

    if coordinate_frame not in {"lidar", "camera"}:
        raise ValueError("coordinate_frame must be 'lidar' or 'camera'")
    center = np.asarray(detection.position, dtype=np.float64)
    dimensions = np.asarray(detection.dimensions, dtype=np.float64)
    if center.shape != (3,) or dimensions.shape != (3,):
        raise ValueError("Detection3D position and dimensions must have shape (3,)")
    if not np.isfinite(center).all() or not np.isfinite(dimensions).all() or (dimensions <= 0.0).any():
        raise ValueError("Detection3D position/dimensions must be finite and dimensions positive")

    length, width, height = dimensions
    offsets = np.asarray(
        (
            (-length / 2, -width / 2, -height / 2),
            (-length / 2, -width / 2, height / 2),
            (-length / 2, width / 2, -height / 2),
            (-length / 2, width / 2, height / 2),
            (length / 2, -width / 2, -height / 2),
            (length / 2, -width / 2, height / 2),
            (length / 2, width / 2, -height / 2),
            (length / 2, width / 2, height / 2),
        ),
        dtype=np.float64,
    )
    cosine, sine = np.cos(detection.yaw), np.sin(detection.yaw)
    if coordinate_frame == "lidar":
        rotation = np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    else:
        rotation = np.asarray(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))
    return offsets @ rotation.T + center


def _project_corners(corners_camera: np.ndarray, projection: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (corners_camera, np.ones((len(corners_camera), 1), dtype=np.float64)), axis=1
    )
    projected = homogeneous @ projection.T
    denominator = projected[:, 2]
    valid = np.isfinite(denominator) & (denominator > 0.0)
    if not valid.any():
        return np.empty((0, 2), dtype=np.float64)
    pixels = projected[valid, :2] / denominator[valid, None]
    return pixels[np.isfinite(pixels).all(axis=1)]


def project_detection_corners_to_image(
    detection: Detection3D,
    calibration: Any,
    *,
    coordinate_frame: str = "lidar",
) -> np.ndarray:
    """Return valid projected corner pixels without applying image FOV clipping."""

    corners = box_corners_3d(detection, coordinate_frame=coordinate_frame)
    if coordinate_frame == "lidar":
        corners = lidar_to_camera(corners, calibration)
    projection = getattr(calibration, "P2", None)
    if projection is None and isinstance(calibration, dict):
        projection = calibration.get("P2")
    if projection is None:
        raise ValueError("calibration is missing P2")
    projection = np.asarray(projection, dtype=np.float64)
    if projection.shape != (3, 4) or not np.isfinite(projection).all():
        raise ValueError("P2 must be a finite matrix with shape (3, 4)")
    finite_front = np.isfinite(corners).all(axis=1) & (corners[:, 2] > 0.0)
    if not finite_front.any():
        return np.empty((0, 2), dtype=np.float64)
    return _project_corners(corners[finite_front], projection)


def project_detection_to_image(
    detection: Detection3D,
    calibration: Any,
    *,
    image_shape: tuple[int, ...] | None = None,
    coordinate_frame: str = "lidar",
) -> np.ndarray | None:
    """Project a 3D box to a clipped 2D bbox, or return ``None`` if invisible.

    This is only the geometric projection stage. It does not compare against a
    camera detection and therefore is not, by itself, sensor fusion.
    """

    pixels = project_detection_corners_to_image(
        detection,
        calibration,
        coordinate_frame=coordinate_frame,
    )
    if pixels.size == 0:
        return None
    bbox = np.asarray((pixels[:, 0].min(), pixels[:, 1].min(), pixels[:, 0].max(), pixels[:, 1].max()))
    if image_shape is None:
        return bbox
    if len(image_shape) < 2:
        raise ValueError(f"image_shape must contain height and width, got {image_shape}")
    height, width = image_shape[:2]
    # 先判断 bbox 是否与图像有交集，避免完全在 FOV 外的 box 被压到边界。
    if bbox[2] < 0.0 or bbox[0] >= width or bbox[3] < 0.0 or bbox[1] >= height:
        return None
    return np.asarray(
        (
            np.clip(bbox[0], 0.0, width - 1.0),
            np.clip(bbox[1], 0.0, height - 1.0),
            np.clip(bbox[2], 0.0, width - 1.0),
            np.clip(bbox[3], 0.0, height - 1.0),
        ),
        dtype=np.float64,
    )


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def associate_camera_lidar_detections(
    lidar_detections: list[Detection3D],
    camera_detections: list[Camera2DDetection],
    calibration: Any,
    *,
    image_shape: tuple[int, ...] | None = None,
    iou_threshold: float | None = 0.1,
    center_distance_threshold: float | None = 80.0,
    coordinate_frame: str = "lidar",
) -> FusionResult:
    """Associate projected LiDAR boxes with camera boxes one-to-one.

    A candidate passes when class IDs agree and either enabled geometric gate
    passes. Highest IoU is preferred, then smallest center distance. The final
    ``FusedDetection`` is the sensor-fusion result; projection and association
    matrices remain available separately for inspection.
    """

    if iou_threshold is None and center_distance_threshold is None:
        raise ValueError("at least one association threshold must be enabled")
    if iou_threshold is not None and not np.isfinite(iou_threshold):
        raise ValueError("iou_threshold must be finite")
    if center_distance_threshold is not None and (
        not np.isfinite(center_distance_threshold) or center_distance_threshold < 0.0
    ):
        raise ValueError("center_distance_threshold must be finite and non-negative")

    projected_boxes = [
        project_detection_to_image(
            detection,
            calibration,
            image_shape=image_shape,
            coordinate_frame=coordinate_frame,
        )
        for detection in lidar_detections
    ]
    iou_matrix = np.zeros((len(lidar_detections), len(camera_detections)), dtype=np.float64)
    # 先计算几何关系，再按 IoU 优先级做一对一贪心匹配，避免同一目标被重复融合。
    center_distances = np.full_like(iou_matrix, np.inf)
    candidates: list[tuple[float, float, int, int]] = []
    for lidar_index, projected_bbox in enumerate(projected_boxes):
        if projected_bbox is None:
            continue
        projected_center = (projected_bbox[:2] + projected_bbox[2:]) / 2.0
        for camera_index, camera_detection in enumerate(camera_detections):
            if lidar_detections[lidar_index].class_id != camera_detection.class_id:
                continue
            iou = _bbox_iou(projected_bbox, camera_detection.bbox)
            camera_center = (camera_detection.bbox[:2] + camera_detection.bbox[2:]) / 2.0
            center_distance = float(np.linalg.norm(projected_center - camera_center))
            iou_matrix[lidar_index, camera_index] = iou
            center_distances[lidar_index, camera_index] = center_distance
            passes_iou = iou_threshold is not None and iou >= iou_threshold
            passes_distance = center_distance_threshold is not None and center_distance <= center_distance_threshold
            if passes_iou or passes_distance:
                candidates.append((-iou, center_distance, lidar_index, camera_index))

    candidates.sort()
    used_lidar: set[int] = set()
    used_camera: set[int] = set()
    fused: list[FusedDetection] = []
    for negative_iou, center_distance, lidar_index, camera_index in candidates:
        if lidar_index in used_lidar or camera_index in used_camera:
            continue
        used_lidar.add(lidar_index)
        used_camera.add(camera_index)
        lidar_detection = lidar_detections[lidar_index]
        camera_detection = camera_detections[camera_index]
        iou = -negative_iou
        distance_scale = max(center_distance_threshold or 1.0, 1e-12)
        geometry_confidence = max(iou, float(np.exp(-center_distance / distance_scale)))
        confidence = float(
            np.clip(
                geometry_confidence * 0.5 * (lidar_detection.score + camera_detection.score),
                0.0,
                1.0,
            )
        )
        fused.append(
            FusedDetection(
                lidar_detection=lidar_detection,
                camera_detection=camera_detection,
                projected_bbox=projected_boxes[lidar_index],
                fusion_confidence=confidence,
                track_id=camera_detection.track_id,
            )
        )

    return FusionResult(
        fused_detections=tuple(fused),
        unmatched_lidar=np.asarray(sorted(set(range(len(lidar_detections))) - used_lidar), dtype=np.int64),
        unmatched_camera=np.asarray(sorted(set(range(len(camera_detections))) - used_camera), dtype=np.int64),
        iou_matrix=iou_matrix,
        center_distance_matrix=center_distances,
    )


class SyntheticCameraDetector(BaseCameraDetector):
    """Create experimental camera detections from KITTI 2D annotations.

    This mock adapter is for fusion plumbing validation only, not camera-detector
    metrics or a claim of real image detection capability.
    """

    def __init__(
        self,
        *,
        score_threshold: float = 0.0,
        class_ids: set[int] | None = None,
        bbox_noise_std: float = 0.0,
        random_seed: int | None = None,
    ) -> None:
        if not np.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be finite and within [0, 1]")
        if not np.isfinite(bbox_noise_std) or bbox_noise_std < 0.0:
            raise ValueError("bbox_noise_std must be finite and non-negative")
        self.score_threshold = float(score_threshold)
        self.class_ids = None if class_ids is None else set(class_ids)
        self.bbox_noise_std = float(bbox_noise_std)
        self._random_generator = np.random.default_rng(random_seed)

    def predict(self, frame: FrameData) -> list[Camera2DDetection]:
        if frame.annotations is None:
            return []
        detections: list[Camera2DDetection] = []
        for annotation in frame.annotations:
            # mock detector 只复用标注框的像素位置，不执行任何图像模型推理。
            class_id = KITTI_CLASS_IDS.get(annotation.class_name)
            if class_id is None or (self.class_ids is not None and class_id not in self.class_ids):
                continue
            score = 1.0 if annotation.score is None else float(annotation.score)
            if score < self.score_threshold:
                continue
            bbox = annotation.bbox_2d.astype(np.float64, copy=True)
            if self.bbox_noise_std:
                bbox += self._random_generator.normal(0.0, self.bbox_noise_std, size=4)
                bbox[2:] = np.maximum(bbox[2:], bbox[:2])
            detections.append(Camera2DDetection(bbox, score, class_id, annotation.track_id))
        return detections


project_detection = project_detection_to_image
associate_detections = associate_camera_lidar_detections
