"""Configurable LiDAR preprocessing stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from robot_perception.pointcloud.ground import GroundSegmentationResult, segment_ground_ransac


@dataclass(frozen=True)
class PointCloudPreprocessResult:
    """Intermediate and final point clouds retained for shape and count reporting."""

    raw_points: np.ndarray
    valid_points: np.ndarray
    roi_points: np.ndarray
    voxel_points: np.ndarray
    outlier_filtered_points: np.ndarray
    ground_points: np.ndarray
    non_ground_points: np.ndarray
    plane_model: np.ndarray | None

    @property
    def stage_shapes(self) -> dict[str, tuple[int, int]]:
        """Shapes of the raw and every processed point cloud stage."""

        return {
            "raw": self.raw_points.shape,
            "valid": self.valid_points.shape,
            "roi": self.roi_points.shape,
            "voxel": self.voxel_points.shape,
            "outlier_filtered": self.outlier_filtered_points.shape,
            "ground": self.ground_points.shape,
            "non_ground": self.non_ground_points.shape,
        }


def _validate_point_cloud(points: np.ndarray) -> np.ndarray:
    cloud = np.asarray(points)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"points must have shape (N, C) with C >= 3, got {cloud.shape}")
    if not np.issubdtype(cloud.dtype, np.number):
        raise ValueError("points must use a numeric dtype")
    return cloud


def remove_invalid_points(points: np.ndarray) -> np.ndarray:
    """Drop rows containing NaN or Inf while preserving all point attributes."""

    cloud = _validate_point_cloud(points)
    return cloud[np.isfinite(cloud).all(axis=1)].copy()


def crop_roi(
    points: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Keep finite points inside the inclusive axis-aligned ROI."""

    cloud = _validate_point_cloud(points)
    if x_min > x_max or y_min > y_max or z_min > z_max:
        raise ValueError("each ROI minimum must be less than or equal to its maximum")
    xyz = cloud[:, :3]
    mask = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= x_min)
        & (xyz[:, 0] <= x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] <= y_max)
        & (xyz[:, 2] >= z_min)
        & (xyz[:, 2] <= z_max)
    )
    return cloud[mask].copy()


def voxel_downsample(points: np.ndarray, *, voxel_size: float) -> np.ndarray:
    """Average points sharing a voxel while preserving all input columns."""

    cloud = _validate_point_cloud(points)
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if len(cloud) == 0:
        return cloud.copy()
    if not np.isfinite(cloud).all():
        raise ValueError("voxel downsampling requires finite point values")

    # floor 后的整数坐标是体素 key；同一 key 的点用 add.at 累加并取均值。
    voxel_indices = np.floor(cloud[:, :3] / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
    accumulators = np.zeros((len(counts), cloud.shape[1]), dtype=np.result_type(cloud.dtype, np.float32))
    np.add.at(accumulators, inverse, cloud)
    return accumulators / counts[:, None]


def statistical_outlier_filter(points: np.ndarray, *, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    """Remove points whose mean neighbor distance exceeds a statistical threshold."""

    cloud = _validate_point_cloud(points)
    if nb_neighbors < 1:
        raise ValueError("nb_neighbors must be at least 1")
    if std_ratio < 0.0:
        raise ValueError("std_ratio must be non-negative")
    if len(cloud) <= 2:
        return cloud.copy()
    if not np.isfinite(cloud).all():
        raise ValueError("outlier filtering requires finite point values")

    neighbor_count = min(nb_neighbors + 1, len(cloud))
    # KD-tree 将邻域查询从 O(N^2) 降为可接受的近邻查询复杂度。
    distances, _ = cKDTree(cloud[:, :3]).query(cloud[:, :3], k=neighbor_count)
    mean_distances = distances[:, 1:].mean(axis=1)
    threshold = mean_distances.mean() + std_ratio * mean_distances.std()
    return cloud[mean_distances <= threshold].copy()


def radius_outlier_filter(points: np.ndarray, *, radius: float, min_neighbors: int) -> np.ndarray:
    """Keep points having at least ``min_neighbors`` other points within ``radius``."""

    cloud = _validate_point_cloud(points)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if min_neighbors < 0:
        raise ValueError("min_neighbors must be non-negative")
    if len(cloud) == 0 or min_neighbors == 0:
        return cloud.copy()
    if not np.isfinite(cloud).all():
        raise ValueError("outlier filtering requires finite point values")

    # query_ball_point 返回包含自身的邻居数，因此要减去 1 才是“其他点”数量。
    counts = cKDTree(cloud[:, :3]).query_ball_point(cloud[:, :3], radius, return_length=True)
    return cloud[(counts - 1) >= min_neighbors].copy()


def filter_outliers(points: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """Apply the configured statistical or radius outlier filter."""

    method = config.get("method", "statistical")
    if method == "statistical":
        return statistical_outlier_filter(
            points,
            nb_neighbors=int(config["nb_neighbors"]),
            std_ratio=float(config["std_ratio"]),
        )
    if method == "radius":
        return radius_outlier_filter(
            points,
            radius=float(config["radius"]),
            min_neighbors=int(config["min_neighbors"]),
        )
    raise ValueError(f"unsupported outlier filter method: {method!r}")


def preprocess_point_cloud(
    points: np.ndarray,
    *,
    pointcloud_config: Mapping[str, Any],
    ground_config: Mapping[str, Any],
) -> PointCloudPreprocessResult:
    """Run invalid removal, ROI, voxel, outlier, and RANSAC ground stages."""

    if ground_config.get("method") != "ransac":
        raise ValueError(f"unsupported ground segmentation method: {ground_config.get('method')!r}")
    # 每一阶段保留独立数组，便于诊断某个参数导致的点数骤减。
    raw_points = _validate_point_cloud(points).copy()
    valid_points = remove_invalid_points(raw_points)
    roi_points = crop_roi(valid_points, **pointcloud_config["roi"])
    voxel_points = voxel_downsample(roi_points, voxel_size=float(pointcloud_config["voxel_size"]))
    outlier_points = filter_outliers(voxel_points, pointcloud_config["outlier_filter"])
    segmentation: GroundSegmentationResult = segment_ground_ransac(
        outlier_points,
        distance_threshold=float(ground_config["distance_threshold"]),
        ransac_n=int(ground_config["ransac_n"]),
        num_iterations=int(ground_config["num_iterations"]),
        random_seed=ground_config.get("random_seed"),
    )
    return PointCloudPreprocessResult(
        raw_points=raw_points,
        valid_points=valid_points,
        roi_points=roi_points,
        voxel_points=voxel_points,
        outlier_filtered_points=outlier_points,
        ground_points=segmentation.ground_points,
        non_ground_points=segmentation.non_ground_points,
        plane_model=segmentation.plane_model,
    )
