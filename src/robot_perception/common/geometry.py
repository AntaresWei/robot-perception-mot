"""Vectorized coordinate transforms used by the calibration task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _matrix_from_calibration(calibration: Any, name: str) -> np.ndarray:
    if isinstance(calibration, Mapping):
        value = calibration.get(name)
    else:
        value = getattr(calibration, name, None)
    if value is None:
        raise ValueError(f"calibration is missing {name}")
    matrix = np.asarray(value, dtype=np.float64)
    expected_shape = {"Tr_velo_to_cam": (3, 4), "R0_rect": (3, 3), "P2": (3, 4)}[name]
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    return matrix


def _validate_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] not in (3, 4):
        raise ValueError(f"points must have shape (N, 3) or (N, 4), got {array.shape}")
    return array.astype(np.float64, copy=False)


def _homogeneous(points: np.ndarray) -> np.ndarray:
    # 加上 w=1 后，3x4 外参矩阵可以同时完成旋转和位移。
    return np.concatenate((points, np.ones((points.shape[0], 1), dtype=points.dtype)), axis=1)


def lidar_to_camera(
    points: np.ndarray,
    calibration: Any,
    *,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Transform LiDAR points to the unrectified camera frame.

    Non-finite input points are discarded. The optional mask maps returned rows
    back to the original input rows.
    """

    array = _validate_points(points)
    # 只保留有限行，并返回原始索引 mask，方便上游同步过滤强度等附加字段。
    finite_mask = np.isfinite(array).all(axis=1)
    xyz = array[finite_mask, :3]
    transform = _matrix_from_calibration(calibration, "Tr_velo_to_cam")
    camera_points = _homogeneous(xyz) @ transform.T
    if return_mask:
        return camera_points, finite_mask
    return camera_points


def camera_to_image(
    points: np.ndarray,
    calibration: Any,
    *,
    image_shape: tuple[int, ...] | None = None,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points into image pixels.

    Rectification is applied before projection. Points with non-finite values,
    non-positive rectified depth, or pixels outside ``image_shape`` are removed.
    ``image_shape`` may be ``(height, width)`` or an image array shape.
    """

    array = _validate_points(points)
    finite_mask = np.isfinite(array).all(axis=1)
    camera_points = array[finite_mask, :3]
    rectification = _matrix_from_calibration(calibration, "R0_rect")
    projection = _matrix_from_calibration(calibration, "P2")

    rectified_points = camera_points @ rectification.T
    depths = rectified_points[:, 2]
    # z<=0 的点位于相机后方或投影无定义，必须在除法前排除。
    positive_depth_mask = depths > 0.0
    valid_points = rectified_points[positive_depth_mask]
    valid_depths = depths[positive_depth_mask]
    projected_homogeneous = _homogeneous(valid_points) @ projection.T
    denominator = projected_homogeneous[:, 2]
    valid_projection_mask = np.isfinite(denominator) & (denominator > 0.0)
    projected_homogeneous = projected_homogeneous[valid_projection_mask]
    valid_depths = valid_depths[valid_projection_mask]
    pixels = projected_homogeneous[:, :2] / projected_homogeneous[:, 2:3]

    if image_shape is not None:
        if len(image_shape) < 2:
            raise ValueError(f"image_shape must contain height and width, got {image_shape}")
        height, width = image_shape[:2]
        in_image = (
            (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < height)
        )
    else:
        in_image = np.ones(pixels.shape[0], dtype=bool)

    pixels = pixels[in_image]
    # 将连续过滤后的索引映射回调用者传入的原始点云坐标空间。
    valid_indices = np.flatnonzero(finite_mask)[positive_depth_mask][valid_projection_mask][in_image]
    output_mask = np.zeros(array.shape[0], dtype=bool)
    output_mask[valid_indices] = True
    if return_mask:
        return pixels, output_mask
    return pixels


def lidar_to_image(
    points: np.ndarray,
    calibration: Any,
    *,
    image_shape: tuple[int, ...] | None = None,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Transform LiDAR points through camera and rectified image coordinates."""

    # camera_to_image 返回的是“有限 LiDAR 点”空间中的 mask，这里再映射回原始输入。
    camera_points, finite_mask = lidar_to_camera(points, calibration, return_mask=True)
    pixels, camera_mask = camera_to_image(
        camera_points,
        calibration,
        image_shape=image_shape,
        return_mask=True,
    )
    output_mask = np.zeros(len(np.asarray(points)), dtype=bool)
    finite_indices = np.flatnonzero(finite_mask)
    output_mask[finite_indices[camera_mask]] = True
    if return_mask:
        return pixels, output_mask
    return pixels
