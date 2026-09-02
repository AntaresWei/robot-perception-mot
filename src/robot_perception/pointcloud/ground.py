"""RANSAC ground-plane segmentation for LiDAR point clouds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundSegmentationResult:
    """Ground segmentation output with a normalized plane ``[a, b, c, d]``."""

    ground_points: np.ndarray
    non_ground_points: np.ndarray
    plane_model: np.ndarray | None


def _fit_plane(points_xyz: np.ndarray) -> np.ndarray | None:
    centered = points_xyz - points_xyz.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size < 3 or singular_values[-2] <= np.finfo(float).eps:
        return None
    normal = right_vectors[-1]
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= np.finfo(float).eps:
        return None
    normal = normal / normal_norm
    return np.concatenate((normal, (-normal @ points_xyz.mean(axis=0),)))


def segment_ground_ransac(
    points: np.ndarray,
    *,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    random_seed: int | None = None,
) -> GroundSegmentationResult:
    """Split a finite point cloud into plane inliers and non-ground points.

    The plane model uses the normalized equation ``ax + by + cz + d = 0``.
    Empty clouds and clouds with fewer than ``ransac_n`` points return all
    points as non-ground and use ``None`` as the plane model.
    """

    cloud = np.asarray(points)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError(f"points must have shape (N, C) with C >= 3, got {cloud.shape}")
    if not np.isfinite(cloud).all():
        raise ValueError("ground segmentation requires finite point values")
    if distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative")
    if ransac_n < 3:
        raise ValueError("ransac_n must be at least 3")
    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1")
    if len(cloud) < ransac_n:
        return GroundSegmentationResult(cloud[:0].copy(), cloud.copy(), None)

    xyz = cloud[:, :3].astype(np.float64, copy=False)
    generator = np.random.default_rng(random_seed)
    best_mask: np.ndarray | None = None
    best_count = 0
    best_error = np.inf

    for _ in range(num_iterations):
        sample = xyz[generator.choice(len(xyz), size=ransac_n, replace=False)]
        plane = _fit_plane(sample)
        if plane is None:
            continue
        distances = np.abs(xyz @ plane[:3] + plane[3])
        inlier_mask = distances <= distance_threshold
        inlier_count = int(inlier_mask.sum())
        mean_error = float(distances[inlier_mask].mean()) if inlier_count else np.inf
        if inlier_count > best_count or (inlier_count == best_count and mean_error < best_error):
            best_mask = inlier_mask
            best_count = inlier_count
            best_error = mean_error

    if best_mask is None or best_count < ransac_n:
        return GroundSegmentationResult(cloud[:0].copy(), cloud.copy(), None)

    refined_plane = _fit_plane(xyz[best_mask])
    if refined_plane is None:
        return GroundSegmentationResult(cloud[:0].copy(), cloud.copy(), None)
    refined_mask = np.abs(xyz @ refined_plane[:3] + refined_plane[3]) <= distance_threshold
    return GroundSegmentationResult(
        ground_points=cloud[refined_mask].copy(),
        non_ground_points=cloud[~refined_mask].copy(),
        plane_model=refined_plane,
    )
