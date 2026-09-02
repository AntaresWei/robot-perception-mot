"""Tests for the LiDAR preprocessing pipeline."""

from __future__ import annotations

import numpy as np

from robot_perception.pointcloud import (
    crop_roi,
    preprocess_point_cloud,
    radius_outlier_filter,
    remove_invalid_points,
    segment_ground_ransac,
    statistical_outlier_filter,
    voxel_downsample,
)


def test_remove_invalid_points() -> None:
    points = np.asarray(((1.0, 2.0, 3.0, 0.1), (np.nan, 0.0, 0.0, 0.2), (0.0, 0.0, np.inf, 0.3)))

    result = remove_invalid_points(points)

    np.testing.assert_allclose(result, ((1.0, 2.0, 3.0, 0.1),))


def test_crop_roi_keeps_inclusive_boundaries() -> None:
    points = np.asarray(((-1.0, 0.0, 0.0, 0.0), (0.0, -2.0, -3.0, 0.1), (1.0, 2.0, 3.0, 0.2), (1.1, 0.0, 0.0, 0.3)))

    result = crop_roi(points, x_min=0.0, x_max=1.0, y_min=-2.0, y_max=2.0, z_min=-3.0, z_max=3.0)

    np.testing.assert_allclose(result, ((0.0, -2.0, -3.0, 0.1), (1.0, 2.0, 3.0, 0.2)))


def test_voxel_downsample_averages_points_and_intensity() -> None:
    points = np.asarray(((0.01, 0.01, 0.01, 0.2), (0.09, 0.09, 0.09, 0.6), (0.20, 0.0, 0.0, 1.0)))

    result = voxel_downsample(points, voxel_size=0.1)

    np.testing.assert_allclose(result, ((0.05, 0.05, 0.05, 0.4), (0.2, 0.0, 0.0, 1.0)))


def test_statistical_and_radius_outlier_filters_remove_isolated_point() -> None:
    cluster = np.asarray(((0.0, 0.0, 0.0, 0.1), (0.1, 0.0, 0.0, 0.1), (0.0, 0.1, 0.0, 0.1), (0.1, 0.1, 0.0, 0.1)))
    points = np.vstack((cluster, ((10.0, 10.0, 10.0, 0.1),)))

    statistical = statistical_outlier_filter(points, nb_neighbors=2, std_ratio=1.0)
    radius = radius_outlier_filter(points, radius=0.2, min_neighbors=2)

    assert len(statistical) == len(cluster)
    assert len(radius) == len(cluster)


def test_ransac_ground_segmentation_on_synthetic_plane() -> None:
    grid_x, grid_y = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    ground = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size), np.ones(grid_x.size)))
    obstacles = np.asarray(((0.0, 0.0, 1.0, 0.5), (1.0, 1.0, 1.5, 0.5), (-1.0, 1.0, 2.0, 0.5)))
    points = np.vstack((ground, obstacles))

    result = segment_ground_ransac(
        points,
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=100,
        random_seed=0,
    )

    assert result.plane_model is not None
    assert len(result.ground_points) == len(ground)
    assert len(result.non_ground_points) == len(obstacles)
    assert abs(result.plane_model[2]) == 1.0
    assert abs(result.plane_model[3]) < 1e-12


def test_empty_input_is_safe_for_the_full_pipeline() -> None:
    pointcloud_config = {
        "voxel_size": 0.1,
        "roi": {"x_min": 0.0, "x_max": 70.0, "y_min": -40.0, "y_max": 40.0, "z_min": -3.0, "z_max": 2.0},
        "outlier_filter": {"method": "statistical", "nb_neighbors": 20, "std_ratio": 2.0, "radius": 0.8, "min_neighbors": 5},
    }
    ground_config = {"method": "ransac", "distance_threshold": 0.15, "ransac_n": 3, "num_iterations": 100, "random_seed": 0}

    result = preprocess_point_cloud(np.empty((0, 4), dtype=np.float32), pointcloud_config=pointcloud_config, ground_config=ground_config)

    assert result.plane_model is None
    assert all(shape == (0, 4) for shape in result.stage_shapes.values())
