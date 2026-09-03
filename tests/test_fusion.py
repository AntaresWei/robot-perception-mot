"""Tests for object-level Camera-LiDAR projection and fusion."""

from __future__ import annotations

import numpy as np
import pytest
import importlib.util
import sys
from pathlib import Path

from robot_perception.common.types import Detection3D, FrameData, KittiTrackingAnnotation
from robot_perception.dataset.calibration import KittiCalibration
from robot_perception.fusion import (
    Camera2DDetection,
    SyntheticCameraDetector,
    associate_camera_lidar_detections,
    box_corners_3d,
    project_detection_to_image,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location("fusion_visualization", ROOT / "scripts" / "run_fusion_demo.py")
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
fusion_visualization = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = fusion_visualization
SCRIPT_SPEC.loader.exec_module(fusion_visualization)


def _calibration() -> KittiCalibration:
    return KittiCalibration(
        P2=np.asarray(((10.0, 0.0, 50.0, 0.0), (0.0, 10.0, 50.0, 0.0), (0.0, 0.0, 1.0, 0.0))),
        R0_rect=np.eye(3),
        Tr_velo_to_cam=np.hstack((np.eye(3), np.zeros((3, 1)))),
    )


def _detection(position=(0.0, 0.0, 10.0), class_id=1) -> Detection3D:
    return Detection3D(
        position=np.asarray(position, dtype=np.float64),
        dimensions=np.asarray((2.0, 2.0, 2.0), dtype=np.float64),
        yaw=0.0,
        score=0.9,
        class_id=class_id,
    )


def test_box_corners_have_expected_shape() -> None:
    corners = box_corners_3d(_detection(), coordinate_frame="camera")

    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.mean(axis=0), (0.0, 0.0, 10.0))


def test_project_detection_to_image_returns_3d_box_bbox() -> None:
    bbox = project_detection_to_image(_detection(), _calibration(), image_shape=(100, 100), coordinate_frame="camera")

    assert bbox is not None
    assert bbox[0] < bbox[2]
    assert bbox[1] < bbox[3]
    assert (bbox >= 0.0).all() and (bbox < 100.0).all()


def test_perfect_overlap_fuses_two_modalities() -> None:
    lidar_detection = _detection()
    projected_bbox = project_detection_to_image(
        lidar_detection, _calibration(), image_shape=(100, 100), coordinate_frame="camera"
    )
    assert projected_bbox is not None
    camera_detection = Camera2DDetection(projected_bbox, 0.8, 1, track_id=12)

    result = associate_camera_lidar_detections(
        [lidar_detection],
        [camera_detection],
        _calibration(),
        image_shape=(100, 100),
        iou_threshold=0.99,
        center_distance_threshold=None,
        coordinate_frame="camera",
    )

    assert len(result.fused_detections) == 1
    assert result.fused_detections[0].track_id == 12
    assert result.fused_detections[0].fusion_confidence > 0.7
    np.testing.assert_allclose(result.iou_matrix, ((1.0,),))


def test_no_overlap_leaves_both_detections_unmatched() -> None:
    result = associate_camera_lidar_detections(
        [_detection()],
        [Camera2DDetection(np.asarray((80.0, 80.0, 90.0, 90.0)), 0.8, 1)],
        _calibration(),
        image_shape=(100, 100),
        iou_threshold=0.5,
        center_distance_threshold=1.0,
        coordinate_frame="camera",
    )

    assert result.fused_detections == ()
    np.testing.assert_array_equal(result.unmatched_lidar, (0,))
    np.testing.assert_array_equal(result.unmatched_camera, (0,))


def test_invalid_projection_returns_none() -> None:
    assert project_detection_to_image(
        _detection(position=(0.0, 0.0, -5.0)),
        _calibration(),
        image_shape=(100, 100),
        coordinate_frame="camera",
    ) is None


def test_camera_fov_filtering_returns_none_for_box_outside_image() -> None:
    assert project_detection_to_image(
        _detection(position=(100.0, 0.0, 10.0)),
        _calibration(),
        image_shape=(100, 100),
        coordinate_frame="camera",
    ) is None


def test_synthetic_camera_detector_is_experimental_and_preserves_track_id() -> None:
    annotation = KittiTrackingAnnotation(
        frame_id=0,
        track_id=4,
        class_name="Car",
        truncation=0.0,
        occlusion=0,
        alpha=0.0,
        bbox_2d=np.asarray((10.0, 20.0, 30.0, 40.0)),
        dimensions=np.asarray((4.0, 2.0, 1.5)),
        location=np.asarray((0.0, 0.0, 10.0)),
        yaw=0.0,
    )
    frame = FrameData("0000", 0, None, np.empty((0, 4)), None, annotations=[annotation])

    detections = SyntheticCameraDetector(class_ids={1}).predict(frame)

    assert len(detections) == 1
    assert detections[0].track_id == 4
    np.testing.assert_allclose(detections[0].bbox, (10.0, 20.0, 30.0, 40.0))


def test_camera_detection_rejects_invalid_bbox() -> None:
    with pytest.raises(ValueError, match="bbox"):
        Camera2DDetection(np.asarray((10.0, 20.0, 5.0, 40.0)), 0.8, 1)


def test_visualization_selects_raw_non_ground_or_none() -> None:
    raw = np.asarray(((1.0, 0.0, 0.0, 0.1), (2.0, 0.0, 0.0, 0.2)))
    non_ground = raw[1:]

    np.testing.assert_array_equal(fusion_visualization._select_lidar_points("raw", raw, non_ground), raw)
    np.testing.assert_array_equal(
        fusion_visualization._select_lidar_points("non_ground", raw, non_ground), non_ground
    )
    assert fusion_visualization._select_lidar_points("none", raw, non_ground).shape == (0, 4)
    with pytest.raises(ValueError, match="lidar_points_mode"):
        fusion_visualization._select_lidar_points("ground", raw, non_ground)


def test_visualization_depth_colours_vary_with_depth() -> None:
    colours = fusion_visualization._depth_colours(np.asarray((2.0, 10.0, 20.0)), 0.0, 100.0)

    assert colours.shape == (3, 3)
    assert colours.dtype == np.uint8
    assert not np.array_equal(colours[0], colours[-1])


def test_visualization_projection_depths_align_with_pixels() -> None:
    points = np.asarray(((0.0, 0.0, 10.0, 0.1), (0.0, 0.0, -1.0, 0.2)))

    pixels, depths = fusion_visualization._project_points_with_depth(points, _calibration(), (100, 100, 3))

    np.testing.assert_allclose(pixels, ((50.0, 50.0),))
    np.testing.assert_allclose(depths, (10.0,))
