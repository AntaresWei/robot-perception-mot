"""Unit tests for real-evaluation geometry helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from robot_perception.common.types import KittiTrackingAnnotation
from robot_perception.dataset.calibration import KittiCalibration

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("real_eval", ROOT / "scripts" / "evaluate_real_mot.py")
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_camera_annotation_is_converted_to_lidar_center() -> None:
    annotation = KittiTrackingAnnotation(
        frame_id=0, track_id=3, class_name="Car", truncation=0.0, occlusion=0,
        alpha=0.0, bbox_2d=np.ones(4), dimensions=np.asarray((4.0, 2.0, 2.0)),
        location=np.asarray((1.0, 2.0, 10.0)), yaw=0.0,
    )
    calibration = KittiCalibration(P2=np.hstack((np.eye(3), np.zeros((3, 1)))), R0_rect=np.eye(3), Tr_velo_to_cam=np.hstack((np.eye(3), np.asarray([[1.0], [0.0], [0.0]]))))

    result = module._camera_annotation_to_lidar(annotation, calibration)

    np.testing.assert_allclose(result, (0.0, 1.0, 10.0))


def test_standard_mot_metrics_use_global_identity_assignment() -> None:
    truths = [
        [(7, 1, np.asarray((0.0, 0.0, 0.0)))],
        [(7, 1, np.asarray((1.0, 0.0, 0.0)))],
    ]
    predictions = [
        [(3, 1, np.asarray((0.0, 0.0, 0.0)))],
        [(3, 1, np.asarray((1.0, 0.0, 0.0)))],
    ]

    metrics = module._standard_mot_metrics(truths, predictions, distance_threshold=0.5)

    assert metrics["MOTA"] == 1.0
    assert metrics["IDF1"] == 1.0
    assert metrics["IDP"] == 1.0
    assert metrics["IDR"] == 1.0
    assert metrics["IDS"] == 0
    assert metrics["GT"] == 2
    assert metrics["matches"] == 2


def test_standard_mot_metrics_count_identity_switch_and_class_gate() -> None:
    truths = [
        [(7, 1, np.asarray((0.0, 0.0, 0.0))), (8, 2, np.asarray((5.0, 0.0, 0.0)))],
        [(7, 1, np.asarray((1.0, 0.0, 0.0))), (8, 2, np.asarray((6.0, 0.0, 0.0)))],
    ]
    predictions = [
        [(3, 1, np.asarray((0.0, 0.0, 0.0))), (4, 2, np.asarray((5.0, 0.0, 0.0)))],
        [(5, 1, np.asarray((1.0, 0.0, 0.0))), (4, 1, np.asarray((6.0, 0.0, 0.0)))],
    ]

    metrics = module._standard_mot_metrics(truths, predictions, distance_threshold=0.5)

    assert metrics["IDS"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["GT"] == 4
