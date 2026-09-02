"""Tests for GT and noisy-GT detector adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_perception.common.types import FrameData, KittiTrackingAnnotation
from robot_perception.detection import (
    BaseDetector,
    GTDetector,
    MMDet3DPointPillarsAdapter,
    NoisyGTDetector,
)


def _annotation(
    *,
    class_name: str = "Car",
    score: float | None = None,
    location: tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> KittiTrackingAnnotation:
    return KittiTrackingAnnotation(
        frame_id=0,
        track_id=4,
        class_name=class_name,
        truncation=0.0,
        occlusion=0,
        alpha=0.0,
        bbox_2d=np.asarray((1.0, 2.0, 3.0, 4.0)),
        dimensions=np.asarray((4.0, 2.0, 1.5)),
        location=np.asarray(location),
        yaw=0.25,
        score=score,
    )


def _frame(annotations: list[KittiTrackingAnnotation] | None) -> FrameData:
    return FrameData(
        sequence_id="0000",
        frame_id=0,
        image=None,
        lidar_points=np.empty((0, 4)),
        calibration=None,
        annotations=annotations,
    )


def test_gt_detector_implements_base_interface_and_normalizes_detection() -> None:
    detector = GTDetector()

    assert isinstance(detector, BaseDetector)
    detection = detector.predict(_frame([_annotation()]))[0]

    np.testing.assert_allclose(detection.position, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(detection.dimensions, (4.0, 2.0, 1.5))
    assert detection.yaw == pytest.approx(0.25)
    assert detection.score == pytest.approx(1.0)
    assert detection.class_id == 1


def test_gt_detector_filters_unknown_classes_and_selected_classes() -> None:
    frame = _frame(
        [
            _annotation(class_name="DontCare"),
            _annotation(class_name="Pedestrian"),
            _annotation(class_name="Car"),
        ]
    )

    detections = GTDetector(class_ids=[1]).predict(frame)

    assert [detection.class_id for detection in detections] == [1]


def test_gt_detector_applies_score_threshold() -> None:
    frame = _frame([_annotation(score=0.4), _annotation(score=0.8)])

    detections = GTDetector(score_threshold=0.5).predict(frame)

    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.8)


def test_gt_detector_returns_empty_when_annotations_are_unavailable() -> None:
    assert GTDetector().predict(_frame(None)) == []


def test_noisy_gt_detector_is_reproducible_and_changes_detection() -> None:
    frame = _frame([_annotation()])
    first = NoisyGTDetector(position_std=0.5, dimensions_std=0.2, yaw_std=0.1, score_std=0.1, random_seed=3)
    second = NoisyGTDetector(position_std=0.5, dimensions_std=0.2, yaw_std=0.1, score_std=0.1, random_seed=3)

    first_detection = first.predict(frame)[0]
    second_detection = second.predict(frame)[0]

    np.testing.assert_allclose(first_detection.position, second_detection.position)
    np.testing.assert_allclose(first_detection.dimensions, second_detection.dimensions)
    assert first_detection.yaw == pytest.approx(second_detection.yaw)
    assert first_detection.score == pytest.approx(second_detection.score)
    assert not np.allclose(first_detection.position, (1.0, 2.0, 3.0))
    assert (first_detection.dimensions > 0.0).all()


def test_noisy_gt_detector_applies_threshold_after_score_noise() -> None:
    detector = NoisyGTDetector(score_std=0.0, default_score=0.4, score_threshold=0.5, random_seed=0)

    assert detector.predict(_frame([_annotation()])) == []


def test_mmdet3d_pointpillars_adapter_converts_model_output() -> None:
    output = {
        "pred_instances_3d": {
            "bboxes_3d": np.asarray(
                (
                    (1.0, 2.0, 3.0, 1.5, 4.0, 1.7, 0.1),
                    (5.0, 6.0, 7.0, 0.8, 1.8, 1.6, -0.2),
                )
            ),
            "scores_3d": np.asarray((0.9, 0.2)),
            "labels_3d": np.asarray((0, 2)),
        }
    }

    adapter = MMDet3DPointPillarsAdapter(
        model=object(),
        inference_fn=lambda _model, points: output,
        score_threshold=0.5,
    )
    detections = adapter.predict(
        FrameData("0000", 0, None, np.zeros((1, 4), dtype=np.float32), None)
    )

    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].position, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(detections[0].dimensions, (4.0, 1.5, 1.7))
    assert detections[0].class_id == 2
    assert detections[0].score == pytest.approx(0.9)


def test_mmdet3d_adapter_accepts_object_prediction_wrapper() -> None:
    class Instances:
        bboxes_3d = type("Boxes", (), {"tensor": np.asarray(((1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 0.0),))})()
        scores_3d = np.asarray((0.8,))
        labels_3d = np.asarray((1,))

    class Result:
        pred_instances_3d = Instances()

    adapter = MMDet3DPointPillarsAdapter(
        model=object(),
        inference_fn=lambda _model, points: Result(),
    )

    detections = adapter.predict(FrameData("0000", 0, None, np.zeros((1, 4)), None))

    assert len(detections) == 1
    assert detections[0].class_id == 3


def test_mmdet3d_adapter_accepts_v1_tuple_result() -> None:
    result = [
        {
            "boxes_3d": np.asarray(((1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 0.0),)),
            "scores_3d": np.asarray((0.8,)),
            "labels_3d": np.asarray((2,)),
        }
    ]
    adapter = MMDet3DPointPillarsAdapter(
        model=object(),
        inference_fn=lambda _model, points: (result, {"metadata": "unused"}),
    )

    detections = adapter.predict(FrameData("0000", 0, None, np.zeros((1, 4)), None))

    assert len(detections) == 1
    assert detections[0].class_id == 1


def test_mmdet3d_adapter_rejects_missing_real_dependencies_or_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config_path and checkpoint_path"):
        MMDet3DPointPillarsAdapter()
    with pytest.raises(FileNotFoundError, match="config"):
        MMDet3DPointPillarsAdapter(tmp_path / "config.py", tmp_path / "model.pth")
