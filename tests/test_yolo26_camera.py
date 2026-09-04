"""Tests for the dependency-light YOLO26 ONNX camera detector adapter."""

from __future__ import annotations

import numpy as np

from robot_perception.common.types import FrameData
from robot_perception.fusion import Yolo26OnnxCameraDetector


class _FakeNet:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.input_shape: tuple[int, ...] | None = None

    def setInput(self, value: np.ndarray) -> None:
        self.input_shape = value.shape

    def forward(self) -> np.ndarray:
        return self.output


def _frame() -> FrameData:
    return FrameData("0000", 0, np.zeros((100, 200, 3), dtype=np.uint8), np.empty((0, 4)), None)


def test_yolo26_adapter_filters_to_kitti_classes_and_reverses_letterbox() -> None:
    net = _FakeNet(np.asarray([[[160.0, 240.0, 480.0, 400.0, 0.9, 0.0], [0.0, 0.0, 20.0, 20.0, 0.95, 7.0]]]))
    detector = Yolo26OnnxCameraDetector("unused.onnx", input_size=640, score_threshold=0.25, net=net)

    detections = detector.predict(_frame())

    assert net.input_shape == (1, 3, 640, 640)
    assert len(detections) == 1
    assert detections[0].class_id == 1
    np.testing.assert_allclose(detections[0].bbox, (50.0, 25.0, 150.0, 75.0), atol=1.0)


def test_yolo26_adapter_accepts_transposed_rows_and_empty_image() -> None:
    net = _FakeNet(np.asarray([[[160.0], [240.0], [480.0], [400.0], [0.9], [2.0]]]))
    detector = Yolo26OnnxCameraDetector("unused.onnx", input_size=640, net=net)

    detections = detector.predict(_frame())

    assert len(detections) == 1
    assert detections[0].class_id == 3
    assert detector.predict(FrameData("0000", 1, None, np.empty((0, 4)), None)) == []
