"""OpenCV DNN adapter for a KITTI three-class YOLO26 ONNX camera detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_perception.common.types import FrameData
from robot_perception.fusion.camera_lidar import BaseCameraDetector, Camera2DDetection


class Yolo26OnnxCameraDetector(BaseCameraDetector):
    """Run a fine-tuned YOLO26 ONNX model without adding runtime dependencies.

    The exported YOLO26 graph returns rows in ``[x1, y1, x2, y2, score, cls]``
    letterbox-input coordinates. ``class_mapping`` maps the three YOLO training
    labels (Car/Pedestrian/Cyclist) to the project's stable class IDs.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        class_mapping: dict[int, int] | None = None,
        net: Any | None = None,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0 or not np.isfinite(score_threshold):
            raise ValueError("score_threshold must be finite and within [0, 1]")
        if not 0.0 <= iou_threshold <= 1.0 or not np.isfinite(iou_threshold):
            raise ValueError("iou_threshold must be finite and within [0, 1]")
        if input_size < 32:
            raise ValueError("input_size must be at least 32")
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = int(input_size)
        self.class_mapping = dict({0: 1, 1: 2, 2: 3} if class_mapping is None else class_mapping)
        if not self.class_mapping or any(int(key) < 0 or int(value) <= 0 for key, value in self.class_mapping.items()):
            raise ValueError("class_mapping must map non-negative model IDs to positive project class IDs")
        self.model_path = Path(model_path)
        if net is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"YOLO26 ONNX model not found: {self.model_path}")
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("OpenCV is required for YOLO26 ONNX inference") from error
            net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self._net = net

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Resize with centered padding and retain the reverse pixel transform."""

        import cv2

        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.input_size - resized_width) / 2.0
        pad_y = (self.input_size - resized_height) / 2.0
        left, top = round(pad_x - 0.1), round(pad_y - 0.1)
        right, bottom = round(pad_x + 0.1), round(pad_y + 0.1)
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)), scale, float(left), float(top)

    @staticmethod
    def _rows(output: Any) -> np.ndarray:
        """Normalize common YOLO26 ONNX output layouts to ``(N, 6)`` rows."""

        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"YOLO26 ONNX output must be 2D after batch removal, got {array.shape}")
        if array.shape[1] == 6:
            rows = array
        elif array.shape[0] == 6:
            rows = array.T
        else:
            raise ValueError(f"YOLO26 ONNX output must have a 6-value dimension, got {array.shape}")
        if not np.isfinite(rows).all():
            raise ValueError("YOLO26 ONNX output contains NaN or Inf")
        return rows

    def predict(self, frame: FrameData) -> list[Camera2DDetection]:
        """Infer KITTI Car/Pedestrian/Cyclist 2D boxes in original image pixels."""

        if frame.image is None:
            return []
        image = np.asarray(frame.image)
        if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
            raise ValueError("frame.image must be a finite BGR image with shape (H, W, 3)")
        import cv2

        letterboxed, scale, pad_x, pad_y = self._letterbox(image)
        blob = cv2.dnn.blobFromImage(letterboxed, 1.0 / 255.0, (self.input_size, self.input_size), swapRB=True)
        self._net.setInput(blob)
        rows = self._rows(self._net.forward())
        height, width = image.shape[:2]
        candidates: dict[int, list[tuple[list[int], float, np.ndarray]]] = {}
        for x1, y1, x2, y2, score, model_class in rows:
            source_class = int(round(float(model_class)))
            class_id = self.class_mapping.get(source_class)
            if class_id is None or score < self.score_threshold:
                continue
            bbox = np.asarray(((x1 - pad_x) / scale, (y1 - pad_y) / scale, (x2 - pad_x) / scale, (y2 - pad_y) / scale))
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, width - 1.0)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, height - 1.0)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            xywh = [int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2] - bbox[0])), int(round(bbox[3] - bbox[1]))]
            candidates.setdefault(class_id, []).append((xywh, float(score), bbox))

        detections: list[Camera2DDetection] = []
        for class_id, values in candidates.items():
            kept = cv2.dnn.NMSBoxes([value[0] for value in values], [value[1] for value in values], self.score_threshold, self.iou_threshold)
            for index in np.asarray(kept, dtype=np.int64).reshape(-1):
                _, score, bbox = values[int(index)]
                detections.append(Camera2DDetection(bbox, score, class_id))
        return sorted(detections, key=lambda detection: (-detection.score, detection.class_id))
