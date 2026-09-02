"""Detector interface and common KITTI class definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from robot_perception.common.types import Detection3D, FrameData


# 将语义相近的 KITTI 标签归到稳定 ID，Tracker 只依赖数值类别而不读取原始文本。
KITTI_CLASS_IDS: dict[str, int] = {
    "Car": 1,
    "Van": 1,
    "Truck": 1,
    "Pedestrian": 2,
    "Person_sitting": 2,
    "Cyclist": 3,
    "Tram": 4,
    "Misc": 5,
}

KITTI_CLASS_NAMES: dict[int, str] = {
    1: "Car",
    2: "Pedestrian",
    3: "Cyclist",
    4: "Tram",
    5: "Misc",
}


class BaseDetector(ABC):
    """Detector boundary consumed by later tracking code."""

    @abstractmethod
    def predict(self, frame: FrameData) -> list[Detection3D]:
        """Return normalized 3D detections for a single synchronized frame."""


def class_name_for_id(class_id: int) -> str:
    """Return a human-readable class label for CLI summaries."""

    return KITTI_CLASS_NAMES.get(class_id, f"unknown:{class_id}")
