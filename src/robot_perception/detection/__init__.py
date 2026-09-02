"""Detector interfaces and detector-output adapters."""

from robot_perception.detection.adapter import GTDetector, MMDet3DPointPillarsAdapter, NoisyGTDetector
from robot_perception.detection.base import BaseDetector, KITTI_CLASS_IDS, KITTI_CLASS_NAMES, class_name_for_id

__all__ = [
    "BaseDetector",
    "GTDetector",
    "KITTI_CLASS_IDS",
    "KITTI_CLASS_NAMES",
    "MMDet3DPointPillarsAdapter",
    "NoisyGTDetector",
    "class_name_for_id",
]
