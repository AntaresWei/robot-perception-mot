"""Object-level Camera-LiDAR projection, association, and camera adapters."""

from robot_perception.fusion.camera_lidar import (
    BaseCameraDetector,
    Camera2DDetection,
    FusedDetection,
    FusionResult,
    SyntheticCameraDetector,
    associate_camera_lidar_detections,
    box_corners_3d,
    project_detection_corners_to_image,
    project_detection_to_image,
)
from robot_perception.fusion.yolo26 import Yolo26OnnxCameraDetector

__all__ = [
    "BaseCameraDetector",
    "Camera2DDetection",
    "FusedDetection",
    "FusionResult",
    "SyntheticCameraDetector",
    "Yolo26OnnxCameraDetector",
    "associate_camera_lidar_detections",
    "box_corners_3d",
    "project_detection_corners_to_image",
    "project_detection_to_image",
]

from robot_perception.fusion.camera_lidar import (
    BaseCameraDetector,
    Camera2DDetection,
    FusedDetection,
    FusionResult,
    SyntheticCameraDetector,
    associate_camera_lidar_detections,
    box_corners_3d,
    project_detection_corners_to_image,
    project_detection_to_image,
)

__all__ = [
    "BaseCameraDetector",
    "Camera2DDetection",
    "FusedDetection",
    "FusionResult",
    "SyntheticCameraDetector",
    "associate_camera_lidar_detections",
    "box_corners_3d",
    "project_detection_corners_to_image",
    "project_detection_to_image",
]
