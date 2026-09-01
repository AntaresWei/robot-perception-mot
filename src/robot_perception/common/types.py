"""Dependency-free shared data contracts for the initial project scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

Vector: TypeAlias = np.ndarray
Matrix: TypeAlias = np.ndarray
PointCloud: TypeAlias = np.ndarray


@dataclass
class Calibration:
    """Camera-LiDAR calibration matrices, expressed in their documented frames."""

    projection_matrix: Matrix | None = None
    rectification_matrix: Matrix | None = None
    lidar_to_camera_matrix: Matrix | None = None


@dataclass
class FrameData:
    """One synchronized KITTI frame shared across pipeline stages."""

    sequence_id: str
    frame_id: int
    image: np.ndarray | None
    lidar_points: PointCloud
    calibration: Calibration | None
    annotations: list[object] | None = None
    timestamp: float | None = None


@dataclass
class Detection3D:
    """A detector-agnostic 3D object observation."""

    position: Vector
    dimensions: Vector
    yaw: float
    score: float
    class_id: int
    velocity: Vector | None = None


@dataclass
class TrackState:
    """The state snapshot of one multi-object track."""

    track_id: int
    state: Vector
    covariance: Matrix
    age: int
    hits: int
    time_since_update: int
    status: str
