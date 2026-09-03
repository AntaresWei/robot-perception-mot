"""Single-object track state and lifecycle management."""

from __future__ import annotations

import numpy as np

from robot_perception.common.types import Detection3D, TrackState
from robot_perception.tracking.kalman import ConstantVelocityKalmanFilter


class Track:
    """Wrap one constant-velocity KF and expose a track lifecycle."""

    TENTATIVE = "Tentative"
    CONFIRMED = "Confirmed"
    LOST = "Lost"
    DELETED = "Deleted"

    def __init__(
        self,
        track_id: int,
        detection: Detection3D,
        *,
        min_hits: int,
        initial_covariance: np.ndarray | float,
        process_noise: np.ndarray | float,
        measurement_noise: np.ndarray | float,
    ) -> None:
        if track_id < 0:
            raise ValueError("track_id must be non-negative")
        if min_hits < 1:
            raise ValueError("min_hits must be at least 1")
        self.track_id = int(track_id)
        self.min_hits = int(min_hits)
        self._filter = ConstantVelocityKalmanFilter(
            self._validate_detection(detection).position,
            initial_covariance=initial_covariance,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        self.dimensions = detection.dimensions.astype(np.float64, copy=True)
        # KF 只估计位置和速度，尺寸/朝向/类别保留最近一次观测值。
        self.yaw = float(detection.yaw)
        self.score = float(detection.score)
        self.class_id = int(detection.class_id)
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.status = self.CONFIRMED if min_hits == 1 else self.TENTATIVE

    @staticmethod
    def _validate_detection(detection: Detection3D) -> Detection3D:
        if not isinstance(detection, Detection3D):
            raise TypeError("track updates require Detection3D instances")
        if detection.position.shape != (3,) or detection.dimensions.shape != (3,):
            raise ValueError("Detection3D position and dimensions must have shape (3,)")
        if not np.isfinite(detection.position).all() or not np.isfinite(detection.dimensions).all():
            raise ValueError("Detection3D position and dimensions must be finite")
        if (detection.dimensions <= 0.0).any() or not np.isfinite(detection.yaw):
            raise ValueError("Detection3D dimensions must be positive and yaw finite")
        return detection

    @property
    def state(self) -> np.ndarray:
        return self._filter.state

    @property
    def covariance(self) -> np.ndarray:
        return self._filter.covariance

    @property
    def measurement_noise(self) -> np.ndarray:
        return self._filter.measurement_noise

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Predict one frame and age the track before association."""

        if self.status == self.DELETED:
            raise RuntimeError("cannot predict a deleted track")
        # 每个 frame 先预测一次；若之后没有匹配，time_since_update 会继续保留。
        self.age += 1
        self.time_since_update += 1
        return self._filter.predict(dt)

    def update(self, detection: Detection3D) -> np.ndarray:
        """Correct the KF and promote tentative/lost tracks when eligible."""

        if self.status == self.DELETED:
            raise RuntimeError("cannot update a deleted track")
        detection = self._validate_detection(detection)
        state = self._filter.update(detection.position)
        self.dimensions = detection.dimensions.astype(np.float64, copy=True)
        self.yaw = float(detection.yaw)
        self.score = float(detection.score)
        self.class_id = int(detection.class_id)
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.min_hits:
            self.status = self.CONFIRMED
        return state

    def mark_missed(self, max_age: int) -> None:
        """Move an unmatched track to Lost or Deleted based on missed age."""

        if max_age < 1:
            raise ValueError("max_age must be at least 1")
        if self.status == self.DELETED:
            return
        # Lost 轨迹仍可在 max_age 内被后续观测恢复，超过窗口才永久删除。
        self.status = self.DELETED if self.time_since_update >= max_age else self.LOST

    def to_track_state(self) -> TrackState:
        """Return the stable cross-module TrackState data contract."""

        return TrackState(
            track_id=self.track_id,
            state=self.state,
            covariance=self.covariance,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            status=self.status,
        )
