"""3D multi-object tracker built from the project's KF and association APIs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from robot_perception.common.types import Detection3D, TrackState
from robot_perception.tracking.association import AssociationResult, associate_detections_to_tracks
from robot_perception.tracking.track import Track


class MultiObjectTracker:
    """Manage Track birth, association, aging, confirmation, and deletion."""

    def __init__(
        self,
        *,
        min_hits: int = 3,
        max_age: int = 5,
        gating_threshold: float = 9.21,
        initial_covariance: np.ndarray | float = 1.0,
        process_noise: np.ndarray | float = 1e-2,
        measurement_noise: np.ndarray | float = 1e-1,
    ) -> None:
        if min_hits < 1 or max_age < 1:
            raise ValueError("min_hits and max_age must be at least 1")
        if not np.isfinite(gating_threshold) or gating_threshold < 0.0:
            raise ValueError("gating_threshold must be finite and non-negative")
        self.min_hits = int(min_hits)
        self.max_age = int(max_age)
        self.gating_threshold = float(gating_threshold)
        self.initial_covariance = initial_covariance
        self.process_noise = process_noise
        # association 需要显式的 3x3 R；这里将 YAML 中常用的标量/对角向量展开。
        self.measurement_noise = self._measurement_covariance(measurement_noise)
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1
        self.frame_index = 0
        self.last_association: AssociationResult | None = None

    @staticmethod
    def _measurement_covariance(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            matrix = np.eye(3, dtype=np.float64) * float(array)
        elif array.shape == (3,):
            matrix = np.diag(array)
        elif array.shape == (3, 3):
            matrix = array.copy()
        else:
            raise ValueError(f"measurement_noise must be scalar, (3,), or (3, 3), got {array.shape}")
        if not np.isfinite(matrix).all() or (np.linalg.eigvalsh((matrix + matrix.T) * 0.5) < 0.0).any():
            raise ValueError("measurement_noise must be finite and positive semidefinite")
        return (matrix + matrix.T) * 0.5

    @property
    def tracks(self) -> tuple[Track, ...]:
        """Return non-deleted tracks, including Lost tracks awaiting expiry."""

        return tuple(track for track in self._tracks.values() if track.status != Track.DELETED)

    @property
    def all_tracks(self) -> tuple[Track, ...]:
        """Return every track object, including deleted history for inspection."""

        return tuple(self._tracks.values())

    def predict(self, dt: float = 1.0) -> tuple[TrackState, ...]:
        """Predict all active tracks without performing association."""

        for track in self.tracks:
            track.predict(dt)
        return tuple(track.to_track_state() for track in self.tracks)

    @staticmethod
    def _validate_detections(detections: Iterable[Detection3D]) -> list[Detection3D]:
        values = list(detections)
        if any(not isinstance(detection, Detection3D) for detection in values):
            raise TypeError("tracker input must contain Detection3D instances")
        return values

    def _create_track(self, detection: Detection3D) -> Track:
        track = Track(
            self._next_track_id,
            detection,
            min_hits=self.min_hits,
            initial_covariance=self.initial_covariance,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def update(self, detections: Iterable[Detection3D], *, dt: float = 1.0) -> list[TrackState]:
        """Run one complete tracking frame and return confirmed TrackState values."""

        observations = self._validate_detections(detections)
        active_tracks = list(self.tracks)
        # 使用列表快照保证本帧关联索引在新增/删除 Track 时仍保持稳定。
        for track in active_tracks:
            track.predict(dt)

        if active_tracks:
            states = np.stack([track.state for track in active_tracks])
            covariances = np.stack([track.covariance for track in active_tracks])
            detection_positions = (
                np.stack([detection.position for detection in observations])
                if observations
                else np.empty((0, 3), dtype=np.float64)
            )
            association = associate_detections_to_tracks(
                states,
                covariances,
                detection_positions,
                measurement_noise=np.asarray(self.measurement_noise, dtype=np.float64),
                gating_threshold=self.gating_threshold,
            )
        else:
            association = AssociationResult(
                matches=np.empty((0, 2), dtype=np.int64),
                unmatched_tracks=np.empty(0, dtype=np.int64),
                unmatched_detections=np.arange(len(observations), dtype=np.int64),
            )
        self.last_association = association

        for track_index, detection_index in association.matches:
            # association 返回的是 active_tracks 的局部索引，不是 Track ID。
            active_tracks[int(track_index)].update(observations[int(detection_index)])
        for track_index in association.unmatched_tracks:
            active_tracks[int(track_index)].mark_missed(self.max_age)
        for detection_index in association.unmatched_detections:
            # 未匹配 detection 代表新目标，分配单调递增 ID。
            self._create_track(observations[int(detection_index)])

        self.frame_index += 1
        return [
            track.to_track_state()
            for track in self.tracks
            if track.status == Track.CONFIRMED and track.time_since_update == 0
        ]

    step = update
