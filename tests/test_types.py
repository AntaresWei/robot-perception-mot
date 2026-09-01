"""Tests for the shared dataclass contracts."""

from __future__ import annotations

import numpy as np

from robot_perception.common.types import Calibration, Detection3D, FrameData, TrackState


def test_core_dataclasses_preserve_declared_values() -> None:
    calibration = Calibration(p2=np.eye(3, 4))
    frame = FrameData(
        sequence_id="0000",
        frame_id=0,
        image=None,
        lidar_points=np.empty((0, 4)),
        calibration=calibration,
    )
    detection = Detection3D(
        position=np.array((1.0, 2.0, 3.0)),
        dimensions=np.array((4.0, 2.0, 1.5)),
        yaw=0.0,
        score=0.9,
        class_id=1,
    )
    track = TrackState(
        track_id=1,
        state=np.array((1.0, 2.0, 3.0, 0.0, 0.0, 0.0)),
        covariance=np.eye(6),
        age=1,
        hits=1,
        time_since_update=0,
        status="tentative",
    )

    assert frame.calibration is calibration
    assert detection.position.tolist() == [1.0, 2.0, 3.0]
    assert track.track_id == 1
