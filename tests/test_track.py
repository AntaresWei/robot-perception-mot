"""Tests for one Track lifecycle and state contract."""

from __future__ import annotations

import numpy as np
import pytest

from robot_perception.common.types import Detection3D
from robot_perception.tracking import Track


def _detection(position=(0.0, 0.0, 0.0), velocity=None) -> Detection3D:
    return Detection3D(
        position=np.asarray(position, dtype=np.float64),
        dimensions=np.ones(3),
        yaw=0.0,
        score=0.9,
        class_id=1,
        velocity=None if velocity is None else np.asarray(velocity, dtype=np.float64),
    )


def _track(min_hits=3) -> Track:
    return Track(
        1,
        _detection(),
        min_hits=min_hits,
        initial_covariance=np.eye(6),
        process_noise=np.eye(6) * 1e-3,
        measurement_noise=np.eye(3) * 1e-2,
    )


def test_new_track_starts_tentative_and_maintains_required_fields() -> None:
    track = _track()

    assert track.status == Track.TENTATIVE
    assert track.track_id == 1
    assert track.age == 1
    assert track.hits == 1
    assert track.time_since_update == 0
    assert track.state.shape == (6,)
    assert track.covariance.shape == (6, 6)


def test_update_confirms_track_after_min_hits() -> None:
    track = _track(min_hits=2)
    track.predict()
    track.update(_detection((1.0, 0.0, 0.0)))

    assert track.status == Track.CONFIRMED
    assert track.hits == 2
    assert track.time_since_update == 0
    assert track.state[0] > 0.0


def test_missed_track_becomes_lost_then_deleted() -> None:
    track = _track(min_hits=1)
    track.predict()
    track.mark_missed(max_age=2)
    assert track.status == Track.LOST
    track.predict()
    track.mark_missed(max_age=2)
    assert track.status == Track.DELETED


def test_deleted_track_cannot_be_reused() -> None:
    track = _track()
    track.predict()
    track.mark_missed(max_age=1)

    with pytest.raises(RuntimeError):
        track.predict()
    with pytest.raises(RuntimeError):
        track.update(_detection())


def test_invalid_detection_is_rejected() -> None:
    track = _track()
    invalid = Detection3D(np.zeros(2), np.ones(3), 0.0, 0.9, 1)

    with pytest.raises(ValueError):
        track.update(invalid)
