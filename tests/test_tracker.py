"""Tests for complete 3D multi-object tracker frame updates."""

from __future__ import annotations

import numpy as np
import pytest

from robot_perception.common.types import Detection3D
from robot_perception.tracking import MultiObjectTracker, Track


def _detection(position, class_id=1) -> Detection3D:
    return Detection3D(np.asarray(position, dtype=np.float64), np.ones(3), 0.0, 1.0, class_id)


def _tracker(**kwargs: object) -> MultiObjectTracker:
    return MultiObjectTracker(
        min_hits=kwargs.pop("min_hits", 1),
        max_age=kwargs.pop("max_age", 3),
        gating_threshold=kwargs.pop("gating_threshold", 16.0),
        initial_covariance=np.eye(6),
        process_noise=np.eye(6) * 1e-3,
        measurement_noise=np.eye(3) * 1e-2,
        **kwargs,
    )


def test_single_object_is_created_and_updated() -> None:
    tracker = _tracker()
    first = tracker.update([_detection((0.0, 0.0, 0.0))])
    second = tracker.update([_detection((1.0, 0.0, 0.0))])

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id == 1
    assert second[0].state[0] > 0.0


def test_multiple_objects_keep_distinct_ids() -> None:
    tracker = _tracker()
    states = tracker.update([_detection((0.0, 0.0, 0.0)), _detection((10.0, 0.0, 0.0))])

    assert [state.track_id for state in states] == [1, 2]
    assert len({state.track_id for state in states}) == 2


def test_new_object_birth_gets_new_monotonic_id() -> None:
    tracker = _tracker()
    tracker.update([_detection((0.0, 0.0, 0.0))])
    states = tracker.update([_detection((0.5, 0.0, 0.0)), _detection((20.0, 0.0, 0.0))])

    assert [state.track_id for state in states] == [1, 2]
    assert tracker.all_tracks[-1].track_id == 2


def test_temporary_missed_detection_keeps_track_then_recovers() -> None:
    tracker = _tracker(max_age=3)
    tracker.update([_detection((0.0, 0.0, 0.0))])
    assert tracker.update([]) == []
    recovered = tracker.update([_detection((1.0, 0.0, 0.0))])

    assert len(recovered) == 1
    assert recovered[0].track_id == 1
    assert tracker.all_tracks[0].status == Track.CONFIRMED


def test_track_is_deleted_after_permanent_disappearance() -> None:
    tracker = _tracker(max_age=2)
    tracker.update([_detection((0.0, 0.0, 0.0))])
    tracker.update([])
    tracker.update([])

    assert tracker.tracks == ()
    assert tracker.all_tracks[0].status == Track.DELETED


def test_deleted_id_is_not_reused_for_later_birth() -> None:
    tracker = _tracker(max_age=1)
    tracker.update([_detection((0.0, 0.0, 0.0))])
    tracker.update([])
    states = tracker.update([_detection((50.0, 0.0, 0.0))])

    assert [state.track_id for state in states] == [2]
    assert [track.track_id for track in tracker.all_tracks] == [1, 2]


def test_crossing_trajectories_keep_ids_with_motion_prediction() -> None:
    tracker = _tracker(gating_threshold=25.0)
    frames = [
        [_detection((-2.0, 0.0, 0.0)), _detection((2.0, 0.0, 0.0))],
        [_detection((-1.0, 0.0, 0.0)), _detection((1.0, 0.0, 0.0))],
        [_detection((-0.1, 0.0, 0.0)), _detection((0.1, 0.0, 0.0))],
        [_detection((1.0, 0.0, 0.0)), _detection((-1.0, 0.0, 0.0))],
    ]
    first = tracker.update(frames[0])
    ids = {round(state.state[0], 1): state.track_id for state in first}
    for detections in frames[1:]:
        states = tracker.update(detections)
        assert len(states) == 2

    assert len(ids) == 2
    assert {state.track_id for state in states} == {1, 2}


def test_tracker_rejects_non_detection_input() -> None:
    tracker = _tracker()

    with pytest.raises(TypeError):
        tracker.update([np.zeros(3)])
