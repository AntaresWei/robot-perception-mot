"""Tests for gated Hungarian association and the Euclidean baseline."""

from __future__ import annotations

import numpy as np

from robot_perception.tracking.association import (
    associate_detections_to_tracks,
    euclidean_nearest_neighbor,
    hungarian_association,
    mahalanobis_association,
)


def _states(positions: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    states = np.zeros((len(positions), 6), dtype=np.float64)
    states[:, :3] = np.asarray(positions)
    return states


def _covariances(count: int, scale: float = 0.01) -> np.ndarray:
    return np.broadcast_to(np.eye(6) * scale, (count, 6, 6)).copy()


def test_perfect_mahalanobis_matching() -> None:
    result = associate_detections_to_tracks(
        _states(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
        _covariances(2),
        np.asarray(((0.1, 0.0, 0.0), (9.9, 0.0, 0.0))),
        measurement_noise=np.eye(3) * 0.01,
        gating_threshold=10.0,
    )

    np.testing.assert_array_equal(result.matches, ((0, 0), (1, 1)))
    assert result.unmatched_tracks.size == 0
    assert result.unmatched_detections.size == 0


def test_ambiguous_matching_is_one_to_one_and_minimizes_cost() -> None:
    result = hungarian_association(
        np.asarray(((1.0, 2.0), (2.0, 100.0))),
    )

    np.testing.assert_array_equal(result.matches, ((0, 1), (1, 0)))


def test_unequal_sizes_return_unmatched_indices() -> None:
    result = hungarian_association(np.asarray(((0.1, 5.0, 9.0), (8.0, 0.2, 9.0))))

    np.testing.assert_array_equal(result.matches, ((0, 0), (1, 1)))
    np.testing.assert_array_equal(result.unmatched_detections, (2,))
    assert result.unmatched_tracks.size == 0


def test_empty_inputs_are_supported() -> None:
    tracks_empty = hungarian_association(np.empty((0, 3)))
    detections_empty = hungarian_association(np.empty((3, 0)))

    assert tracks_empty.matches.shape == (0, 2)
    assert detections_empty.matches.shape == (0, 2)
    assert tracks_empty.unmatched_tracks.size == 0
    np.testing.assert_array_equal(detections_empty.unmatched_tracks, (0, 1, 2))
    assert detections_empty.unmatched_detections.size == 0


def test_gated_pairs_are_not_matched() -> None:
    result = mahalanobis_association(
        np.asarray(((0.0, 0.0, 0.0), (100.0, 0.0, 0.0))),
        np.asarray(((0.1, 0.0, 0.0), (100.1, 0.0, 0.0))),
        np.broadcast_to(np.eye(3) * 0.01, (2, 3, 3)).copy(),
        threshold=2.0,
    )

    np.testing.assert_array_equal(result.matches, ((0, 0), (1, 1)))
    assert result.valid_mask is not None
    assert result.valid_mask[0, 1] is np.False_


def test_no_valid_pairs_leave_everything_unmatched() -> None:
    result = mahalanobis_association(
        np.asarray(((0.0, 0.0, 0.0),)),
        np.asarray(((100.0, 100.0, 100.0),)),
        np.eye(3) * 0.01,
        threshold=1.0,
    )

    assert result.matches.shape == (0, 2)
    np.testing.assert_array_equal(result.unmatched_tracks, (0,))
    np.testing.assert_array_equal(result.unmatched_detections, (0,))


def test_euclidean_nearest_neighbor_baseline_is_available() -> None:
    result = euclidean_nearest_neighbor(
        np.asarray(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))),
        np.asarray(((0.2, 0.0, 0.0), (5.1, 0.0, 0.0))),
        max_distance=1.0,
    )

    np.testing.assert_array_equal(result.matches, ((0, 0), (1, 1)))
