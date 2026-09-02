"""Tests for Mahalanobis distance and gating."""

from __future__ import annotations

import numpy as np
import pytest

from robot_perception.tracking.gating import mahalanobis_distance, mahalanobis_gating


def test_close_measurement_is_inside_gate() -> None:
    result = mahalanobis_gating(
        np.asarray(((0.0, 0.0, 0.0),)),
        np.asarray(((0.5, 0.0, 0.0),)),
        np.eye(3),
        threshold=1.0,
    )

    np.testing.assert_allclose(result.distances, ((0.25,),))
    assert result.valid_mask.tolist() == [[True]]


def test_distant_measurement_is_outside_gate() -> None:
    result = mahalanobis_gating(
        np.asarray(((0.0, 0.0, 0.0),)),
        np.asarray(((3.0, 0.0, 0.0),)),
        np.eye(3),
        threshold=1.0,
    )

    np.testing.assert_allclose(result.distances, ((9.0,),))
    assert result.valid_mask.tolist() == [[False]]


def test_covariance_changes_mahalanobis_distance() -> None:
    innovation = np.asarray((2.0, 0.0, 0.0))

    low_uncertainty = mahalanobis_distance(innovation, np.eye(3))
    high_uncertainty = mahalanobis_distance(innovation, np.eye(3) * 4.0)

    assert low_uncertainty == pytest.approx(4.0)
    assert high_uncertainty == pytest.approx(1.0)


def test_measurement_and_prediction_form_matches_innovation_form() -> None:
    covariance = np.diag((2.0, 1.0, 4.0))

    direct = mahalanobis_distance(np.asarray((3.0, 1.0, 2.0)), np.asarray((1.0, 1.0, 0.0)), covariance)
    innovation = mahalanobis_distance(np.asarray((2.0, 0.0, 2.0)), covariance)

    assert direct == pytest.approx(innovation)


def test_gating_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        mahalanobis_gating(np.zeros((1, 3)), np.zeros((1, 2)), np.eye(3), threshold=1.0)
