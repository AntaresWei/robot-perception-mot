"""Tests for the constant-velocity Kalman filter."""

from __future__ import annotations

import numpy as np
import pytest

from robot_perception.tracking import ConstantVelocityKalmanFilter


def _filter(**kwargs: object) -> ConstantVelocityKalmanFilter:
    return ConstantVelocityKalmanFilter(
        np.zeros(3),
        initial_covariance=np.eye(6),
        process_noise=np.eye(6) * 1e-3,
        measurement_noise=np.eye(3) * 1e-2,
        **kwargs,
    )


def test_initialization() -> None:
    kalman = _filter(initial_velocity=np.asarray((1.0, 2.0, 3.0)))

    np.testing.assert_allclose(kalman.state, (0.0, 0.0, 0.0, 1.0, 2.0, 3.0))
    assert kalman.covariance.shape == (6, 6)
    assert kalman.measurement_matrix.shape == (3, 6)
    assert kalman.transition_matrix.shape == (6, 6)


def test_prediction_updates_position_using_velocity() -> None:
    kalman = _filter(initial_velocity=np.asarray((1.0, -2.0, 0.5)))

    predicted = kalman.predict()

    np.testing.assert_allclose(predicted[:3], (1.0, -2.0, 0.5))
    np.testing.assert_allclose(predicted[3:], (1.0, -2.0, 0.5))


def test_update_moves_state_toward_measurement() -> None:
    kalman = _filter()
    kalman.predict()

    updated = kalman.update(np.asarray((10.0, 0.0, 0.0)))

    assert 0.0 < updated[0] < 10.0
    assert updated[3] > 0.0
    np.testing.assert_allclose(updated[[1, 2, 4, 5]], 0.0, atol=1e-12)


def test_variable_dt_changes_transition_and_prediction() -> None:
    kalman = _filter(initial_velocity=np.asarray((2.0, 0.0, 0.0)))

    predicted = kalman.predict(dt=0.25)

    np.testing.assert_allclose(predicted[0], 0.5)
    np.testing.assert_allclose(kalman.transition_matrix[0, 3], 0.25)


def test_covariance_remains_symmetric_positive_definite() -> None:
    kalman = _filter()
    for measurement in (np.asarray((1.0, 0.0, 0.0)), np.asarray((2.0, 0.0, 0.0)), np.asarray((3.0, 0.0, 0.0))):
        kalman.predict(dt=0.5)
        kalman.update(measurement)

    covariance = kalman.covariance
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    assert np.isfinite(covariance).all()


def test_filter_reduces_noisy_measurement_rmse() -> None:
    rng = np.random.default_rng(3)
    steps = np.arange(60, dtype=np.float64)
    truth = np.column_stack((0.7 * steps, -0.2 * steps, 0.1 * steps))
    measurements = truth + rng.normal(0.0, 0.8, size=truth.shape)
    kalman = ConstantVelocityKalmanFilter(
        measurements[0],
        initial_covariance=np.eye(6) * 10.0,
        process_noise=np.eye(6) * 1e-3,
        measurement_noise=np.eye(3) * 0.64,
    )
    filtered = [kalman.state[:3]]
    for measurement in measurements[1:]:
        kalman.predict()
        filtered.append(kalman.update(measurement)[:3])

    raw_rmse = float(np.sqrt(np.mean((measurements - truth) ** 2)))
    filtered_rmse = float(np.sqrt(np.mean((np.asarray(filtered) - truth) ** 2)))
    assert filtered_rmse < raw_rmse


@pytest.mark.parametrize(
    "constructor_kwargs",
    [
        {"initial_position": np.zeros(2)},
        {"initial_position": np.full(3, np.nan)},
        {"initial_position": np.zeros(3), "initial_covariance": np.eye(5)},
        {"initial_position": np.zeros(3), "measurement_noise": np.diag((-1.0, 1.0, 1.0))},
    ],
)
def test_invalid_initialization_is_rejected(constructor_kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ConstantVelocityKalmanFilter(**constructor_kwargs)


def test_invalid_predict_and_update_inputs_are_rejected() -> None:
    kalman = _filter()

    with pytest.raises(ValueError, match="dt"):
        kalman.predict(dt=0.0)
    with pytest.raises(ValueError, match="measurement"):
        kalman.update(np.zeros(2))
    with pytest.raises(ValueError, match="measurement"):
        kalman.update(np.full(3, np.inf))
