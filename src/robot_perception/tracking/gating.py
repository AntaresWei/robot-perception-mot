"""Mahalanobis-distance gating for detection-to-track candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GatingResult:
    """Pairwise squared distances and the corresponding valid-candidate mask."""

    distances: np.ndarray
    valid_mask: np.ndarray


def _vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    return vector


def _covariance(value: Any, dimension: int, name: str) -> np.ndarray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension}), got {covariance.shape}")
    if not np.isfinite(covariance).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    if not np.allclose(covariance, covariance.T, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return (covariance + covariance.T) * 0.5


def mahalanobis_distance(
    measurement_or_innovation: Any,
    predicted_measurement_or_covariance: Any,
    innovation_covariance: Any | None = None,
) -> float:
    """Return ``y.T @ S^-1 @ y`` without explicitly forming ``S^-1``.

    The function accepts either ``(measurement, predicted_measurement, S)`` or
    ``(innovation, S)`` for convenient use in gating and unit tests.
    """

    if innovation_covariance is None:
        innovation = _vector(measurement_or_innovation, "innovation")
        covariance = _covariance(predicted_measurement_or_covariance, innovation.size, "innovation_covariance")
    else:
        measurement = _vector(measurement_or_innovation, "measurement")
        predicted = _vector(predicted_measurement_or_covariance, "predicted_measurement")
        if measurement.shape != predicted.shape:
            raise ValueError("measurement and predicted_measurement must have the same shape")
        innovation = measurement - predicted
        covariance = _covariance(innovation_covariance, innovation.size, "innovation_covariance")
    try:
        solved = np.linalg.solve(covariance, innovation)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError("innovation covariance is singular") from error
    distance = float(innovation @ solved)
    if not np.isfinite(distance):
        raise FloatingPointError("Mahalanobis distance is NaN or Inf")
    return max(distance, 0.0)


mahalanobis_distance_squared = mahalanobis_distance


def mahalanobis_gating(
    predicted_measurements: np.ndarray,
    detections: np.ndarray,
    innovation_covariances: np.ndarray,
    *,
    threshold: float,
) -> GatingResult:
    """Compute vectorized pairwise Mahalanobis distances and gate candidates.

    ``innovation_covariances`` may be one ``(D, D)`` matrix shared by all
    tracks or one ``(N, D, D)`` matrix per predicted measurement.
    Non-finite measurement rows become invalid candidates rather than raising.
    """

    predicted = np.asarray(predicted_measurements, dtype=np.float64)
    observed = np.asarray(detections, dtype=np.float64)
    if predicted.ndim != 2 or observed.ndim != 2:
        raise ValueError("predicted_measurements and detections must be 2D arrays")
    if predicted.shape[1] != observed.shape[1]:
        raise ValueError("predicted_measurements and detections must have the same dimension")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    dimension = predicted.shape[1]
    covariance = np.asarray(innovation_covariances, dtype=np.float64)
    if covariance.ndim == 2:
        covariance = np.broadcast_to(covariance, (predicted.shape[0], dimension, dimension)).copy()
    elif covariance.shape != (predicted.shape[0], dimension, dimension):
        raise ValueError(
            "innovation_covariances must have shape "
            f"({dimension}, {dimension}) or ({predicted.shape[0]}, {dimension}, {dimension}), got {covariance.shape}"
        )
    if not np.isfinite(covariance).all():
        raise ValueError("innovation_covariances contains NaN or Inf values")
    if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-12):
        raise ValueError("innovation_covariances must be symmetric")

    num_tracks, num_detections = predicted.shape[0], observed.shape[0]
    distances = np.full((num_tracks, num_detections), np.inf, dtype=np.float64)
    valid_rows = np.isfinite(predicted).all(axis=1)
    valid_detections = np.isfinite(observed).all(axis=1)
    residuals = observed[None, :, :] - predicted[:, None, :]
    for track_index in np.flatnonzero(valid_rows):
        detection_indices = np.flatnonzero(valid_detections)
        if detection_indices.size == 0:
            continue
        try:
            solved = np.linalg.solve(covariance[track_index], residuals[track_index, detection_indices].T).T
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                f"innovation covariance is singular for track index {track_index}"
            ) from error
        distances[track_index, detection_indices] = np.einsum(
            "ij,ij->i", residuals[track_index, detection_indices], solved
        )
    distances = np.maximum(distances, 0.0)
    valid_mask = np.isfinite(distances) & (distances <= threshold)
    return GatingResult(distances=distances, valid_mask=valid_mask)


gate_measurements = mahalanobis_gating
