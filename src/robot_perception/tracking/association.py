"""One-to-one detection-to-track association utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from robot_perception.tracking.gating import GatingResult, mahalanobis_gating


@dataclass(frozen=True)
class AssociationResult:
    """Hungarian association output using original track/detection indices."""

    matches: np.ndarray
    unmatched_tracks: np.ndarray
    unmatched_detections: np.ndarray
    cost_matrix: np.ndarray | None = None
    valid_mask: np.ndarray | None = None


def _cost_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got {matrix.shape}")
    if np.isnan(matrix).any():
        raise ValueError("cost_matrix contains NaN values")
    return matrix


def hungarian_association(
    cost_matrix: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> AssociationResult:
    """Perform one-to-one assignment while rejecting invalid pairs."""

    costs = _cost_matrix(cost_matrix)
    num_tracks, num_detections = costs.shape
    if valid_mask is None:
        valid = np.isfinite(costs)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != costs.shape:
            raise ValueError(f"valid_mask must have shape {costs.shape}, got {valid.shape}")
        valid = valid & np.isfinite(costs)

    matches = np.empty((0, 2), dtype=np.int64)
    if num_tracks and num_detections and valid.any():
        valid_costs = costs[valid]
        max_cost = float(np.max(valid_costs))
        invalid_cost = max(max_cost + abs(max_cost) + 1.0, 1.0)
        assignment_costs = np.where(valid, costs, invalid_cost)
        assigned_tracks, assigned_detections = linear_sum_assignment(assignment_costs)
        assigned_valid = valid[assigned_tracks, assigned_detections]
        matches = np.column_stack((assigned_tracks[assigned_valid], assigned_detections[assigned_valid])).astype(
            np.int64, copy=False
        )

    matched_tracks = matches[:, 0] if len(matches) else np.empty(0, dtype=np.int64)
    matched_detections = matches[:, 1] if len(matches) else np.empty(0, dtype=np.int64)
    unmatched_tracks = np.setdiff1d(np.arange(num_tracks, dtype=np.int64), matched_tracks, assume_unique=True)
    unmatched_detections = np.setdiff1d(
        np.arange(num_detections, dtype=np.int64), matched_detections, assume_unique=True
    )
    return AssociationResult(
        matches=matches,
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_detections,
        cost_matrix=costs.copy(),
        valid_mask=valid.copy(),
    )


def mahalanobis_association(
    predicted_measurements: np.ndarray,
    detections: np.ndarray,
    innovation_covariances: np.ndarray,
    *,
    threshold: float,
) -> AssociationResult:
    """Build a Mahalanobis cost matrix, gate it, and run Hungarian assignment."""

    gating: GatingResult = mahalanobis_gating(
        predicted_measurements,
        detections,
        innovation_covariances,
        threshold=threshold,
    )
    return hungarian_association(gating.distances, valid_mask=gating.valid_mask)


def associate_detections_to_tracks(
    track_states: np.ndarray,
    track_covariances: np.ndarray,
    detections: np.ndarray,
    *,
    measurement_matrix: np.ndarray | None = None,
    measurement_noise: np.ndarray | None = None,
    gating_threshold: float = 9.21,
) -> AssociationResult:
    """Associate 3D detections with predicted Kalman track states.

    Track states use the project's six-dimensional constant-velocity layout by
    default. The API accepts custom linear measurement matrices for reuse.
    """

    states = np.asarray(track_states, dtype=np.float64)
    observations = np.asarray(detections, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError(f"track_states must be 2D, got {states.shape}")
    if observations.ndim != 2:
        raise ValueError(f"detections must be 2D, got {observations.shape}")
    state_dimension = states.shape[1]
    measurement_dimension = observations.shape[1]
    matrix = (
        np.hstack((np.eye(3), np.zeros((3, state_dimension - 3))))
        if measurement_matrix is None
        else np.asarray(measurement_matrix, dtype=np.float64)
    )
    if matrix.ndim != 2 or matrix.shape != (measurement_dimension, state_dimension):
        raise ValueError(
            f"measurement_matrix must have shape ({measurement_dimension}, {state_dimension}), got {matrix.shape}"
        )
    covariances = np.asarray(track_covariances, dtype=np.float64)
    if covariances.ndim == 2:
        covariances = np.broadcast_to(covariances, (len(states), state_dimension, state_dimension)).copy()
    if covariances.shape != (len(states), state_dimension, state_dimension):
        raise ValueError(
            f"track_covariances must have shape ({state_dimension}, {state_dimension}) or "
            f"({len(states)}, {state_dimension}, {state_dimension}), got {covariances.shape}"
        )
    noise = np.zeros((measurement_dimension, measurement_dimension), dtype=np.float64)
    if measurement_noise is not None:
        noise = np.asarray(measurement_noise, dtype=np.float64)
    if noise.shape != (measurement_dimension, measurement_dimension):
        raise ValueError(f"measurement_noise must have shape ({measurement_dimension}, {measurement_dimension})")
    innovation_covariances = np.einsum("ij,njk,lk->nil", matrix, covariances, matrix) + noise
    predicted_measurements = states @ matrix.T
    return mahalanobis_association(
        predicted_measurements,
        observations,
        innovation_covariances,
        threshold=gating_threshold,
    )


def euclidean_cost_matrix(track_positions: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distances for the baseline association."""

    tracks = np.asarray(track_positions, dtype=np.float64)
    observations = np.asarray(detections, dtype=np.float64)
    if tracks.ndim != 2 or observations.ndim != 2 or tracks.shape[1] != observations.shape[1]:
        raise ValueError("track_positions and detections must be 2D arrays with matching dimensions")
    if not np.isfinite(tracks).all() or not np.isfinite(observations).all():
        raise ValueError("track_positions and detections must contain finite values")
    return np.linalg.norm(tracks[:, None, :] - observations[None, :, :], axis=2)


def euclidean_nearest_neighbor(
    track_positions: np.ndarray,
    detections: np.ndarray,
    *,
    max_distance: float | None = None,
) -> AssociationResult:
    """One-to-one Euclidean nearest-neighbor baseline using Hungarian costs."""

    costs = euclidean_cost_matrix(track_positions, detections)
    if max_distance is None:
        valid_mask = np.isfinite(costs)
    else:
        if not np.isfinite(max_distance) or max_distance < 0.0:
            raise ValueError("max_distance must be finite and non-negative")
        valid_mask = np.isfinite(costs) & (costs <= max_distance)
    return hungarian_association(costs, valid_mask=valid_mask)


associate = hungarian_association
associate_mahalanobis = mahalanobis_association
