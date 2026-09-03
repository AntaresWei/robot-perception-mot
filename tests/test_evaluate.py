"""Focused tests for benchmark metric generation and baseline edge cases."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from robot_perception.common.types import Detection3D


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_script", ROOT / "scripts" / "evaluate.py")
assert SPEC is not None and SPEC.loader is not None
evaluate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate
SPEC.loader.exec_module(evaluate)


def _truth(track_id: int, position: tuple[float, float, float]) -> object:
    return evaluate.TruthObject(track_id, 1, np.asarray(position, dtype=np.float64))


def _observation(track_id: int, position: tuple[float, float, float]) -> object:
    return evaluate.TrackObservation(track_id, 1, np.asarray(position, dtype=np.float64))


def test_compute_metrics_perfect_sequence() -> None:
    truths = [[_truth(7, (0.0, 0.0, 0.0))], [_truth(7, (1.0, 0.0, 0.0))]]
    predictions = [[_observation(1, (0.0, 0.0, 0.0))], [_observation(1, (1.0, 0.0, 0.0))]]

    metrics = evaluate.compute_metrics(truths, predictions, distance_threshold=0.5)

    assert metrics["MOTA"] == 1.0
    assert metrics["IDF1"] == 1.0
    assert metrics["IDS"] == 0
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0


def test_compute_metrics_counts_switch_fp_fn() -> None:
    truths = [[_truth(7, (0.0, 0.0, 0.0))], [_truth(7, (1.0, 0.0, 0.0))]]
    predictions = [
        [_observation(1, (0.0, 0.0, 0.0)), _observation(99, (20.0, 0.0, 0.0))],
        [_observation(2, (1.0, 0.0, 0.0))],
    ]

    metrics = evaluate.compute_metrics(truths, predictions, distance_threshold=0.5)

    assert metrics["IDS"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 0


def test_nearest_neighbor_empty_input_is_safe() -> None:
    tracker = evaluate.NearestNeighborBaseline(max_age=2, association_distance=1.0)
    assert tracker.update([]) == []
    detection = Detection3D(np.zeros(3), np.ones(3), 0.0, 1.0, 1)
    assert len(tracker.update([detection])) == 1
