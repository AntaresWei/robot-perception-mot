#!/usr/bin/env python3
"""Run a synthetic multi-object tracking demo and write baseline metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_perception.common.types import Detection3D
from robot_perception.tracking import MultiObjectTracker


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read tracker demo configuration") from error
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return content


def _match_positions(
    truth: np.ndarray,
    states: list[object],
    distance_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not len(truth) or not len(states):
        return [], list(range(len(truth))), list(range(len(states)))
    estimated = np.stack([state.state[:3] for state in states])
    # 仅用于 demo 指标的独立匹配，不复用 Tracker 内部关联结果，避免指标自证。
    distances = np.linalg.norm(truth[:, None, :] - estimated[None, :, :], axis=2)
    truth_indices, state_indices = linear_sum_assignment(distances)
    valid = distances[truth_indices, state_indices] <= distance_threshold
    matches = list(zip(truth_indices[valid].tolist(), state_indices[valid].tolist()))
    matched_truth = {index for index, _ in matches}
    matched_states = {index for _, index in matches}
    return matches, sorted(set(range(len(truth))) - matched_truth), sorted(set(range(len(states))) - matched_states)


def _plot_trajectories(
    truth_history: np.ndarray,
    estimate_history: dict[int, list[tuple[float, float]]],
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/robot-perception-mot-mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 6))
    for object_index in range(truth_history.shape[1]):
        axis.plot(
            truth_history[:, object_index, 0],
            truth_history[:, object_index, 1],
            linestyle="--",
            linewidth=1.5,
            label=f"GT {object_index + 1}",
        )
    for track_id, positions in sorted(estimate_history.items()):
        values = np.asarray(positions)
        axis.plot(values[:, 0], values[:, 1], linewidth=1.2, label=f"Track {track_id}")
    axis.set_title("Synthetic 3D MOT trajectory (XY projection)")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "tracker.yaml")
    parser.add_argument("--metrics-output", type=Path, default=REPOSITORY_ROOT / "outputs" / "metrics" / "tracker_baseline.json")
    parser.add_argument("--visualization-output", type=Path, default=REPOSITORY_ROOT / "outputs" / "visualizations" / "tracker_demo.png")
    args = parser.parse_args()

    try:
        config = _load_yaml(args.config)
        tracker_config = config.get("tracker")
        demo_config = config.get("demo")
        if not isinstance(tracker_config, dict) or not isinstance(demo_config, dict):
            raise ValueError("tracker config requires tracker and demo mappings")
        num_frames = int(demo_config["num_frames"])
        num_objects = int(demo_config["num_objects"])
        measurement_std = float(demo_config["measurement_std"])
        miss_probability = float(demo_config["miss_probability"])
        association_distance = float(demo_config["association_distance"])
        if num_frames < 2 or num_objects < 2 or measurement_std < 0.0 or not 0.0 <= miss_probability < 1.0:
            raise ValueError("invalid synthetic demo parameters")

        rng = np.random.default_rng(int(demo_config["random_seed"]))
        initial_positions = np.asarray(((-8.0, -2.0, 0.0), (0.0, 3.0, 0.0), (8.0, -2.0, 0.0)))[:num_objects]
        velocities = np.asarray(((0.30, 0.08, 0.0), (0.05, -0.12, 0.0), (-0.30, 0.08, 0.0)))[:num_objects]
        truth_history = np.asarray(
            [initial_positions + frame * velocities for frame in range(num_frames)],
            dtype=np.float64,
        )
        tracker = MultiObjectTracker(
            min_hits=int(tracker_config["min_hits"]),
            max_age=int(tracker_config["max_age"]),
            gating_threshold=float(tracker_config.get("gating_threshold", 9.21)),
            initial_covariance=float(tracker_config.get("initial_covariance", 1.0)),
            process_noise=float(tracker_config.get("process_noise", 0.01)),
            measurement_noise=float(tracker_config.get("measurement_noise", 0.1)),
        )
        last_track_for_truth: dict[int, int] = {}
        estimate_history: dict[int, list[tuple[float, float]]] = {}
        false_positives = false_negatives = id_switches = id_true_positives = 0
        total_truth = num_frames * num_objects

        for frame_index, truth in enumerate(truth_history):
            # 演示输入模拟真实 detector：高斯位置噪声 + 随机漏检。
            detections = [
                Detection3D(
                    position=position + rng.normal(0.0, measurement_std, size=3),
                    dimensions=np.ones(3),
                    yaw=0.0,
                    score=1.0,
                    class_id=1,
                )
                for position in truth
                if rng.random() >= miss_probability
            ]
            states = tracker.update(detections)
            for state in states:
                estimate_history.setdefault(state.track_id, []).append((float(state.state[0]), float(state.state[1])))
            matches, unmatched_truth, unmatched_states = _match_positions(truth, states, association_distance)
            # MOTA/IDF1/IDS 只统计已确认且本帧有观测更新的 TrackState。
            false_negatives += len(unmatched_truth)
            false_positives += len(unmatched_states)
            id_true_positives += len(matches)
            for truth_index, state_index in matches:
                track_id = states[state_index].track_id
                previous_track_id = last_track_for_truth.get(truth_index)
                if previous_track_id is not None and previous_track_id != track_id:
                    id_switches += 1
                last_track_for_truth[truth_index] = track_id

        mota = 1.0 - (false_negatives + false_positives + id_switches) / total_truth
        idf1_denominator = 2 * id_true_positives + false_positives + false_negatives
        idf1 = 0.0 if idf1_denominator == 0 else 2 * id_true_positives / idf1_denominator
        metrics = {
            "dataset": "synthetic_constant_velocity",
            "num_frames": num_frames,
            "num_objects": num_objects,
            "measurement_std": measurement_std,
            "miss_probability": miss_probability,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "id_true_positives": id_true_positives,
            "ids": id_switches,
            "MOTA": mota,
            "IDF1": idf1,
            "config": str(args.config),
        }
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        _plot_trajectories(truth_history, estimate_history, args.visualization_output)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {args.metrics_output}")
    print(f"Saved visualization: {args.visualization_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
