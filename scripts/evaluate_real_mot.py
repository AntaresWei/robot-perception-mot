#!/usr/bin/env python3
"""Evaluate real PointPillars -> Tracker MOT metrics on KITTI Tracking.

Run in the ``mmdet3d-kitti`` environment. KITTI labels are stored in rectified
camera coordinates; this script converts their bottom-center boxes to LiDAR
centers before matching against PointPillars detections and TrackState output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from robot_perception.common.types import Detection3D, FrameData  # noqa: E402
from robot_perception.dataset import KittiTrackingDataset  # noqa: E402
from robot_perception.dataset.calibration import parse_kitti_calibration  # noqa: E402
from robot_perception.detection import MMDet3DPointPillarsAdapter  # noqa: E402
from robot_perception.detection.base import KITTI_CLASS_IDS  # noqa: E402
from robot_perception.tracking import MultiObjectTracker  # noqa: E402
from robot_perception.tracking.association import hungarian_association  # noqa: E402


def _load_yaml(path: Path) -> dict[str, object]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "NOT AVAILABLE"


def _calibration_path(root: Path, split: str, sequence_id: str) -> Path:
    for candidate in (
        root / split / "calib" / f"{sequence_id}.txt",
        root / "data_tracking_calib" / split / "calib" / f"{sequence_id}.txt",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"calibration not found for sequence {sequence_id}")


def _camera_annotation_to_lidar(annotation: object, calibration: object) -> np.ndarray | None:
    class_id = KITTI_CLASS_IDS.get(annotation.class_name)
    if class_id is None or class_id > 3:
        return None
    # KITTI location is bottom-center in rectified camera coordinates. Convert
    # the 3D box center to unrectified camera, then solve Tr_velo_to_cam.
    camera_rect = np.asarray(annotation.location, dtype=np.float64).copy()
    camera_rect[1] -= float(annotation.dimensions[2]) / 2.0
    rect = np.asarray(calibration.R0_rect, dtype=np.float64)
    camera = np.linalg.solve(rect, camera_rect)
    transform = np.asarray(calibration.Tr_velo_to_cam, dtype=np.float64)
    rotation, translation = transform[:, :3], transform[:, 3]
    return np.linalg.solve(rotation, camera - translation)


def _truth_frame(frame: FrameData, calibration: object) -> list[tuple[int, int, np.ndarray]]:
    truth: list[tuple[int, int, np.ndarray]] = []
    for annotation in frame.annotations or []:
        position = _camera_annotation_to_lidar(annotation, calibration)
        class_id = KITTI_CLASS_IDS.get(annotation.class_name)
        if position is not None and class_id is not None and np.isfinite(position).all():
            truth.append((int(annotation.track_id), int(class_id), position))
    return truth


def _match_frame(
    truth: list[tuple[int, int, np.ndarray]],
    predictions: list[tuple[int, int, np.ndarray]],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not truth or not predictions:
        return np.empty((0, 2), dtype=np.int64), np.arange(len(truth)), np.arange(len(predictions))
    costs = np.full((len(truth), len(predictions)), np.inf, dtype=np.float64)
    for ti, (_, truth_class, truth_position) in enumerate(truth):
        for pi, (_, prediction_class, prediction_position) in enumerate(predictions):
            if truth_class == prediction_class:
                costs[ti, pi] = np.linalg.norm(truth_position - prediction_position)
    result = hungarian_association(costs, valid_mask=np.isfinite(costs) & (costs <= threshold))
    return result.matches, result.unmatched_tracks, result.unmatched_detections


def _standard_mot_metrics(
    truth_frames: list[list[tuple[int, int, np.ndarray]]],
    prediction_frames: list[list[tuple[int, int, np.ndarray]]],
    *,
    distance_threshold: float,
) -> dict[str, float | int | str | None]:
    """Compute global identity metrics with the MOTChallenge accumulator.

    KITTI identities are carried by ``label_02`` track IDs.  The accumulator
    performs frame-level distance matching and then computes the global ID
    mapping used by standard IDF1/IDP/IDR, unlike the legacy frame-wise
    ``2 * matches / (2 * matches + FP + FN)`` approximation.
    """

    if len(truth_frames) != len(prediction_frames):
        raise ValueError("truth and prediction frame counts must match")
    if not np.isfinite(distance_threshold) or distance_threshold < 0.0:
        raise ValueError("distance_threshold must be finite and non-negative")
    try:
        import motmetrics as mm
    except ImportError:  # pragma: no cover - exercised in the legacy detector environment
        return _standard_mot_metrics_fallback(
            truth_frames,
            prediction_frames,
            distance_threshold=distance_threshold,
        )

    accumulator = mm.MOTAccumulator(auto_id=False)
    for frame_index, (truth, predictions) in enumerate(zip(truth_frames, prediction_frames)):
        gt_ids = [int(item[0]) for item in truth]
        prediction_ids = [int(item[0]) for item in predictions]
        distances = np.full((len(truth), len(predictions)), np.nan, dtype=np.float64)
        for truth_index, (_, truth_class, truth_position) in enumerate(truth):
            for prediction_index, (_, prediction_class, prediction_position) in enumerate(predictions):
                # Class gating prevents a Car track from being assigned to a
                # Pedestrian/Cyclist hypothesis before MOT matching.
                if truth_class != prediction_class:
                    continue
                distance = float(np.linalg.norm(truth_position - prediction_position))
                if np.isfinite(distance) and distance <= distance_threshold:
                    distances[truth_index, prediction_index] = distance
        accumulator.update(gt_ids, prediction_ids, distances, frameid=frame_index)

    metric_names = (
        "num_frames",
        "num_objects",
        "num_matches",
        "num_misses",
        "num_false_positives",
        "num_switches",
        "num_fragmentations",
        "mota",
        "idf1",
        "idp",
        "idr",
        "precision",
        "recall",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
    )
    summary = mm.metrics.create().compute(accumulator, metrics=list(metric_names), name="sequence")
    row = summary.loc["sequence"]
    raw: dict[str, float | int | None] = {}
    for metric_name in metric_names:
        value = row[metric_name]
        if isinstance(value, (np.integer, int)):
            raw[metric_name] = int(value)
        elif isinstance(value, (np.floating, float)):
            raw[metric_name] = float(value) if np.isfinite(value) else None
        else:
            raw[metric_name] = value
    # 用项目已有的指标命名风格输出，同时保留 MOTAccumulator 的细分统计。
    return {
        "MOTA": raw["mota"],
        "IDF1": raw["idf1"],
        "IDP": raw["idp"],
        "IDR": raw["idr"],
        "IDS": raw["num_switches"],
        "FP": raw["num_false_positives"],
        "FN": raw["num_misses"],
        "GT": raw["num_objects"],
        "matches": raw["num_matches"],
        "distance_matches": int(raw["num_matches"] or 0) + int(raw["num_switches"] or 0),
        "num_frames": raw["num_frames"],
        "num_fragmentations": raw["num_fragmentations"],
        "precision": raw["precision"],
        "recall": raw["recall"],
        "mostly_tracked": raw["mostly_tracked"],
        "partially_tracked": raw["partially_tracked"],
        "mostly_lost": raw["mostly_lost"],
        "metric_backend": "motmetrics",
    }


def _standard_mot_metrics_fallback(
    truth_frames: list[list[tuple[int, int, np.ndarray]]],
    prediction_frames: list[list[tuple[int, int, np.ndarray]]],
    *,
    distance_threshold: float,
) -> dict[str, float | int | str | None]:
    """MOTChallenge-compatible fallback for the PointPillars legacy env.

    The detector environment intentionally remains unchanged and may not have
    the optional ``motmetrics`` package.  This fallback keeps the same global
    identity assignment definition for MOTA/IDF1/IDP/IDR using SciPy's
    Hungarian solver; when ``motmetrics`` is available it remains the primary
    implementation.
    """

    from scipy.optimize import linear_sum_assignment

    total_gt = total_fp = total_fn = 0
    pair_counts: dict[tuple[int, int], int] = {}
    gt_counts: dict[int, int] = {}
    matched_counts: dict[int, int] = {}
    previous_ids: dict[int, int] = {}
    identity_switches = 0
    distance_matches = 0
    for truth, predictions in zip(truth_frames, prediction_frames):
        for gt_id, _, _ in truth:
            gt_counts[int(gt_id)] = gt_counts.get(int(gt_id), 0) + 1
        matches, unmatched_truth, unmatched_predictions = _match_frame(
            truth, predictions, distance_threshold
        )
        total_gt += len(truth)
        total_fn += len(unmatched_truth)
        total_fp += len(unmatched_predictions)
        for truth_index, prediction_index in matches:
            gt_id = int(truth[int(truth_index)][0])
            prediction_id = int(predictions[int(prediction_index)][0])
            pair_counts[(gt_id, prediction_id)] = pair_counts.get((gt_id, prediction_id), 0) + 1
            matched_counts[gt_id] = matched_counts.get(gt_id, 0) + 1
            previous = previous_ids.get(gt_id)
            if previous is not None and previous != prediction_id:
                identity_switches += 1
            previous_ids[gt_id] = prediction_id
            distance_matches += 1

    gt_ids = sorted(gt_counts)
    prediction_ids = sorted({prediction_id for _, prediction_id in pair_counts})
    if gt_ids and prediction_ids:
        counts = np.zeros((len(gt_ids), len(prediction_ids)), dtype=np.float64)
        gt_index = {track_id: index for index, track_id in enumerate(gt_ids)}
        prediction_index = {track_id: index for index, track_id in enumerate(prediction_ids)}
        for (gt_id, prediction_id), count in pair_counts.items():
            counts[gt_index[gt_id], prediction_index[prediction_id]] = count
        rows, columns = linear_sum_assignment(-counts)
        idtp = int(counts[rows, columns].sum())
    else:
        idtp = 0
    idfn = total_gt - idtp
    idfp = (sum(len(frame) for frame in prediction_frames)) - idtp
    idf1_denominator = 2 * idtp + idfp + idfn
    mostly_tracked = sum(
        matched_counts.get(track_id, 0) / max(count, 1) >= 0.8
        for track_id, count in gt_counts.items()
    )
    partially_tracked = sum(
        0.2 <= matched_counts.get(track_id, 0) / max(count, 1) < 0.8
        for track_id, count in gt_counts.items()
    )
    mostly_lost = sum(
        matched_counts.get(track_id, 0) / max(count, 1) < 0.2
        for track_id, count in gt_counts.items()
    )
    return {
        "MOTA": float(1.0 - (total_fn + total_fp + identity_switches) / max(total_gt, 1)),
        "IDF1": float(2.0 * idtp / idf1_denominator) if idf1_denominator else None,
        "IDP": float(idtp / (idtp + idfp)) if idtp + idfp else None,
        "IDR": float(idtp / (idtp + idfn)) if idtp + idfn else None,
        "IDS": int(identity_switches),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "GT": int(total_gt),
        # MOTAccumulator 将 SWITCH 事件从 num_matches 中分离；保持相同定义。
        "matches": int(max(distance_matches - identity_switches, 0)),
        "distance_matches": int(distance_matches),
        "num_frames": len(truth_frames),
        "num_fragmentations": None,
        "precision": float(distance_matches / (distance_matches + total_fp)) if distance_matches + total_fp else None,
        "recall": float(distance_matches / total_gt) if total_gt else None,
        "mostly_tracked": int(mostly_tracked),
        "partially_tracked": int(partially_tracked),
        "mostly_lost": int(mostly_lost),
        "metric_backend": "scipy_fallback",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "metrics" / "real_pointpillars_mot.json")
    args = parser.parse_args(argv)
    try:
        dataset_config = _load_yaml(REPO_ROOT / "configs" / "dataset.yaml").get("dataset")
        detector_config = _load_yaml(REPO_ROOT / "configs" / "detector.yaml").get("detector")
        tracker_config = _load_yaml(REPO_ROOT / "configs" / "tracker.yaml").get("tracker")
        if not isinstance(dataset_config, dict) or not isinstance(detector_config, dict) or not isinstance(tracker_config, dict):
            raise ValueError("dataset, detector, and tracker config sections are required")
        root = _resolve(str(dataset_config.get("root", "data")))
        split = str(dataset_config.get("split", "training"))
        dataset = KittiTrackingDataset(root, split=split, sequences=[args.sequence], load_images=False, load_annotations=True)
        calibration = parse_kitti_calibration(_calibration_path(root, split, args.sequence))
        real = detector_config.get("real")
        if not isinstance(real, dict):
            raise ValueError("detector.real is required")
        detector = MMDet3DPointPillarsAdapter(
            _resolve(str(real["config"])),
            _resolve(str(real["checkpoint"])),
            device=args.device,
            score_threshold=float(real.get("score_threshold", 0.3)),
            class_mapping={int(key): int(value) for key, value in dict(real.get("class_mapping", {})).items()},
        )
        tracker = MultiObjectTracker(
            min_hits=int(tracker_config.get("min_hits", 3)),
            max_age=int(tracker_config.get("max_age", 5)),
            gating_threshold=float(tracker_config.get("gating_threshold", 9.21)),
            initial_covariance=tracker_config.get("initial_covariance", 1.0),
            process_noise=tracker_config.get("process_noise", 0.01),
            measurement_noise=tracker_config.get("measurement_noise", 0.1),
        )
        total_gt = total_fp = total_fn = total_ids = idtp = 0
        previous_ids: dict[int, int] = {}
        frame_count = 0
        start_time = perf_counter()
        frame_keys = getattr(dataset, "_frame_keys", ())
        truth_frames: list[list[tuple[int, int, np.ndarray]]] = []
        prediction_frames: list[list[tuple[int, int, np.ndarray]]] = []
        processed_frame_ids: list[int] = []
        for index, key in enumerate(frame_keys):
            if key.frame_id < args.start_frame or (args.end_frame >= 0 and key.frame_id > args.end_frame):
                continue
            frame = dataset[index]
            detections = detector.predict(frame)
            tracker.update(detections)
            predictions = [
                (track.track_id, track.class_id, track.state[:3].copy())
                for track in tracker.tracks
                if track.status == "Confirmed" and track.time_since_update == 0
            ]
            truth = _truth_frame(frame, calibration)
            matches, unmatched_truth, unmatched_predictions = _match_frame(truth, predictions, args.distance_threshold)
            truth_frames.append(truth)
            prediction_frames.append(predictions)
            processed_frame_ids.append(int(key.frame_id))
            total_gt += len(truth)
            total_fn += len(unmatched_truth)
            total_fp += len(unmatched_predictions)
            idtp += len(matches)
            for truth_index, prediction_index in matches:
                gt_id = truth[int(truth_index)][0]
                track_id = predictions[int(prediction_index)][0]
                if gt_id in previous_ids and previous_ids[gt_id] != track_id:
                    total_ids += 1
                previous_ids[gt_id] = track_id
            frame_count += 1
            if frame_count % 10 == 0 or frame_count == 1:
                print(f"Processed {frame_count} frames; frame={key.frame_id:06d}; detector={len(detections)} tracks={len(predictions)}")
        elapsed = perf_counter() - start_time
        if not processed_frame_ids:
            raise ValueError("no frames selected")
        idf1_denominator = 2 * idtp + total_fp + total_fn
        standard = _standard_mot_metrics(
            truth_frames,
            prediction_frames,
            distance_threshold=args.distance_threshold,
        )
        metrics = {
            "status": "OK",
            "detector": "MMDetection3D PointPillars",
            "sequence": str(args.sequence),
            "frame_range": {"start": processed_frame_ids[0], "end": processed_frame_ids[-1]},
            "requested_frame_range": {"start": args.start_frame, "end": args.end_frame},
            "frames": frame_count,
            # 顶层指标采用标准 MOTChallenge 计算；旧的逐帧近似单独保留。
            **standard,
            "runtime_seconds": elapsed,
            "FPS": frame_count / elapsed if elapsed else 0.0,
            "distance_threshold_m": args.distance_threshold,
            "legacy_framewise": {
                "MOTA": 1.0 - (total_fn + total_fp + total_ids) / max(total_gt, 1),
                "IDF1": 2 * idtp / idf1_denominator if idf1_denominator else 0.0,
            },
            "metric_definition": "MOTChallenge global identity metrics with global ID assignment",
            "git_commit_hash": _git_commit_hash(),
            "config": {
                "dataset": str(REPO_ROOT / "configs" / "dataset.yaml"),
                "detector": str(REPO_ROOT / "configs" / "detector.yaml"),
                "tracker": str(REPO_ROOT / "configs" / "tracker.yaml"),
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
