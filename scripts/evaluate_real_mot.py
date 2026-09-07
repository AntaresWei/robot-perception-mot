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
        idf1_denominator = 2 * idtp + total_fp + total_fn
        metrics = {
            "status": "OK",
            "detector": "MMDetection3D PointPillars",
            "sequence": str(args.sequence),
            "frame_range": {"start": args.start_frame, "end": args.end_frame},
            "frames": frame_count,
            "GT": total_gt,
            "matches": idtp,
            "FP": total_fp,
            "FN": total_fn,
            "IDS": total_ids,
            "MOTA": 1.0 - (total_fn + total_fp + total_ids) / max(total_gt, 1),
            "IDF1": 2 * idtp / idf1_denominator if idf1_denominator else 0.0,
            "runtime_seconds": elapsed,
            "FPS": frame_count / elapsed if elapsed else 0.0,
            "distance_threshold_m": args.distance_threshold,
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
