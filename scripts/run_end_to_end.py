#!/usr/bin/env python3
"""Run the real KITTI Camera-LiDAR detection, tracking, and fusion pipeline.

This script is intentionally executable in the ``mmdet3d-kitti`` environment:
PointPillars and the project core run in one process, while YOLO26 is consumed
through its ONNX graph with OpenCV DNN. No detector-specific code is imported
by the tracker itself.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from robot_perception.fusion import (  # noqa: E402
    Camera2DDetection,
    FusedDetection,
    Yolo26OnnxCameraDetector,
    associate_camera_lidar_detections,
    project_detection_to_image,
)
from robot_perception.tracking import MultiObjectTracker  # noqa: E402


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required by run_end_to_end.py") from error
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _calibration_path(root: Path, split: str, sequence_id: str) -> Path:
    candidates = (
        root / split / "calib" / f"{sequence_id}.txt",
        root / "data_tracking_calib" / split / "calib" / f"{sequence_id}.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"calibration file not found for {sequence_id} under {root}")


def _track_detections(tracker: MultiObjectTracker) -> list[Detection3D]:
    """Expose confirmed Track estimates as tracker-owned Detection3D objects."""

    detections: list[Detection3D] = []
    for track in tracker.tracks:
        if track.status != "Confirmed" or track.time_since_update != 0:
            continue
        detections.append(
            Detection3D(
                position=track.state[:3].copy(),
                dimensions=track.dimensions.copy(),
                yaw=track.yaw,
                score=track.score,
                class_id=track.class_id,
            )
        )
    return detections


def _attach_track_ids(
    fusion_result: object,
    track_detections: list[Detection3D],
    tracker: MultiObjectTracker,
) -> tuple[FusedDetection, ...]:
    """Restore Track IDs after fusion association without changing fusion core code."""

    confirmed_tracks = [
        track for track in tracker.tracks if track.status == "Confirmed" and track.time_since_update == 0
    ]
    fused: list[FusedDetection] = []
    for item in fusion_result.fused_detections:
        if not track_detections or not confirmed_tracks:
            fused.append(item)
            continue
        distances = [float(np.linalg.norm(track.state[:3] - item.lidar_detection.position)) for track in confirmed_tracks]
        track_id = confirmed_tracks[int(np.argmin(distances))].track_id
        camera_detection = Camera2DDetection(
            bbox=item.camera_detection.bbox,
            score=item.camera_detection.score,
            class_id=item.camera_detection.class_id,
            track_id=track_id,
        )
        fused.append(
            FusedDetection(
                lidar_detection=item.lidar_detection,
                camera_detection=camera_detection,
                projected_bbox=item.projected_bbox,
                fusion_confidence=item.fusion_confidence,
                track_id=track_id,
            )
        )
    return tuple(fused)


def _draw_frame(
    frame: FrameData,
    calibration: object,
    lidar_detections: list[Detection3D],
    camera_detections: list[Camera2DDetection],
    track_detections: list[Detection3D],
    fused_detections: tuple[FusedDetection, ...],
) -> np.ndarray:
    """Draw real YOLO26 boxes, projected PointPillars boxes, tracks, and fusion."""

    import cv2

    image = frame.image.copy()
    # Camera detector boxes are yellow; all YOLO classes are already restricted
    # to the KITTI three-class mapping by Yolo26OnnxCameraDetector.
    for detection in camera_detections:
        x1, y1, x2, y2 = np.rint(detection.bbox).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(image, f"YOLO {detection.class_id} {detection.score:.2f}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    # LiDAR detections are projected independently; this is geometric projection,
    # before camera association or sensor fusion.
    for detection in lidar_detections:
        bbox = project_detection_to_image(detection, calibration, image_shape=image.shape, coordinate_frame="lidar")
        if bbox is None:
            continue
        x1, y1, x2, y2 = np.rint(bbox).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(image, f"PP {detection.class_id} {detection.score:.2f}", (x1, min(image.shape[0] - 2, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
    # Fused boxes use green and carry the actual tracker ID restored above.
    for item in fused_detections:
        x1, y1, x2, y2 = np.rint(item.projected_bbox).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(image, f"Fused ID {item.track_id} c={item.fusion_confidence:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)
    # Track-only projected boxes are blue, which makes unmatched tracks visible.
    for track in track_detections:
        bbox = project_detection_to_image(track, calibration, image_shape=image.shape, coordinate_frame="lidar")
        if bbox is None:
            continue
        x1, y1, x2, y2 = np.rint(bbox).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 128, 0), 1)
    cv2.putText(image, f"E2E: PP3D={len(lidar_detections)} YOLO26_2D={len(camera_detections)} tracks={len(track_detections)} fused={len(fused_detections)}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=REPO_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--detector-config", type=Path, default=REPO_ROOT / "configs" / "detector.yaml")
    parser.add_argument("--tracker-config", type=Path, default=REPO_ROOT / "configs" / "tracker.yaml")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", help="PointPillars device, e.g. cuda:0")
    parser.add_argument("--min-hits", type=int, help="Override tracker confirmation hits for replay smoke/experiments")
    parser.add_argument("--camera-model", type=Path, help="Override detector.camera.model (use hold-out ONNX for honest evaluation)")
    args = parser.parse_args(argv)
    writer = None
    writer_size: tuple[int, int] | None = None
    try:
        dataset_settings = _load_yaml(args.dataset_config).get("dataset")
        detector_settings = _load_yaml(args.detector_config).get("detector")
        tracker_settings = _load_yaml(args.tracker_config).get("tracker")
        if not isinstance(dataset_settings, dict) or not isinstance(detector_settings, dict) or not isinstance(tracker_settings, dict):
            raise ValueError("dataset, detector, and tracker mappings are required")
        root = _resolve(args.root or dataset_settings.get("root", "data"))
        split = str(dataset_settings.get("split", "training"))
        dataset = KittiTrackingDataset(root, split=split, sequences=[args.sequence], load_images=True, load_annotations=True)
        calibration = parse_kitti_calibration(_calibration_path(root, split, dataset.sequence_ids[0]))
        real = detector_settings.get("real")
        camera = detector_settings.get("camera")
        if not isinstance(real, dict) or not isinstance(camera, dict):
            raise ValueError("detector.yaml requires detector.real and detector.camera mappings")
        pp = MMDet3DPointPillarsAdapter(
            _resolve(str(real["config"])),
            _resolve(str(real["checkpoint"])),
            device=args.device or str(real.get("device", "cuda:0")),
            score_threshold=float(real.get("score_threshold", 0.3)),
            class_mapping={int(key): int(value) for key, value in dict(real.get("class_mapping", {})).items()},
        )
        yolo = Yolo26OnnxCameraDetector(
            _resolve(args.camera_model or str(camera["model"])),
            score_threshold=float(camera.get("score_threshold", 0.25)),
            iou_threshold=float(camera.get("iou_threshold", 0.45)),
            input_size=int(camera.get("input_size", 640)),
            class_mapping={int(key): int(value) for key, value in dict(camera.get("class_mapping", {0: 1, 1: 2, 2: 3})).items()},
        )
        tracker = MultiObjectTracker(
            min_hits=int(args.min_hits if args.min_hits is not None else tracker_settings.get("min_hits", 3)),
            max_age=int(tracker_settings.get("max_age", 5)),
            gating_threshold=float(tracker_settings.get("gating_threshold", 9.21)),
            initial_covariance=tracker_settings.get("initial_covariance", 1.0),
            process_noise=tracker_settings.get("process_noise", 0.01),
            measurement_noise=tracker_settings.get("measurement_noise", 0.1),
        )
        fusion_config = _load_yaml(REPO_ROOT / "configs" / "fusion.yaml")
        fusion_settings = fusion_config.get("fusion", {})
        if not isinstance(fusion_settings, dict):
            raise ValueError("fusion config must contain a fusion mapping")
        output_video = args.output_video or REPO_ROOT / "outputs" / "videos" / f"end_to_end_{args.sequence}.mp4"
        output_json = args.output_json or REPO_ROOT / "outputs" / "metrics" / f"end_to_end_{args.sequence}.json"
        output_video = _resolve(output_video)
        output_json = _resolve(output_json)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        total = Counter()
        frame_count = 0
        processed_frame_ids: list[int] = []
        timings: list[float] = []
        start_time = perf_counter()
        indexed_frame_ids = [key.frame_id for key in getattr(dataset, "_frame_keys", ())]
        for index, frame_id in enumerate(indexed_frame_ids):
            if frame_id < args.start_frame or (args.end_frame >= 0 and frame_id > args.end_frame):
                continue
            frame = dataset[index]
            frame_start = perf_counter()
            lidar_detections = pp.predict(frame)
            camera_detections = yolo.predict(frame)
            tracker.update(lidar_detections)
            track_detections = _track_detections(tracker)
            fusion_result = associate_camera_lidar_detections(
                track_detections,
                camera_detections,
                calibration,
                image_shape=frame.image.shape,
                iou_threshold=float(fusion_settings.get("iou_threshold", 0.1)),
                center_distance_threshold=float(fusion_settings.get("center_distance_threshold", 80.0)),
                coordinate_frame="lidar",
            )
            fused_detections = _attach_track_ids(fusion_result, track_detections, tracker)
            image = _draw_frame(frame, calibration, lidar_detections, camera_detections, track_detections, fused_detections)
            height, width = image.shape[:2]
            if height % 2 or width % 2:
                image = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            video_size = image.shape[1], image.shape[0]
            if writer is None:
                # mp4v requires even dimensions; pad only the video frame, never
                # the geometric image coordinates used by projection/fusion.
                writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, video_size)
                if not writer.isOpened():
                    raise OSError(f"failed to open video writer: {output_video}")
                writer_size = video_size
            elif writer_size != video_size:
                raise RuntimeError("video frame dimensions changed")
            writer.write(image)
            elapsed = perf_counter() - frame_start
            timings.append(elapsed)
            frame_count += 1
            processed_frame_ids.append(frame_id)
            total.update({"lidar_detections": len(lidar_detections), "camera_detections": len(camera_detections), "tracks": len(track_detections), "fused": len(fused_detections)})
            if frame_count % 10 == 0 or frame_count == 1:
                print(f"Processed {frame_count} frames; KITTI frame={frame_id:06d}; PP={len(lidar_detections)} YOLO={len(camera_detections)} tracks={len(track_detections)} fused={len(fused_detections)}")
        if not frame_count:
            raise ValueError("no frames selected")
        wall_time = perf_counter() - start_time
        payload = {
            "status": "OK",
            "sequence": str(args.sequence),
            "frame_range": {"start": int(processed_frame_ids[0]), "end": int(processed_frame_ids[-1])},
            "frames_processed": frame_count,
            "detectors": {"lidar": "MMDetection3D PointPillars", "camera": "YOLO26 ONNX"},
            "models": {
                "lidar_config": str(_resolve(str(real["config"]))),
                "lidar_checkpoint": str(_resolve(str(real["checkpoint"]))),
                "camera_onnx": str(_resolve(args.camera_model or str(camera["model"]))),
            },
            "thresholds": {
                "lidar_score": float(real.get("score_threshold", 0.3)),
                "camera_score": float(camera.get("score_threshold", 0.25)),
                "camera_iou": float(camera.get("iou_threshold", 0.45)),
            },
            "class_mapping": {"Car": 1, "Pedestrian": 2, "Cyclist": 3},
            "totals": dict(total),
            "averages": {
                "lidar_detections_per_frame": total["lidar_detections"] / frame_count,
                "camera_detections_per_frame": total["camera_detections"] / frame_count,
                "tracks_per_frame": total["tracks"] / frame_count,
                "fused_per_frame": total["fused"] / frame_count,
            },
            "runtime": {
                "wall_seconds": wall_time,
                "end_to_end_fps": frame_count / wall_time,
                "latency_ms_per_frame": float(np.mean(timings) * 1000.0),
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": {"dataset": str(args.dataset_config), "detector": str(args.detector_config), "tracker": str(args.tracker_config)},
        }
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    finally:
        if writer is not None:
            writer.release()
    print(f"Saved end-to-end video: {output_video}")
    print(f"Saved end-to-end summary: {output_json}")
    print(f"Frames: {frame_count}; average fused/frame: {total['fused'] / frame_count:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
