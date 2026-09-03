#!/usr/bin/env python3
"""Visualize experimental object-level Camera-LiDAR fusion on one KITTI frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_perception.common.geometry import lidar_to_image
from robot_perception.common.types import Detection3D
from robot_perception.dataset import KittiTrackingDataset
from robot_perception.dataset.calibration import parse_kitti_calibration
from robot_perception.detection import GTDetector
from robot_perception.fusion import (
    SyntheticCameraDetector,
    associate_camera_lidar_detections,
    project_detection_corners_to_image,
)


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read fusion configuration") from error
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return config


def _calibration_path(root: Path, split: str, sequence_id: str) -> Path:
    candidates = (
        root / split / "calib" / f"{sequence_id}.txt",
        root / "data_tracking_calib" / split / "calib" / f"{sequence_id}.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"calibration file not found for {sequence_id} under {root}")


def _draw_box(image: np.ndarray, bbox: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    import cv2

    x1, y1, x2, y2 = np.rint(bbox).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--fusion-config", type=Path, default=REPOSITORY_ROOT / "configs" / "fusion.yaml")
    parser.add_argument("--root", type=Path, help="KITTI root; overrides dataset.root")
    parser.add_argument("--split", choices=("training", "testing"), help="KITTI split; overrides config")
    parser.add_argument("--sequence", required=True, help="KITTI sequence ID")
    parser.add_argument("--frame", required=True, type=int, help="KITTI frame ID")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        dataset_settings = _load_yaml(args.dataset_config).get("dataset")
        fusion_settings = _load_yaml(args.fusion_config).get("fusion")
        if not isinstance(dataset_settings, dict) or not isinstance(fusion_settings, dict):
            raise ValueError("dataset and fusion mappings are required")
        root_value = args.root or dataset_settings.get("root")
        if root_value is None:
            raise ValueError("provide --root or set dataset.root")
        root = Path(root_value)
        if not root.is_absolute():
            root = REPOSITORY_ROOT / root
        split = args.split or str(dataset_settings.get("split", "training"))
        dataset = KittiTrackingDataset(root, split=split, sequences=[args.sequence], load_images=True, load_annotations=True)
        frame = dataset.get_frame(args.sequence, args.frame)
        if frame.image is None:
            raise ValueError("fusion visualization requires a camera image")
        calibration = parse_kitti_calibration(_calibration_path(root, split, frame.sequence_id))

        # GTDetector 这里只作为实验用 LiDAR 侧输入；KITTI 相机标注位置是 bottom-center。
        gt_detections = GTDetector(class_ids={1, 2, 3}).predict(frame)
        lidar_detections: list[Detection3D] = []
        for detection in gt_detections:
            centered_position = detection.position.copy()
            centered_position[1] -= detection.dimensions[2] / 2.0
            lidar_detections.append(
                Detection3D(centered_position, detection.dimensions, detection.yaw, detection.score, detection.class_id)
            )
        camera_detections = SyntheticCameraDetector(
            score_threshold=float(fusion_settings.get("camera_score_threshold", 0.0)),
            class_ids={1, 2, 3},
            bbox_noise_std=float(fusion_settings.get("camera_bbox_noise_std", 0.0)),
            random_seed=fusion_settings.get("random_seed"),
        ).predict(frame)
        fusion_result = associate_camera_lidar_detections(
            lidar_detections,
            camera_detections,
            calibration,
            image_shape=frame.image.shape,
            iou_threshold=float(fusion_settings["iou_threshold"]),
            center_distance_threshold=float(fusion_settings["center_distance_threshold"]),
            coordinate_frame="camera",
        )
        # 点云投影只展示几何结果，与后面的跨模态匹配是两个独立阶段。
        pixels, _ = lidar_to_image(frame.lidar_points, calibration, image_shape=frame.image.shape, return_mask=True)
        image = frame.image.copy()
        import cv2

        for u, v in pixels[:: max(1, len(pixels) // 20000)].astype(np.int32):
            cv2.circle(image, (int(u), int(v)), 1, (0, 255, 0), -1)
        for fused in fusion_result.fused_detections:
            _draw_box(image, fused.projected_bbox, (0, 255, 0), f"3D/ID {fused.track_id}")
            _draw_box(image, fused.camera_detection.bbox, (255, 0, 0), f"2D/ID {fused.track_id}")
        output = args.output or REPOSITORY_ROOT / "outputs" / "visualizations" / f"fusion_{frame.sequence_id}_{frame.frame_id:06d}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), image):
            raise OSError(f"failed to write visualization: {output}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Saved fusion visualization: {output}")
    print(f"Projected LiDAR points: {len(pixels)}")
    print(f"Fused objects: {len(fusion_result.fused_detections)}")
    print(f"Unmatched LiDAR: {fusion_result.unmatched_lidar.tolist()}")
    print(f"Unmatched camera: {fusion_result.unmatched_camera.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
