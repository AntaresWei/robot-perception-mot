#!/usr/bin/env python3
"""Visualize experimental object-level Camera-LiDAR fusion on one KITTI frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_perception.common.geometry import lidar_to_camera, lidar_to_image
from robot_perception.common.types import Detection3D
from robot_perception.dataset import KittiTrackingDataset
from robot_perception.dataset.calibration import parse_kitti_calibration
from robot_perception.detection import GTDetector
from robot_perception.fusion import (
    SyntheticCameraDetector,
    associate_camera_lidar_detections,
)
from robot_perception.pointcloud import preprocess_point_cloud


LIDAR_POINTS_MODES = frozenset({"raw", "non_ground", "none"})


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


def _select_lidar_points(
    mode: str,
    raw_points: np.ndarray,
    non_ground_points: np.ndarray,
) -> np.ndarray:
    """Return the configured visualization point cloud without mutating inputs."""

    if mode not in LIDAR_POINTS_MODES:
        raise ValueError(f"lidar_points_mode must be one of {sorted(LIDAR_POINTS_MODES)}, got {mode!r}")
    if mode == "raw":
        return raw_points
    if mode == "non_ground":
        return non_ground_points
    return np.empty((0, raw_points.shape[1]), dtype=raw_points.dtype)


def _project_points_with_depth(
    points: np.ndarray,
    calibration: object,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Project points and return rectified camera depths aligned with pixels."""

    pixels, projection_mask = lidar_to_image(
        points,
        calibration,
        image_shape=image_shape,
        return_mask=True,
    )
    if not len(pixels):
        return pixels, np.empty(0, dtype=np.float64)
    # lidar_to_image 的 mask 始终对应输入 points 的行；以同一 mask 取 rectified
    # camera z，保证每个 pixel 的颜色来自其自身的有效投影深度。
    camera_points, finite_mask = lidar_to_camera(points, calibration, return_mask=True)
    rectification = np.asarray(getattr(calibration, "R0_rect"), dtype=np.float64)
    rectified_depths = (camera_points @ rectification.T)[:, 2]
    depths_by_input = np.full(len(points), np.nan, dtype=np.float64)
    depths_by_input[finite_mask] = rectified_depths
    return pixels, depths_by_input[projection_mask]


def _depth_colours(depths: np.ndarray, lower_percentile: float, upper_percentile: float) -> np.ndarray:
    """Map rectified camera depth to OpenCV BGR colours using TURBO."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for fusion visualization") from error
    values = np.asarray(depths, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("projection depths must be a finite one-dimensional array")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("depth_percentiles must satisfy 0 <= low < high <= 100")
    if not len(values):
        return np.empty((0, 3), dtype=np.uint8)
    low, high = np.percentile(values, (lower_percentile, upper_percentile))
    if high <= low:
        normalized = np.full(len(values), 0.5, dtype=np.float64)
    else:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return cv2.applyColorMap(np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)


def _draw_depth_projection(
    image: np.ndarray,
    pixels: np.ndarray,
    depths: np.ndarray,
    *,
    max_points: int,
    point_radius: int,
    depth_percentiles: tuple[float, float],
) -> np.ndarray:
    """Draw a bounded subsample of projected points, coloured by camera depth."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for fusion visualization") from error
    if max_points < 1 or point_radius < 1:
        raise ValueError("max_projected_points and point_radius must be positive")
    if pixels.shape != (len(depths), 2):
        raise ValueError("pixels must have shape (N, 2) aligned with depths")
    output = image.copy()
    if not len(pixels):
        return output
    indices = np.linspace(0, len(pixels) - 1, min(len(pixels), max_points), dtype=np.int64)
    selected_pixels = pixels[indices]
    selected_colours = _depth_colours(depths[indices], *depth_percentiles)
    # geometry 已向量化处理投影；OpenCV circle API 需要逐点执行实际绘制。
    for (u, v), colour in zip(np.rint(selected_pixels).astype(np.int32), selected_colours):
        cv2.circle(output, (int(u), int(v)), point_radius, tuple(int(value) for value in colour), -1)
    return output


def _render_fusion_image(
    image: np.ndarray,
    pixels: np.ndarray,
    depths: np.ndarray,
    fusion_result: object,
    *,
    mode: str,
    visualization_settings: dict[str, object],
) -> np.ndarray:
    """Render one point-cloud mode plus the unchanged object-level fusion overlays."""

    depth_values = visualization_settings.get("depth_percentiles", (2.0, 98.0))
    if not isinstance(depth_values, (list, tuple)) or len(depth_values) != 2:
        raise ValueError("visualization.depth_percentiles must contain exactly two values")
    output = _draw_depth_projection(
        image,
        pixels,
        depths,
        max_points=int(visualization_settings.get("max_projected_points", 20000)),
        point_radius=int(visualization_settings.get("point_radius", 1)),
        depth_percentiles=(float(depth_values[0]), float(depth_values[1])),
    )
    import cv2

    cv2.putText(
        output,
        f"LiDAR points: {mode} (depth colour)",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for fused in fusion_result.fused_detections:
        _draw_box(output, fused.projected_bbox, (0, 255, 0), f"3D/ID {fused.track_id}")
        _draw_box(output, fused.camera_detection.bbox, (255, 0, 0), f"2D/ID {fused.track_id}")
    return output


def _output_paths(output: Path | None, sequence_id: str, frame_id: int, mode: str) -> dict[str, Path]:
    """Keep the configured main output and always create raw/non-ground comparisons."""

    output_dir = REPOSITORY_ROOT / "outputs" / "visualizations" if output is None else output.parent
    stem = f"fusion_{sequence_id}_{frame_id:06d}"
    primary = output or output_dir / f"{stem}_{mode}.png"
    return {
        "primary": primary,
        "raw": output_dir / f"{stem}_raw.png",
        "non_ground": output_dir / f"{stem}_non_ground.png",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--fusion-config", type=Path, default=REPOSITORY_ROOT / "configs" / "fusion.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "configs" / "pipeline.yaml")
    parser.add_argument("--root", type=Path, help="KITTI root; overrides dataset.root")
    parser.add_argument("--split", choices=("training", "testing"), help="KITTI split; overrides config")
    parser.add_argument("--sequence", required=True, help="KITTI sequence ID")
    parser.add_argument("--frame", required=True, type=int, help="KITTI frame ID")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        dataset_settings = _load_yaml(args.dataset_config).get("dataset")
        fusion_config = _load_yaml(args.fusion_config)
        fusion_settings = fusion_config.get("fusion")
        visualization_settings = fusion_config.get("visualization", {})
        pipeline_settings = _load_yaml(args.pipeline_config)
        pointcloud_settings = pipeline_settings.get("pointcloud")
        ground_settings = pipeline_settings.get("ground")
        if not isinstance(dataset_settings, dict) or not isinstance(fusion_settings, dict):
            raise ValueError("dataset and fusion mappings are required")
        if not isinstance(visualization_settings, dict):
            raise ValueError("visualization mapping is required when provided")
        if not isinstance(pointcloud_settings, dict) or not isinstance(ground_settings, dict):
            raise ValueError("pipeline config requires pointcloud and ground mappings")
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
        mode = str(visualization_settings.get("lidar_points_mode", "raw"))
        if mode not in LIDAR_POINTS_MODES:
            raise ValueError(f"lidar_points_mode must be one of {sorted(LIDAR_POINTS_MODES)}, got {mode!r}")
        # raw 保留 calibration debug 输入；non_ground 严格来自完整预处理流水线的最终输出。
        preprocess_result = preprocess_point_cloud(
            frame.lidar_points,
            pointcloud_config=pointcloud_settings,
            ground_config=ground_settings,
        )
        points_by_mode = {
            "raw": _select_lidar_points("raw", frame.lidar_points, preprocess_result.non_ground_points),
            "non_ground": _select_lidar_points("non_ground", frame.lidar_points, preprocess_result.non_ground_points),
            "none": _select_lidar_points("none", frame.lidar_points, preprocess_result.non_ground_points),
        }
        projections = {
            name: _project_points_with_depth(points, calibration, frame.image.shape)
            for name, points in points_by_mode.items()
        }
        paths = _output_paths(args.output, frame.sequence_id, frame.frame_id, mode)
        import cv2

        rendered: dict[str, np.ndarray] = {}
        for render_mode in dict.fromkeys(("raw", "non_ground", mode)):
            pixels, depths = projections[render_mode]
            rendered[render_mode] = _render_fusion_image(
                frame.image,
                pixels,
                depths,
                fusion_result,
                mode=render_mode,
                visualization_settings=visualization_settings,
            )
        for name, output in paths.items():
            source_mode = mode if name == "primary" else name
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), rendered[source_mode]):
                raise OSError(f"failed to write visualization: {output}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Saved fusion visualization ({mode}): {paths['primary']}")
    print(f"Saved raw projection: {paths['raw']}")
    print(f"Saved non-ground projection: {paths['non_ground']}")
    print(f"Projected raw LiDAR points: {len(projections['raw'][0])}")
    print(f"Projected non-ground LiDAR points: {len(projections['non_ground'][0])}")
    print(f"Fused objects: {len(fusion_result.fused_detections)}")
    print(f"Unmatched LiDAR: {fusion_result.unmatched_lidar.tolist()}")
    print(f"Unmatched camera: {fusion_result.unmatched_camera.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
