#!/usr/bin/env python3
"""Project one KITTI LiDAR frame onto its camera image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_perception.common.geometry import lidar_to_image
from robot_perception.dataset import KittiTrackingDataset
from robot_perception.dataset.calibration import parse_kitti_calibration


def _load_config(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read the dataset configuration") from error
    if not path.is_file():
        raise FileNotFoundError(f"dataset config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    dataset_config = config.get("dataset")
    if not isinstance(dataset_config, dict):
        raise ValueError(f"missing dataset mapping in config: {path}")
    return dataset_config


def _calibration_path(root: Path, split: str, sequence_id: str) -> Path:
    # 数据集 Loader 支持两种目录布局，这里对 calibration 使用同样的查找规则。
    candidates = (
        root / split / "calib" / f"{sequence_id}.txt",
        root / "track" / "data_tracking_calib" / split / "calib" / f"{sequence_id}.txt",
        root / "data_tracking_calib" / split / "calib" / f"{sequence_id}.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"calibration file not found for sequence {sequence_id} under {root}")


def _draw_projection(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for projection visualization") from error

    # 坐标变换已在 geometry 中向量化；OpenCV 的 circle API 只能逐点绘制。
    output = image.copy()
    for u, v in pixels.astype(np.int32):
        cv2.circle(output, (int(u), int(v)), 1, (0, 255, 0), -1)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--root", type=Path, help="KITTI root; overrides dataset.root")
    parser.add_argument("--split", choices=("training", "testing"), help="KITTI split; overrides config")
    parser.add_argument("--sequence", required=True, help="KITTI sequence ID, for example 0000")
    parser.add_argument("--frame", required=True, type=int, help="KITTI frame ID, for example 100")
    parser.add_argument("--output", type=Path, help="Output image path")
    args = parser.parse_args()

    try:
        dataset_config = _load_config(args.config)
        root_value = args.root or dataset_config.get("root")
        if root_value is None:
            raise ValueError("provide --root or set dataset.root in the config")
        root = Path(root_value)
        if not root.is_absolute():
            root = REPOSITORY_ROOT / root
        split = args.split or str(dataset_config.get("split", "training"))
        dataset = KittiTrackingDataset(
            root,
            split=split,
            sequences=[args.sequence],
            load_images=True,
            load_annotations=False,
        )
        frame = dataset.get_frame(args.sequence, args.frame)
        calibration = parse_kitti_calibration(_calibration_path(root, split, frame.sequence_id))
        if frame.image is None:
            raise ValueError("the projection demo requires an image")
        pixels, _ = lidar_to_image(
            frame.lidar_points,
            calibration,
            image_shape=frame.image.shape,
            return_mask=True,
        )
        output_image = _draw_projection(frame.image, pixels)
        output_path = args.output or (
            REPOSITORY_ROOT / "outputs" / "visualizations" / f"kitti_{frame.sequence_id}_{frame.frame_id:06d}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        if not cv2.imwrite(str(output_path), output_image):
            raise OSError(f"failed to write visualization: {output_path}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Saved projection visualization: {output_path}")
    print(f"Projected points: {len(pixels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
