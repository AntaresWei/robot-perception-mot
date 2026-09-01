#!/usr/bin/env python3
"""Inspect one KITTI Tracking frame using a YAML config or CLI dataset root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_perception.dataset import KittiTrackingDataset


def _load_dataset_config(path: Path) -> dict[str, object]:
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


def _format_annotation(annotation: object | None) -> str:
    if annotation is None:
        return "None"
    return str(annotation)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--root", type=Path, help="KITTI root; overrides dataset.root in the config")
    parser.add_argument("--split", choices=("training", "testing"), help="KITTI split; overrides config")
    parser.add_argument("--sequence", required=True, help="KITTI sequence ID, for example 0000")
    parser.add_argument("--frame", required=True, type=int, help="KITTI frame ID, for example 100")
    args = parser.parse_args()

    try:
        dataset_config = _load_dataset_config(args.config)
        root_value = args.root or dataset_config.get("root")
        if root_value is None:
            raise ValueError("provide --root or set dataset.root in the config")
        dataset_root = Path(root_value)
        if not dataset_root.is_absolute():
            dataset_root = REPOSITORY_ROOT / dataset_root
        split = args.split or str(dataset_config.get("split", "training"))
        dataset = KittiTrackingDataset(
            dataset_root,
            split=split,
            sequences=[args.sequence],
            load_images=bool(dataset_config.get("load_images", True)),
            load_annotations=bool(dataset_config.get("load_annotations", True)),
        )
        frame = dataset.get_frame(args.sequence, args.frame)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Sequence: {frame.sequence_id}")
    print(f"Frame: {frame.frame_id}")
    print(f"Image shape: {None if frame.image is None else frame.image.shape}")
    print(f"LiDAR shape: {frame.lidar_points.shape}")
    print(f"Number of annotations: {0 if frame.annotations is None else len(frame.annotations)}")
    print(f"First annotation: {_format_annotation(frame.annotations[0] if frame.annotations else None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
