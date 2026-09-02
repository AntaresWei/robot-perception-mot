#!/usr/bin/env python3
"""Run the configured LiDAR preprocessing pipeline for one KITTI frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_perception.dataset import KittiTrackingDataset
from robot_perception.pointcloud import preprocess_point_cloud


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read configuration files") from error
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return content


def _resolve_path(path_value: str | Path) -> Path:
    # 配置文件允许相对仓库根目录的路径，便于换机器运行而不写绝对路径。
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "pipeline.yaml")
    parser.add_argument("--dataset-config", type=Path, help="Dataset YAML; overrides pipeline.dataset_config")
    parser.add_argument("--root", type=Path, help="KITTI root; overrides dataset.root")
    parser.add_argument("--split", choices=("training", "testing"), help="KITTI split; overrides dataset config")
    parser.add_argument("--sequence", required=True, help="KITTI sequence ID, for example 0000")
    parser.add_argument("--frame", required=True, type=int, help="KITTI frame ID, for example 100")
    args = parser.parse_args()

    try:
        pipeline_config = _load_yaml(args.config)
        dataset_config_path = args.dataset_config or _resolve_path(str(pipeline_config["dataset_config"]))
        dataset_config = _load_yaml(dataset_config_path).get("dataset")
        if not isinstance(dataset_config, dict):
            raise ValueError(f"missing dataset mapping in {dataset_config_path}")
        root_value = args.root or dataset_config.get("root")
        if root_value is None:
            raise ValueError("provide --root or set dataset.root in the dataset config")
        dataset = KittiTrackingDataset(
            _resolve_path(root_value),
            split=args.split or str(dataset_config.get("split", "training")),
            sequences=[args.sequence],
            load_images=False,
            load_annotations=False,
        )
        frame = dataset.get_frame(args.sequence, args.frame)
        pointcloud_config = pipeline_config.get("pointcloud")
        ground_config = pipeline_config.get("ground")
        if not isinstance(pointcloud_config, dict) or not isinstance(ground_config, dict):
            raise ValueError("pipeline config requires pointcloud and ground mappings")
        start = perf_counter()
        # 计时只覆盖算法流水线，不包含数据读取和 YAML 解析。
        result = preprocess_point_cloud(
            frame.lidar_points,
            pointcloud_config=pointcloud_config,
            ground_config=ground_config,
        )
        latency_ms = (perf_counter() - start) * 1000.0
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Sequence: {frame.sequence_id}")
    print(f"Frame: {frame.frame_id}")
    print(f"Raw point count: {len(result.raw_points)}")
    print(f"Filtered point count: {len(result.outlier_filtered_points)}")
    print(f"Ground point count: {len(result.ground_points)}")
    print(f"Non-ground point count: {len(result.non_ground_points)}")
    print(f"Processing latency: {latency_ms:.2f} ms")
    print(f"Stage shapes: {result.stage_shapes}")
    print(f"Plane model: {result.plane_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
