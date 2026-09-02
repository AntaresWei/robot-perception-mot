#!/usr/bin/env python3
"""Run a detector adapter on one KITTI frame and print detection statistics."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_perception.dataset import KittiTrackingDataset
from robot_perception.detection import (
    BaseDetector,
    GTDetector,
    MMDet3DPointPillarsAdapter,
    NoisyGTDetector,
    class_name_for_id,
)


def _load_yaml(path: Path) -> dict[str, Any]:
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
    # 配置使用相对仓库路径，避免把本机数据目录写死到脚本中。
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _build_detector(config: dict[str, Any], detector_type: str | None) -> BaseDetector:
    settings = config.get("detector")
    if not isinstance(settings, dict):
        raise ValueError("detector config must contain a detector mapping")
    selected_type = detector_type or settings.get("type", "gt")
    common_kwargs = {
        "score_threshold": float(settings.get("score_threshold", 0.0)),
        "class_ids": settings.get("class_ids"),
        "default_score": float(settings.get("default_score", 1.0)),
    }
    if selected_type == "gt":
        return GTDetector(**common_kwargs)
    if selected_type == "noisy_gt":
        noise = settings.get("noise")
        if not isinstance(noise, dict):
            raise ValueError("noisy_gt requires detector.noise mapping")
        return NoisyGTDetector(
            **common_kwargs,
            position_std=noise.get("position_std", 0.0),
            dimensions_std=noise.get("dimensions_std", 0.0),
            yaw_std=float(noise.get("yaw_std", 0.0)),
            score_std=float(noise.get("score_std", 0.0)),
            random_seed=noise.get("random_seed"),
        )
    if selected_type in {"real", "mmdet3d_pointpillars"}:
        real = settings.get("real")
        if not isinstance(real, dict):
            raise ValueError("real detector requires detector.real mapping")
        config_path = real.get("config")
        checkpoint_path = real.get("checkpoint")
        if config_path is None or checkpoint_path is None:
            raise ValueError("detector.real.config and detector.real.checkpoint must be set")
        return MMDet3DPointPillarsAdapter(
            _resolve_path(config_path),
            _resolve_path(checkpoint_path),
            device=str(real.get("device", "cuda:0")),
            score_threshold=float(real.get("score_threshold", settings.get("score_threshold", 0.0))),
            class_mapping={int(key): int(value) for key, value in dict(real.get("class_mapping", {})).items()},
        )
    raise ValueError(f"unsupported detector type: {selected_type!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--detector-config", type=Path, default=REPOSITORY_ROOT / "configs" / "detector.yaml")
    parser.add_argument("--root", type=Path, help="KITTI root; overrides dataset.root")
    parser.add_argument("--split", choices=("training", "testing"), help="KITTI split; overrides dataset config")
    parser.add_argument("--detector", choices=("gt", "noisy_gt", "real"), help="Detector adapter; overrides config")
    parser.add_argument("--sequence", required=True, help="KITTI sequence ID, for example 0000")
    parser.add_argument("--frame", required=True, type=int, help="KITTI frame ID, for example 100")
    args = parser.parse_args()

    try:
        dataset_settings = _load_yaml(args.dataset_config).get("dataset")
        if not isinstance(dataset_settings, dict):
            raise ValueError("dataset config must contain a dataset mapping")
        root_value = args.root or dataset_settings.get("root")
        if root_value is None:
            raise ValueError("provide --root or set dataset.root in the dataset config")
        dataset = KittiTrackingDataset(
            _resolve_path(root_value),
            split=args.split or str(dataset_settings.get("split", "training")),
            sequences=[args.sequence],
            load_images=False,
            load_annotations=True,
        )
        frame = dataset.get_frame(args.sequence, args.frame)
        detector = _build_detector(_load_yaml(args.detector_config), args.detector)
        detections = detector.predict(frame)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    class_distribution = Counter(class_name_for_id(detection.class_id) for detection in detections)
    # 空 detection 时输出 None，避免对空数组求 min/max 产生误导性数值。
    scores = np.asarray([detection.score for detection in detections], dtype=np.float64)
    score_statistics = (
        {"min": float(scores.min()), "max": float(scores.max()), "mean": float(scores.mean())}
        if scores.size
        else {"min": None, "max": None, "mean": None}
    )
    print(f"Frame: {frame.sequence_id}/{frame.frame_id:06d}")
    print(f"Num detections: {len(detections)}")
    print(f"Class distribution: {dict(sorted(class_distribution.items()))}")
    print(f"Score statistics: {score_statistics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
