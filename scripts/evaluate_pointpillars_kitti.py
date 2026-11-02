#!/usr/bin/env python3
"""Evaluate saved MMDetection3D KITTI predictions without Numba NVVM JIT.

The local legacy MMDetection3D environment runs inference on GPU correctly,
but its Numba 0.53 KITTI evaluator cannot JIT against the host CUDA 12 NVVM.
This script preserves the official KITTI metric code and replaces only its
rotated BEV IoU primitive with an exact Shapely CPU implementation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MMDET3D_ROOT = REPOSITORY_ROOT / "mmdetection3d-1.0"
if str(MMDET3D_ROOT) not in sys.path:
    sys.path.insert(0, str(MMDET3D_ROOT))


def _rotated_polygon(box: np.ndarray) -> Any:
    """Build one camera-BEV rectangle from ``[x, z, length, width, yaw]``."""

    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon

    center_x, center_z, length, width, yaw = (float(value) for value in box)
    if not np.isfinite(box).all() or length <= 0.0 or width <= 0.0:
        return Polygon()
    rectangle = Polygon((
        (-length / 2.0, -width / 2.0),
        (length / 2.0, -width / 2.0),
        (length / 2.0, width / 2.0),
        (-length / 2.0, width / 2.0),
    ))
    # KITTI camera yaw is clockwise in this evaluator's BEV convention,
    # whereas Shapely uses counter-clockwise positive angles.
    return translate(rotate(rectangle, -np.degrees(yaw), origin=(0, 0)), center_x, center_z)


def cpu_rotated_iou(
    boxes: np.ndarray,
    query_boxes: np.ndarray,
    criterion: int = -1,
    device_id: int = 0,
) -> np.ndarray:
    """Return the KITTI evaluator's rotated overlap matrix on CPU.

    ``criterion`` has the exact same meaning as MMDetection3D's original
    ``rotate_iou_gpu_eval``.  ``device_id`` is accepted for API compatibility.
    """

    del device_id
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    overlaps = np.zeros((len(boxes), len(query_boxes)), dtype=np.float32)
    polygons = [_rotated_polygon(box) for box in boxes]
    query_polygons = [_rotated_polygon(box) for box in query_boxes]
    for box_index, polygon in enumerate(polygons):
        if polygon.is_empty:
            continue
        box_area = float(polygon.area)
        for query_index, query_polygon in enumerate(query_polygons):
            if query_polygon.is_empty:
                continue
            intersection = float(polygon.intersection(query_polygon).area)
            if criterion == -1:
                denominator = box_area + float(query_polygon.area) - intersection
            elif criterion == 0:
                denominator = box_area
            elif criterion == 1:
                denominator = float(query_polygon.area)
            else:
                overlaps[box_index, query_index] = intersection
                continue
            overlaps[box_index, query_index] = intersection / denominator if denominator > 0.0 else 0.0
    return overlaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="MMDetection3D config path")
    parser.add_argument("--predictions", type=Path, required=True, help="Saved tools/test.py --out PKL")
    args = parser.parse_args()
    config_path = args.config.resolve()
    prediction_path = args.predictions.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"predictions not found: {prediction_path}")

    # Official configs express data_root relative to the MMDetection3D root.
    os.chdir(MMDET3D_ROOT)

    import mmcv
    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    # ``eval.py`` imports ``rotate_iou_gpu_eval`` lazily.  Registering this
    # minimal replacement before evaluation prevents importing rotate_iou.py,
    # whose old eager Numba CUDA signature is incompatible with CUDA 12 NVVM.
    module_name = "mmdet3d.core.evaluation.kitti_utils.rotate_iou"
    cpu_rotate_iou_module = types.ModuleType(module_name)
    cpu_rotate_iou_module.rotate_iou_gpu_eval = cpu_rotated_iou
    sys.modules[module_name] = cpu_rotate_iou_module

    config = Config.fromfile(str(config_path))

    # The project-local configs use paths relative to the repository root
    # (``mmdetection3d-1.0/data/kitti``), while MMDetection3D resolves them
    # after changing into its own root.  Normalize only this known prefix so
    # both invocation styles continue to work without hard-coded data paths.
    def _normalize_paths(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("mmdetection3d-1.0/data/kitti", "data/kitti")
        if isinstance(value, dict):
            for key in list(value):
                value[key] = _normalize_paths(value[key])
        elif isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = _normalize_paths(item)
        return value

    _normalize_paths(config._cfg_dict)
    dataset = build_dataset(config.data.test)
    predictions = mmcv.load(str(prediction_path))
    if len(predictions) != len(dataset):
        raise ValueError(
            f"prediction count ({len(predictions)}) does not match validation dataset ({len(dataset)})"
        )
    metrics = dataset.evaluate(predictions, metric="mAP")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
