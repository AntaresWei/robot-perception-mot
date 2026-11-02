#!/usr/bin/env python3
"""Prepare KITTI ``velodyne_reduced`` clouds in parallel.

This reproduces MMDetection3D's KITTI camera-FOV reduction without invoking
the legacy serial converter.  It is an offline data preparation utility and
does not alter the raw ``velodyne`` files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MMDET3D_ROOT = REPOSITORY_ROOT / "mmdetection3d-1.0"
if str(MMDET3D_ROOT) not in sys.path:
    sys.path.insert(0, str(MMDET3D_ROOT))


def _reduce_one(task: tuple[str, str, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]) -> int:
    source, destination, rect, transform, projection, image_shape = task
    from mmdet3d.core.bbox import box_np_ops

    points = np.fromfile(source, dtype=np.float32).reshape(-1, 4)
    reduced = box_np_ops.remove_outside_points(
        points,
        rect,
        transform,
        projection,
        image_shape,
    )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    reduced.astype(np.float32, copy=False).tofile(temporary_path)
    temporary_path.replace(destination_path)
    return int(len(reduced))


def _build_tasks(root: Path, info_path: Path) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]]:
    import mmcv

    infos: list[dict[str, Any]] = mmcv.load(str(info_path))
    tasks = []
    for info in infos:
        relative_path = Path(info["point_cloud"]["velodyne_path"])
        source = root / relative_path
        destination = root / relative_path.parent.parent / "velodyne_reduced" / relative_path.name
        shape = tuple(int(value) for value in info["image"]["image_shape"][:2])
        tasks.append((
            str(source),
            str(destination),
            np.asarray(info["calib"]["R0_rect"], dtype=np.float64),
            np.asarray(info["calib"]["Tr_velo_to_cam"], dtype=np.float64),
            np.asarray(info["calib"]["P2"], dtype=np.float64),
            shape,
        ))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT / "mmdetection3d-1.0" / "data" / "kitti")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    all_tasks: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]] = []
    for split in args.splits:
        info_path = root / f"kitti_infos_{'train' if split == 'train' else split}.pkl"
        if not info_path.is_file():
            raise FileNotFoundError(f"KITTI info file not found: {info_path}")
        split_tasks = _build_tasks(root, info_path)
        all_tasks.extend(split_tasks)
        print(f"{split}: {len(split_tasks)} frames")

    total_points = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, count in enumerate(executor.map(_reduce_one, all_tasks), start=1):
            total_points += count
            if index == 1 or index % 250 == 0 or index == len(all_tasks):
                print(f"processed {index}/{len(all_tasks)} frames; reduced points={total_points}")
    print(f"saved reduced clouds under {root}; frames={len(all_tasks)}; points={total_points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
