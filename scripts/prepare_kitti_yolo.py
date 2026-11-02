#!/usr/bin/env python3
"""Convert KITTI Tracking image_02/label_02 to YOLO detection labels.

The converter keeps only the project's three classes. KITTI ``Van`` and
``Truck`` follow the existing project mapping to ``Car``; ``DontCare`` and
unknown classes are ignored. Splitting by sequence avoids frame leakage.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_IDS = {"Car": 0, "Van": 0, "Truck": 0, "Pedestrian": 1, "Person_sitting": 1, "Cyclist": 2}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _paths(root: Path, split: str) -> tuple[Path, Path]:
    for tracking_root in (root / "track", root):
        image_root = tracking_root / "data_tracking_image_2" / split / "image_02"
        label_root = tracking_root / "data_tracking_label_2" / split / "label_02"
        if image_root.is_dir() and label_root.is_dir():
            return image_root, label_root
    standard = root / split
    image_root, label_root = standard / "image_02", standard / "label_02"
    if image_root.is_dir() and label_root.is_dir():
        return image_root, label_root
    raise FileNotFoundError(f"KITTI image_02/label_02 not found under {root}")


def _sequence_split(sequences: list[str]) -> tuple[list[str], list[str]]:
    if len(sequences) < 2:
        return sequences, sequences
    val_count = max(1, round(len(sequences) * 0.2))
    return sequences[:-val_count], sequences[-val_count:]


def convert(
    root: Path,
    output: Path,
    split: str = "training",
    sequences: list[str] | None = None,
    val_sequences: list[str] | None = None,
) -> dict[str, int]:
    image_root, label_root = _paths(root, split)
    available_sequences = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    selected_sequences = available_sequences if sequences is None else [f"{int(value):04d}" for value in sequences]
    missing = sorted(set(selected_sequences) - set(available_sequences))
    if missing:
        raise FileNotFoundError(f"requested KITTI sequences are unavailable: {', '.join(missing)}")
    if val_sequences is None:
        train_sequences, selected_val_sequences = _sequence_split(selected_sequences)
    else:
        selected_val_sequences = [f"{int(value):04d}" for value in val_sequences]
        missing_val = sorted(set(selected_val_sequences) - set(selected_sequences))
        if missing_val:
            raise ValueError(f"validation sequences must be in selected sequences: {', '.join(missing_val)}")
        train_sequences = [sequence for sequence in selected_sequences if sequence not in selected_val_sequences]
    if not selected_sequences:
        raise ValueError("no KITTI sequences found")
    for directory in (output / "images" / "train", output / "images" / "val", output / "labels" / "train", output / "labels" / "val"):
        directory.mkdir(parents=True, exist_ok=True)
    counts = {"images": 0, "labels": 0, "boxes": 0, "train_sequences": len(train_sequences), "val_sequences": len(selected_val_sequences)}
    for sequence_id in selected_sequences:
        subset = "train" if sequence_id in train_sequences else "val"
        label_path = label_root / f"{sequence_id}.txt"
        annotations_by_frame: dict[int, list[str]] = {}
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 10:
                    continue
                class_id = CLASS_IDS.get(fields[2])
                if class_id is None:
                    continue
                frame_id = int(fields[0])
                x1, y1, x2, y2 = map(float, fields[6:10])
                annotations_by_frame.setdefault(frame_id, []).append((class_id, x1, y1, x2, y2))
        image_paths = sorted((image_root / sequence_id).glob("*.png"))
        if not image_paths:
            continue
        # KITTI Tracking 的同一 sequence 使用固定相机分辨率；每个 sequence 只读一次
        # 尺寸，避免为数千帧重复解码 PNG，标签坐标仍按真实图像尺寸归一化。
        with Image.open(image_paths[0]) as image:
            width, height = image.size
        for image_path in image_paths:
            frame_id = int(image_path.stem)
            destination_stem = f"{sequence_id}_{frame_id:06d}"
            destination_image = output / "images" / subset / f"{destination_stem}.png"
            # 训练器可以读取符号链接；避免为 8,008 帧重复占用数 GB 磁盘空间。
            if destination_image.exists() or destination_image.is_symlink():
                destination_image.unlink()
            os.symlink(image_path.resolve(), destination_image)
            counts["images"] += 1
            lines: list[str] = []
            for class_id, x1, y1, x2, y2 in annotations_by_frame.get(frame_id, []):
                left, right = max(0.0, min(x1, width)), max(0.0, min(x2, width))
                top, bottom = max(0.0, min(y1, height)), max(0.0, min(y2, height))
                if right <= left or bottom <= top:
                    continue
                cx, cy = ((left + right) / 2 / width, (top + bottom) / 2 / height)
                bw, bh = ((right - left) / width, (bottom - top) / height)
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (output / "labels" / subset / f"{destination_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts["labels"] += 1
            counts["boxes"] += len(lines)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "yolo26_kitti")
    parser.add_argument("--sequences", nargs="+", help="Optional KITTI sequence IDs, for example 0000 0001")
    parser.add_argument("--val-sequences", nargs="+", help="Validation sequence IDs kept out of training")
    args = parser.parse_args(argv)
    try:
        print(convert(_resolve(args.root), _resolve(args.output), sequences=args.sequences, val_sequences=args.val_sequences))
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
