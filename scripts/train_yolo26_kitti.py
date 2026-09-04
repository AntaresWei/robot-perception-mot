#!/usr/bin/env python3
"""Fine-tune YOLO26s on KITTI Tracking Car/Pedestrian/Cyclist labels.

Run this script in the temporary environment created from the main project:
``/tmp/robot-perception-mot-yolo26/bin/python scripts/train_yolo26_kitti.py``.
It writes the selected best checkpoint under ``weights/`` without modifying the
``robot-perception-mot`` conda environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("YOLO26 training config must be a mapping")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "yolo26_train.yaml")
    parser.add_argument("--epochs", type=int, help="Override configured epochs")
    parser.add_argument("--device", help="Override configured CUDA device")
    args = parser.parse_args(argv)
    try:
        from ultralytics import YOLO

        config = _load(args.config)
        model_path = _resolve(config["model"])
        data_path = _resolve(config["data"])
        if not model_path.is_file() or not data_path.is_file():
            raise FileNotFoundError("YOLO26 model or data configuration is missing")
        epochs = int(args.epochs if args.epochs is not None else config["epochs"])
        if epochs < 1:
            raise ValueError("epochs must be positive")
        model = YOLO(str(model_path))
        model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=int(config["imgsz"]),
            batch=int(config["batch"]),
            device=str(args.device if args.device is not None else config["device"]),
            workers=int(config["workers"]),
            patience=int(config["patience"]),
            seed=int(config["seed"]),
            project=str(_resolve(config["project"])),
            name=str(config["name"]),
            exist_ok=True,
            pretrained=True,
            plots=True,
        )
        best_weight = Path(model.trainer.save_dir) / "weights" / "best.pt"
        target = _resolve(config["output_weight"])
        if not best_weight.is_file():
            raise FileNotFoundError(f"YOLO26 training did not produce {best_weight}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weight, target)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Saved KITTI three-class YOLO26 checkpoint: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
