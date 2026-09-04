#!/usr/bin/env python3
"""Export the trained KITTI three-class YOLO26 checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "weights" / "yolo26s_kitti3class.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "weights" / "yolo26s_kitti3class.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args(argv)
    try:
        from ultralytics import YOLO

        if not args.model.is_file():
            raise FileNotFoundError(f"trained YOLO26 checkpoint not found: {args.model}")
        exported = Path(YOLO(str(args.model)).export(
            format="onnx", imgsz=args.imgsz, device="cpu", simplify=False, opset=12, dynamic=False, nms=False
        ))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != args.output.resolve():
            shutil.copy2(exported, args.output)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Saved YOLO26 ONNX model: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
