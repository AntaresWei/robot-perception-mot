#!/usr/bin/env python3
"""Render a KITTI sequence's final fusion visualization to an MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPOSITORY_ROOT / "scripts"
SRC_DIR = REPOSITORY_ROOT / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robot_perception.dataset import KittiTrackingDataset
from robot_perception.dataset.calibration import parse_kitti_calibration
from run_fusion_demo import _calibration_path, _load_yaml, render_fusion_modes


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _frame_is_selected(frame_id: int, start_frame: int, end_frame: int) -> bool:
    return frame_id >= start_frame and (end_frame < 0 or frame_id <= end_frame)


def _pad_to_even_dimensions(image):
    """Pad odd image dimensions so MP4 codecs do not silently crop a row/column."""

    import cv2

    height, width = image.shape[:2]
    return cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=REPOSITORY_ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--fusion-config", type=Path, default=REPOSITORY_ROOT / "configs" / "fusion.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPOSITORY_ROOT / "configs" / "pipeline.yaml")
    parser.add_argument("--root", type=Path, help="KITTI root; overrides dataset.root")
    parser.add_argument("--split", choices=("training", "testing"), help="KITTI split; overrides config")
    parser.add_argument("--sequence", required=True, help="KITTI sequence ID, for example 0000")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1, help="inclusive end frame; -1 selects the full sequence")
    parser.add_argument("--lidar-points-mode", choices=("raw", "non_ground", "none"))
    parser.add_argument("--fps", type=float, help="output video FPS; overrides visualization.video_fps")
    parser.add_argument("--output", type=Path, help="MP4 output path")
    args = parser.parse_args()

    writer = None
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
        root = _resolve_path(root_value)
        split = args.split or str(dataset_settings.get("split", "training"))
        dataset = KittiTrackingDataset(root, split=split, sequences=[args.sequence], load_images=True, load_annotations=True)
        calibration = parse_kitti_calibration(_calibration_path(root, split, dataset.sequence_ids[0]))
        mode = args.lidar_points_mode or str(visualization_settings.get("video_lidar_points_mode", "non_ground"))
        if mode not in {"raw", "non_ground", "none"}:
            raise ValueError("video_lidar_points_mode must be raw, non_ground, or none")
        fps = float(args.fps if args.fps is not None else visualization_settings.get("video_fps", 10.0))
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        output = args.output or REPOSITORY_ROOT / "outputs" / "videos" / f"fusion_{dataset.sequence_ids[0]}_{mode}.mp4"
        if not output.is_absolute():
            output = REPOSITORY_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)

        import cv2

        frame_count = 0
        total_fused = 0
        start = perf_counter()
        # Dataset 的 LiDAR 帧索引不需要解码图像/点云；先筛选可避免只导出
        # 末段视频时仍读取完整 sequence。
        frame_keys = getattr(dataset, "_frame_keys", ())
        indexed_frame_ids = [key.frame_id for key in frame_keys]
        if len(indexed_frame_ids) != len(dataset):
            indexed_frame_ids = [dataset[index].frame_id for index in range(len(dataset))]
        for index, frame_id in enumerate(indexed_frame_ids):
            if not _frame_is_selected(frame_id, args.start_frame, args.end_frame):
                continue
            frame = dataset[index]
            rendered, counts, fusion_result = render_fusion_modes(
                frame,
                calibration,
                fusion_settings=fusion_settings,
                pointcloud_settings=pointcloud_settings,
                ground_settings=ground_settings,
                visualization_settings=visualization_settings,
                modes=(mode,),
            )
            image = rendered[mode]
            cv2.putText(
                image,
                f"Sequence {frame.sequence_id} | Frame {frame.frame_id:06d} | projected points {counts[mode]}",
                (12, image.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            image = _pad_to_even_dimensions(image)
            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise OSError(f"failed to open MP4 VideoWriter: {output}")
            writer.write(image)
            frame_count += 1
            total_fused += len(fusion_result.fused_detections)
            if frame_count % 10 == 0 or frame_count == 1:
                print(f"Rendered {frame_count} frames; latest KITTI frame={frame.frame_id:06d}")
        if not frame_count:
            raise ValueError("no KITTI frames selected by start_frame/end_frame")
        elapsed = perf_counter() - start
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    finally:
        if writer is not None:
            writer.release()

    print(f"Saved fusion video: {output}")
    print(f"Frames written: {frame_count}")
    print(f"Mode: {mode}")
    print(f"Average fused objects/frame: {total_fused / frame_count:.2f}")
    print(f"Processing time: {elapsed:.2f} s ({frame_count / elapsed:.2f} FPS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
