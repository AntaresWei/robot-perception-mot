"""Tests for KITTI Tracking file parsing and local-data integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_perception.dataset import KittiTrackingDataset


def _write_calibration(path: Path) -> None:
    projection = " ".join(str(value) for value in range(12))
    rotation = " ".join(str(value) for value in range(9))
    transform = " ".join(str(value) for value in range(12))
    path.write_text(
        "\n".join(
            (
                f"P0: {projection}",
                f"P1: {projection}",
                f"P2: {projection}",
                f"P3: {projection}",
                f"R_rect {rotation}",
                f"Tr_velo_cam {transform}",
                f"Tr_imu_velo {transform}",
            )
        ),
        encoding="utf-8",
    )


def test_reads_split_download_layout(tmp_path: Path) -> None:
    sequence_id = "0000"
    image_dir = tmp_path / "data_tracking_image_2" / "training" / "image_02" / sequence_id
    lidar_dir = tmp_path / "data_tracking_velodyne" / "training" / "velodyne" / sequence_id
    calibration_dir = tmp_path / "data_tracking_calib" / "training" / "calib"
    label_dir = tmp_path / "data_tracking_label_2" / "training" / "label_02"
    for directory in (image_dir, lidar_dir, calibration_dir, label_dir):
        directory.mkdir(parents=True)

    np.asarray(((1.0, 2.0, 3.0, 0.5),), dtype=np.float32).tofile(lidar_dir / "000000.bin")
    _write_calibration(calibration_dir / "0000.txt")
    (label_dir / "0000.txt").write_text(
        "0 7 Car 0.0 1 -1.0 1.0 2.0 3.0 4.0 1.5 1.6 3.9 5.0 6.0 7.0 0.2\n",
        encoding="utf-8",
    )

    dataset = KittiTrackingDataset(tmp_path, sequences=[0], load_images=False)
    frame = dataset[0]

    assert len(dataset) == 1
    assert frame.sequence_id == "0000"
    assert frame.lidar_points.shape == (1, 4)
    assert frame.calibration is not None
    assert frame.calibration.p2.shape == (3, 4)
    assert frame.annotations is not None
    assert frame.annotations[0].track_id == 7
    assert frame.annotations[0].dimensions.tolist() == [3.9, 1.6, 1.5]


def test_testing_split_without_labels_returns_no_annotations(tmp_path: Path) -> None:
    sequence_id = "0000"
    image_dir = tmp_path / "data_tracking_image_2" / "testing" / "image_02" / sequence_id
    lidar_dir = tmp_path / "data_tracking_velodyne" / "testing" / "velodyne" / sequence_id
    calibration_dir = tmp_path / "data_tracking_calib" / "testing" / "calib"
    for directory in (image_dir, lidar_dir, calibration_dir):
        directory.mkdir(parents=True)

    np.zeros((1, 4), dtype=np.float32).tofile(lidar_dir / "000000.bin")
    _write_calibration(calibration_dir / "0000.txt")

    dataset = KittiTrackingDataset(tmp_path, split="testing", sequences=[0], load_images=False)

    assert dataset[0].annotations is None


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="local KITTI data is unavailable")
def test_reads_local_kitti_sequence_zero() -> None:
    dataset = KittiTrackingDataset(DATA_ROOT, sequences=["0000"])
    frame = dataset[0]

    assert frame.image is not None
    assert frame.image.ndim == 3
    assert frame.lidar_points.dtype == np.float32
    assert frame.lidar_points.shape[1] == 4
    assert frame.calibration is not None
    assert frame.calibration.tr_velo_to_cam.shape == (3, 4)
    assert frame.annotations is not None
    assert frame.annotations[0].frame_id == 0
