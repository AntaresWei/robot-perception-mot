"""Tests for the minimal KITTI Tracking dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_perception.dataset import KittiTrackingDataset


def _create_dataset_root(
    tmp_path: Path,
    *,
    label_content: str = "0 7 Car 0.0 1 -1.0 1.0 2.0 3.0 4.0 1.5 1.6 3.9 5.0 6.0 7.0 0.2\n",
    points: np.ndarray | None = None,
) -> Path:
    sequence_id = "0000"
    tracking_root = tmp_path / "track"
    image_dir = tracking_root / "data_tracking_image_2" / "training" / "image_02" / sequence_id
    lidar_dir = tracking_root / "data_tracking_velodyne" / "training" / "velodyne" / sequence_id
    label_dir = tracking_root / "data_tracking_label_2" / "training" / "label_02"
    for directory in (image_dir, lidar_dir, label_dir):
        directory.mkdir(parents=True)

    valid_points = np.asarray(((1.0, 2.0, 3.0, 0.5),), dtype=np.float32) if points is None else points
    valid_points.tofile(lidar_dir / "000000.bin")
    (label_dir / "0000.txt").write_text(label_content, encoding="utf-8")
    return tmp_path


def test_dataset_initialization(tmp_path: Path) -> None:
    dataset = KittiTrackingDataset(_create_dataset_root(tmp_path), sequences=[0], load_images=False)

    assert len(dataset) == 1
    assert dataset.sequence_ids == ("0000",)


def test_point_cloud_shape_and_frame_identity(tmp_path: Path) -> None:
    dataset = KittiTrackingDataset(_create_dataset_root(tmp_path), sequences=["0000"], load_images=False)
    frame = dataset.get_frame("0000", 0)

    assert frame.sequence_id == "0000"
    assert frame.frame_id == 0
    assert frame.lidar_points.shape == (1, 4)
    assert frame.lidar_points.dtype == np.float32
    assert frame.lidar_points.tolist() == [[1.0, 2.0, 3.0, 0.5]]
    assert frame.calibration is None


def test_label_parsing(tmp_path: Path) -> None:
    dataset = KittiTrackingDataset(_create_dataset_root(tmp_path), load_images=False)
    annotations = dataset[0].annotations

    assert annotations is not None
    annotation = annotations[0]
    assert annotation.frame_id == 0
    assert annotation.track_id == 7
    assert annotation.class_name == "Car"
    assert annotation.bbox_2d.tolist() == [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_allclose(annotation.dimensions, (3.9, 1.6, 1.5))
    np.testing.assert_allclose(annotation.location, (5.0, 6.0, 7.0))
    assert annotation.yaw == 0.2


def test_empty_label_returns_empty_annotations(tmp_path: Path) -> None:
    dataset = KittiTrackingDataset(_create_dataset_root(tmp_path, label_content=""), load_images=False)

    assert dataset[0].annotations == []


def test_invalid_frame_raises_file_not_found(tmp_path: Path) -> None:
    dataset = KittiTrackingDataset(_create_dataset_root(tmp_path), load_images=False)

    with pytest.raises(FileNotFoundError, match="sequence=0000, frame=000099"):
        dataset.get_frame("0000", 99)


def test_invalid_lidar_size_raises_value_error(tmp_path: Path) -> None:
    root = _create_dataset_root(tmp_path)
    lidar_path = root / "track" / "data_tracking_velodyne" / "training" / "velodyne" / "0000" / "000000.bin"
    np.asarray((1.0, 2.0, 3.0), dtype=np.float32).tofile(lidar_path)
    dataset = KittiTrackingDataset(root, load_images=False)

    with pytest.raises(ValueError, match="N x 4"):
        dataset[0]


@pytest.mark.parametrize("invalid_value", (np.nan, np.inf))
def test_non_finite_lidar_raises_value_error(tmp_path: Path, invalid_value: float) -> None:
    points = np.asarray(((1.0, 2.0, 3.0, invalid_value),), dtype=np.float32)
    dataset = KittiTrackingDataset(_create_dataset_root(tmp_path, points=points), load_images=False)

    with pytest.raises(ValueError, match="NaN or Inf"):
        dataset[0]


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="local KITTI data is unavailable")
def test_reads_local_kitti_frame_100() -> None:
    dataset = KittiTrackingDataset(DATA_ROOT, sequences=["0000"])
    frame = dataset.get_frame("0000", 100)

    assert frame.image is not None
    assert frame.image.ndim == 3
    assert frame.lidar_points.shape[1] == 4
    assert frame.annotations is not None
