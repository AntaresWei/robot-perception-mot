"""Calibration parser and vectorized projection tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_perception.common.geometry import camera_to_image, lidar_to_camera, lidar_to_image
from robot_perception.dataset.calibration import KittiCalibration, parse_kitti_calibration


def _calibration(
    *,
    p2: np.ndarray | None = None,
    rectification: np.ndarray | None = None,
    transform: np.ndarray | None = None,
) -> KittiCalibration:
    return KittiCalibration(
        P2=np.eye(3, 4) if p2 is None else p2,
        R0_rect=np.eye(3) if rectification is None else rectification,
        Tr_velo_to_cam=np.hstack((np.eye(3), np.zeros((3, 1)))) if transform is None else transform,
    )


def _write_calibration(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "P2: " + " ".join(str(value) for value in range(12)),
                "R0_rect: " + " ".join(str(value) for value in range(9)),
                "Tr_velo_to_cam: " + " ".join(str(value) for value in range(12)),
            )
        ),
        encoding="utf-8",
    )


def test_calibration_parser_reads_required_matrices(tmp_path: Path) -> None:
    path = tmp_path / "0000.txt"
    _write_calibration(path)

    calibration = parse_kitti_calibration(path)

    assert calibration.P2.shape == (3, 4)
    assert calibration.R0_rect.shape == (3, 3)
    assert calibration.Tr_velo_to_cam.shape == (3, 4)


def test_identity_transform() -> None:
    points = np.asarray(((1.0, 2.0, 4.0), (2.0, -1.0, 2.0)))

    result = lidar_to_camera(points, _calibration())

    np.testing.assert_allclose(result, points)


def test_known_transform_uses_homogeneous_translation() -> None:
    transform = np.hstack((np.eye(3), np.asarray(((10.0,), (20.0,), (30.0,)))))
    points = np.asarray(((1.0, 2.0, 3.0),))

    result = lidar_to_camera(points, _calibration(transform=transform))

    np.testing.assert_allclose(result, ((11.0, 22.0, 33.0),))


def test_homogeneous_projection() -> None:
    p2 = np.asarray(((2.0, 0.0, 10.0, 0.0), (0.0, 3.0, 20.0, 0.0), (0.0, 0.0, 1.0, 0.0)))

    result = camera_to_image(np.asarray(((1.0, 2.0, 4.0),)), _calibration(p2=p2))

    np.testing.assert_allclose(result, ((10.5, 21.5),))


def test_points_behind_camera_are_filtered() -> None:
    points = np.asarray(((1.0, 2.0, 4.0), (1.0, 2.0, 0.0), (1.0, 2.0, -1.0)))

    pixels, mask = camera_to_image(points, _calibration(), return_mask=True)

    np.testing.assert_allclose(pixels, ((0.25, 0.5),))
    assert mask.tolist() == [True, False, False]


def test_image_boundary_is_half_open() -> None:
    points = np.asarray(((0.0, 0.0, 1.0), (19.0, 9.0, 1.0), (20.0, 9.0, 1.0), (-1.0, 0.0, 1.0)))

    pixels, mask = camera_to_image(points, _calibration(), image_shape=(10, 20), return_mask=True)

    np.testing.assert_allclose(pixels, ((0.0, 0.0), (19.0, 9.0)))
    assert mask.tolist() == [True, True, False, False]


def test_invalid_points_are_filtered_by_lidar_to_image() -> None:
    points = np.asarray(((1.0, 2.0, 4.0, 0.5), (np.nan, 0.0, 1.0, 0.2)))

    pixels, mask = lidar_to_image(points, _calibration(), return_mask=True)

    np.testing.assert_allclose(pixels, ((0.25, 0.5),))
    assert mask.tolist() == [True, False]


@pytest.mark.parametrize("points", (np.zeros(3), np.zeros((2, 2)), np.zeros((2, 5))))
def test_invalid_input_shape(points: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        lidar_to_camera(points, _calibration())


def test_invalid_calibration_shape() -> None:
    with pytest.raises(ValueError, match="P2"):
        KittiCalibration(P2=np.eye(3), R0_rect=np.eye(3), Tr_velo_to_cam=np.eye(3, 4))
