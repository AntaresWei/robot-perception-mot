"""Minimal KITTI camera calibration parser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _matrix(values: list[str], shape: tuple[int, int], name: str, path: Path) -> np.ndarray:
    # 先校验元素数量，再 reshape，避免格式错误被 NumPy 静默解释成错误矩阵。
    if len(values) != shape[0] * shape[1]:
        raise ValueError(f"{name} in {path} must contain {shape[0] * shape[1]} values")
    matrix = np.asarray(values, dtype=np.float64).reshape(shape)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} in {path} contains NaN or Inf values")
    return matrix


@dataclass(frozen=True)
class KittiCalibration:
    """The matrices required for KITTI LiDAR-to-image projection."""

    P2: np.ndarray
    R0_rect: np.ndarray
    Tr_velo_to_cam: np.ndarray

    def __post_init__(self) -> None:
        for name, value, shape in (
            ("P2", self.P2, (3, 4)),
            ("R0_rect", self.R0_rect, (3, 3)),
            ("Tr_velo_to_cam", self.Tr_velo_to_cam, (3, 4)),
        ):
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
            if not np.isfinite(matrix).all():
                raise ValueError(f"{name} contains NaN or Inf values")
            object.__setattr__(self, name, matrix)


# Short name for callers that use the generic calibration terminology.
Calibration = KittiCalibration


def parse_kitti_calibration(path: str | Path) -> KittiCalibration:
    """Parse P2, R0_rect, and Tr_velo_to_cam from a KITTI calibration file."""

    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"calibration file not found: {calibration_path}")

    values_by_key: dict[str, list[str]] = {}
    for line_number, line in enumerate(calibration_path.read_text(encoding="utf-8").splitlines(), start=1):
        # KITTI 文件既有 ``P2: ...`` 也有 ``P2 ...``，统一去掉首个冒号。
        fields = line.replace(":", " ", 1).split()
        if not fields:
            continue
        if len(fields) == 1:
            raise ValueError(f"invalid calibration line at {calibration_path}:{line_number}")
        values_by_key[fields[0]] = fields[1:]

    def required(*keys: str) -> tuple[str, list[str]]:
        for key in keys:
            if key in values_by_key:
                return key, values_by_key[key]
        raise ValueError(f"missing calibration key ({', '.join(keys)}) in {calibration_path}")

    p2_key, p2_values = required("P2")
    # 不依赖固定行号，只按键名读取，兼容不同 KITTI 发行版本的行顺序。
    rect_key, rect_values = required("R0_rect", "R_rect")
    transform_key, transform_values = required("Tr_velo_to_cam", "Tr_velo_cam", "Tr_velo2cam")
    return KittiCalibration(
        P2=_matrix(p2_values, (3, 4), p2_key, calibration_path),
        R0_rect=_matrix(rect_values, (3, 3), rect_key, calibration_path),
        Tr_velo_to_cam=_matrix(transform_values, (3, 4), transform_key, calibration_path),
    )


load_calibration = parse_kitti_calibration
