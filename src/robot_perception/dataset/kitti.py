"""KITTI Tracking dataset reader without detector or training-framework dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_perception.common.types import Calibration, FrameData, KittiTrackingAnnotation


@dataclass(frozen=True)
class _DatasetPaths:
    image_dir: Path
    lidar_dir: Path
    calibration_dir: Path
    label_dir: Path | None


@dataclass(frozen=True)
class _FrameKey:
    sequence_id: str
    frame_id: int


class KittiTrackingDataset(Sequence[FrameData]):
    """Read KITTI Tracking frames from standard or split-download directory layouts.

    Images are returned in OpenCV BGR order. LiDAR points are `float32` arrays
    with shape `(N, 4)` in `[x, y, z, intensity]` order. Labels are available
    only for the training split.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "training",
        sequences: Iterable[str | int] | None = None,
        load_images: bool = True,
        load_annotations: bool = True,
    ) -> None:
        if split not in {"training", "testing"}:
            raise ValueError("split must be 'training' or 'testing'")

        self.root = Path(root)
        self.split = split
        self.load_images = load_images
        self.load_annotations = load_annotations
        self._paths = self._resolve_paths(self.root, split)

        if load_annotations and split == "training" and self._paths.label_dir is None:
            raise FileNotFoundError("training labels were requested but label_02 is unavailable")

        available_sequences = self._available_sequences()
        requested_sequences = (
            available_sequences
            if sequences is None
            else tuple(self._normalise_sequence_id(sequence) for sequence in sequences)
        )
        missing_sequences = sorted(set(requested_sequences) - set(available_sequences))
        if missing_sequences:
            joined = ", ".join(missing_sequences)
            raise FileNotFoundError(f"requested KITTI sequences are unavailable: {joined}")

        self.sequence_ids = tuple(requested_sequences)
        self._frames = tuple(
            _FrameKey(sequence_id, int(path.stem))
            for sequence_id in self.sequence_ids
            for path in sorted((self._paths.lidar_dir / sequence_id).glob("*.bin"))
        )
        if not self._frames:
            raise ValueError("no LiDAR frames were found for the requested KITTI sequences")

        self._annotations_by_sequence: dict[str, tuple[KittiTrackingAnnotation, ...]] = {}
        self._calibrations: dict[str, Calibration] = {}

    @staticmethod
    def _resolve_paths(root: Path, split: str) -> _DatasetPaths:
        standard_root = root / split
        standard_label_dir = standard_root / "label_02"
        standard_paths = _DatasetPaths(
            image_dir=standard_root / "image_02",
            lidar_dir=standard_root / "velodyne",
            calibration_dir=standard_root / "calib",
            label_dir=standard_label_dir if standard_label_dir.is_dir() else None,
        )
        if all(
            path.is_dir()
            for path in (
                standard_paths.image_dir,
                standard_paths.lidar_dir,
                standard_paths.calibration_dir,
            )
        ):
            return standard_paths

        split_download_label_dir = root / "data_tracking_label_2" / split / "label_02"
        split_download_paths = _DatasetPaths(
            image_dir=root / "data_tracking_image_2" / split / "image_02",
            lidar_dir=root / "data_tracking_velodyne" / split / "velodyne",
            calibration_dir=root / "data_tracking_calib" / split / "calib",
            label_dir=split_download_label_dir if split_download_label_dir.is_dir() else None,
        )
        if all(
            path.is_dir()
            for path in (
                split_download_paths.image_dir,
                split_download_paths.lidar_dir,
                split_download_paths.calibration_dir,
            )
        ):
            return split_download_paths

        raise FileNotFoundError(
            "KITTI Tracking layout not found under "
            f"{root}. Expected either {standard_root} or data_tracking_* directories."
        )

    @staticmethod
    def _normalise_sequence_id(sequence: str | int) -> str:
        if isinstance(sequence, int):
            return f"{sequence:04d}"
        if sequence.isdigit():
            return f"{int(sequence):04d}"
        raise ValueError(f"invalid KITTI sequence identifier: {sequence!r}")

    def _available_sequences(self) -> tuple[str, ...]:
        return tuple(sorted(path.name for path in self._paths.lidar_dir.iterdir() if path.is_dir()))

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int | slice) -> FrameData | list[FrameData]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]

        frame_key = self._frames[index]
        image = self._read_image(frame_key) if self.load_images else None
        annotations = self._annotations_for_frame(frame_key) if self.load_annotations else None
        return FrameData(
            sequence_id=frame_key.sequence_id,
            frame_id=frame_key.frame_id,
            image=image,
            lidar_points=self._read_lidar(frame_key),
            calibration=self._calibration_for_sequence(frame_key.sequence_id),
            annotations=annotations,
            timestamp=None,
        )

    def __iter__(self) -> Iterator[FrameData]:
        for index in range(len(self)):
            yield self[index]

    def _read_image(self, frame_key: _FrameKey) -> np.ndarray:
        image_path = self._paths.image_dir / frame_key.sequence_id / f"{frame_key.frame_id:06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"image file not found: {image_path}")

        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required when load_images=True") from error

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode KITTI image: {image_path}")
        return image

    def _read_lidar(self, frame_key: _FrameKey) -> np.ndarray:
        lidar_path = self._paths.lidar_dir / frame_key.sequence_id / f"{frame_key.frame_id:06d}.bin"
        points = np.fromfile(lidar_path, dtype=np.float32)
        if points.size % 4 != 0:
            raise ValueError(f"LiDAR file does not contain N x 4 float32 values: {lidar_path}")
        return points.reshape(-1, 4)

    def _calibration_for_sequence(self, sequence_id: str) -> Calibration:
        if sequence_id not in self._calibrations:
            calibration_path = self._paths.calibration_dir / f"{sequence_id}.txt"
            self._calibrations[sequence_id] = self._parse_calibration(calibration_path)
        return self._calibrations[sequence_id]

    def _annotations_for_frame(self, frame_key: _FrameKey) -> list[KittiTrackingAnnotation] | None:
        if self._paths.label_dir is None:
            return None

        sequence_id = frame_key.sequence_id
        if sequence_id not in self._annotations_by_sequence:
            label_path = self._paths.label_dir / f"{sequence_id}.txt"
            self._annotations_by_sequence[sequence_id] = self._parse_annotations(label_path)
        return [
            annotation
            for annotation in self._annotations_by_sequence[sequence_id]
            if annotation.frame_id == frame_key.frame_id
        ]

    @staticmethod
    def _parse_calibration(path: Path) -> Calibration:
        if not path.is_file():
            raise FileNotFoundError(f"calibration file not found: {path}")

        matrices: dict[str, np.ndarray] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.replace(":", " ", 1).split()
            if len(fields) < 2:
                continue
            matrices[fields[0]] = np.asarray(fields[1:], dtype=np.float32)

        def matrix(*names: str, shape: tuple[int, int]) -> np.ndarray:
            for name in names:
                values = matrices.get(name)
                if values is not None:
                    if values.size != shape[0] * shape[1]:
                        raise ValueError(f"invalid {name} matrix shape in {path}")
                    return values.reshape(shape)
            joined = ", ".join(names)
            raise ValueError(f"missing calibration matrix ({joined}) in {path}")

        return Calibration(
            p0=matrix("P0", shape=(3, 4)),
            p1=matrix("P1", shape=(3, 4)),
            p2=matrix("P2", shape=(3, 4)),
            p3=matrix("P3", shape=(3, 4)),
            r0_rect=matrix("R0_rect", "R_rect", shape=(3, 3)),
            tr_velo_to_cam=matrix("Tr_velo_to_cam", "Tr_velo_cam", "Tr_velo2cam", shape=(3, 4)),
            tr_imu_to_velo=matrix("Tr_imu_to_velo", "Tr_imu_velo", shape=(3, 4)),
        )

    @staticmethod
    def _parse_annotations(path: Path) -> tuple[KittiTrackingAnnotation, ...]:
        if not path.is_file():
            raise FileNotFoundError(f"label file not found: {path}")

        annotations: list[KittiTrackingAnnotation] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 17:
                raise ValueError(f"invalid KITTI Tracking label at {path}:{line_number}")

            annotations.append(
                KittiTrackingAnnotation(
                    frame_id=int(fields[0]),
                    track_id=int(fields[1]),
                    class_name=fields[2],
                    truncation=float(fields[3]),
                    occlusion=int(fields[4]),
                    alpha=float(fields[5]),
                    bbox_2d=np.asarray(fields[6:10], dtype=np.float32),
                    dimensions=np.asarray((fields[12], fields[11], fields[10]), dtype=np.float32),
                    location=np.asarray(fields[13:16], dtype=np.float32),
                    yaw=float(fields[16]),
                    score=float(fields[17]) if len(fields) > 17 else None,
                )
            )
        return tuple(annotations)
