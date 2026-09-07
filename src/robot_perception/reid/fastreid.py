"""FastReID feature encoder kept outside the detector/tracker core."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

from robot_perception.common.types import FrameData


class FastReIDEncoder:
    """Extract L2-normalized vehicle embeddings with FastReID.

    FastReID is imported lazily. The main ``robot-perception-mot`` environment
    therefore remains usable without the optional FastReID dependency. Tests can
    inject a predictor object exposing ``__call__(tensor)``.
    """

    def __init__(
        self,
        config_path: str | Path,
        weights_path: str | Path,
        *,
        fastreid_root: str | Path | None = None,
        device: str = "cuda:0",
        predictor: Any | None = None,
        input_size: tuple[int, int] = (256, 256),
        pixel_mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
        pixel_std: tuple[float, float, float] = (58.395, 57.12, 57.375),
    ) -> None:
        if input_size[0] < 8 or input_size[1] < 8:
            raise ValueError("input_size must contain dimensions >= 8")
        if len(pixel_mean) != 3 or len(pixel_std) != 3:
            raise ValueError("pixel_mean and pixel_std must contain three values")
        if not np.isfinite(pixel_mean).all() or not np.isfinite(pixel_std).all() or (np.asarray(pixel_std) <= 0).any():
            raise ValueError("pixel normalization values must be finite and std values positive")
        self.config_path = Path(config_path)
        self.weights_path = Path(weights_path)
        self.device = str(device)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.pixel_mean = np.asarray(pixel_mean, dtype=np.float32).reshape(1, 1, 3)
        self.pixel_std = np.asarray(pixel_std, dtype=np.float32).reshape(1, 1, 3)
        self._predictor = predictor or self._build_predictor(fastreid_root)

    def _build_predictor(self, fastreid_root: str | Path | None) -> Any:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"FastReID config not found: {self.config_path}")
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"FastReID weights not found: {self.weights_path}")
        if fastreid_root is not None:
            root = str(Path(fastreid_root).resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
        try:
            # FastReID v0.1.1 still imports Mapping/Sequence from collections;
            # expose the Python 3.10-compatible aliases before importing it.
            import collections
            import collections.abc
            # PyTorch 1.10's tensorboard shim expects ``distutils.version`` to
            # be imported explicitly in some Python 3.8 environments.
            try:
                import distutils
                import distutils.version  # noqa: F401
            except ImportError:
                pass

            for name in ("Mapping", "MutableMapping", "Sequence"):
                if not hasattr(collections, name):
                    setattr(collections, name, getattr(collections.abc, name))
            from fastreid.config import get_cfg
            from fastreid.engine import DefaultPredictor
        except ImportError as error:
            raise RuntimeError(
                "FastReID is unavailable; install/checkout FastReID in an optional environment "
                "and pass --fastreid-root"
            ) from error
        cfg = get_cfg()
        cfg.merge_from_file(str(self.config_path))
        cfg.defrost()
        cfg.MODEL.WEIGHTS = str(self.weights_path)
        cfg.MODEL.DEVICE = self.device
        cfg.freeze()
        return DefaultPredictor(cfg)

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"image must have shape (H, W, 3), got {array.shape}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("image must contain finite numeric values")
        return array

    @staticmethod
    def _clip_bbox(bbox: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
        values = np.asarray(bbox, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("bbox must be a finite vector with shape (4,)")
        x1, y1, x2, y2 = values
        left = max(0, min(width - 1, int(np.floor(x1))))
        top = max(0, min(height - 1, int(np.floor(y1))))
        right = max(0, min(width, int(np.ceil(x2))))
        bottom = max(0, min(height, int(np.ceil(y2))))
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    def _preprocess_crop(self, crop: np.ndarray) -> Any:
        import cv2
        import torch

        resized = cv2.resize(crop, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
        # FastReID configs use RGB ImageNet-style normalization; input FrameData
        # images are OpenCV BGR, so reverse channels explicitly here.
        rgb = resized[:, :, ::-1].astype(np.float32)
        normalized = (rgb - self.pixel_mean) / self.pixel_std
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)
        return tensor

    def encode_crop(self, crop: np.ndarray) -> np.ndarray:
        """Encode one already-cropped BGR image into a unit-length vector."""

        image = self._validate_image(crop)
        tensor = self._preprocess_crop(image)
        output = self._predictor(tensor)
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        embedding = np.asarray(output, dtype=np.float32)
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]
        if embedding.ndim != 1 or embedding.size == 0 or not np.isfinite(embedding).all():
            raise ValueError(f"FastReID predictor must return one finite vector, got {embedding.shape}")
        norm = float(np.linalg.norm(embedding))
        if norm <= np.finfo(np.float32).eps:
            raise ValueError("FastReID predictor returned a zero vector")
        return (embedding / norm).astype(np.float32, copy=False)

    def encode(self, frame: FrameData, bbox: np.ndarray) -> np.ndarray | None:
        """Crop ``bbox`` from a FrameData image and return its embedding."""

        if frame.image is None:
            return None
        image = self._validate_image(frame.image)
        clipped = self._clip_bbox(bbox, image.shape[1], image.shape[0])
        if clipped is None:
            return None
        left, top, right, bottom = clipped
        return self.encode_crop(image[top:bottom, left:right])
