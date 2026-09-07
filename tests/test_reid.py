"""Tests for FastReID adapter preprocessing and gated identity gallery."""

from __future__ import annotations

import numpy as np
import pytest

from robot_perception.common.types import FrameData
from robot_perception.reid import AppearanceGallery, FastReIDEncoder, ReIDIdentityManager


class _FakePredictor:
    def __init__(self) -> None:
        self.input = None

    def __call__(self, tensor):
        self.input = tensor
        return np.asarray([[3.0, 4.0]], dtype=np.float32)


def _frame() -> FrameData:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    return FrameData("0000", 0, image, np.empty((0, 4)), None)


def test_fastreid_encoder_crops_and_normalizes_without_importing_fastreid() -> None:
    predictor = _FakePredictor()
    encoder = FastReIDEncoder("unused.yml", "unused.pth", predictor=predictor)

    embedding = encoder.encode(_frame(), np.asarray((50.0, 10.0, 150.0, 90.0)))

    assert embedding is not None
    np.testing.assert_allclose(embedding, (0.6, 0.8), atol=1e-6)
    assert tuple(predictor.input.shape) == (1, 3, 256, 256)


def test_gallery_match_requires_gap_and_similarity() -> None:
    gallery = AppearanceGallery(momentum=0.0, max_age_frames=100)
    gallery.update(7, 1, np.asarray((1.0, 0.0)), 0)

    assert gallery.match(np.asarray((1.0, 0.0)), class_id=1, frame_id=5, min_similarity=0.9, min_gap_frames=10) is None
    match = gallery.match(np.asarray((1.0, 0.0)), class_id=1, frame_id=10, min_similarity=0.9, min_gap_frames=10)
    assert match is not None and match.track_id == 7


def test_reid_manager_reactivates_only_after_reliable_embedding() -> None:
    predictor = _FakePredictor()
    manager = ReIDIdentityManager(
        FastReIDEncoder("unused.yml", "unused.pth", predictor=predictor),
        similarity_threshold=0.99,
        min_gap_frames=3,
        min_fusion_confidence=0.5,
    )
    frame = _frame()
    first = manager.resolve(
        internal_track_id=1, class_id=1, frame_id=0, frame=frame,
        bbox=np.asarray((50.0, 10.0, 150.0, 90.0)), fusion_confidence=0.9,
        active_public_ids=set(),
    )
    assert first.public_track_id == 1 and not first.reactivated
    recovered = manager.resolve(
        internal_track_id=9, class_id=1, frame_id=3, frame=frame,
        bbox=np.asarray((50.0, 10.0, 150.0, 90.0)), fusion_confidence=0.9,
        active_public_ids=set(),
    )
    assert recovered.public_track_id == 1
    assert recovered.reactivated


def test_reid_manager_does_not_reactivate_active_identity() -> None:
    predictor = _FakePredictor()
    manager = ReIDIdentityManager(FastReIDEncoder("unused.yml", "unused.pth", predictor=predictor), min_gap_frames=1)
    frame = _frame()
    manager.resolve(internal_track_id=1, class_id=1, frame_id=0, frame=frame, bbox=np.asarray((1, 1, 50, 50)), fusion_confidence=1.0, active_public_ids=set())
    decision = manager.resolve(internal_track_id=9, class_id=1, frame_id=5, frame=frame, bbox=np.asarray((1, 1, 50, 50)), fusion_confidence=1.0, active_public_ids={1})
    assert decision.public_track_id == 9
