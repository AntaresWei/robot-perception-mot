"""Tests for the offline fusion video script's frame-range contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fusion_video", ROOT / "scripts" / "render_fusion_video.py")
assert SPEC is not None and SPEC.loader is not None
fusion_video = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fusion_video
SPEC.loader.exec_module(fusion_video)


def test_frame_selection_has_inclusive_end_and_negative_all_frames() -> None:
    assert fusion_video._frame_is_selected(0, 0, 153)
    assert fusion_video._frame_is_selected(153, 0, 153)
    assert not fusion_video._frame_is_selected(154, 0, 153)
    assert fusion_video._frame_is_selected(153, 0, -1)
    assert not fusion_video._frame_is_selected(3, 4, -1)


def test_video_padding_preserves_content_and_makes_dimensions_even() -> None:
    image = np.full((375, 1241, 3), 7, dtype=np.uint8)

    padded = fusion_video._pad_to_even_dimensions(image)

    assert padded.shape == (376, 1242, 3)
    np.testing.assert_array_equal(padded[:375, :1241], image)
