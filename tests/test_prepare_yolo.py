"""Tests for KITTI-to-YOLO label conversion boundaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_yolo", ROOT / "scripts" / "prepare_kitti_yolo.py")
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_explicit_validation_sequences_do_not_overlap_training() -> None:
    train, validation = module._sequence_split(["0000", "0001", "0002", "0003"])

    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == {"0000", "0001", "0002", "0003"}


def test_kitti_class_mapping_is_limited_to_three_project_classes() -> None:
    assert module.CLASS_IDS["Car"] == module.CLASS_IDS["Van"] == module.CLASS_IDS["Truck"] == 0
    assert module.CLASS_IDS["Pedestrian"] == 1
    assert module.CLASS_IDS["Cyclist"] == 2
    assert "DontCare" not in module.CLASS_IDS
