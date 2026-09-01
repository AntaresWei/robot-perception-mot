"""Smoke tests for the initial package namespace layout."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "robot_perception",
        "robot_perception.common",
        "robot_perception.dataset",
        "robot_perception.pointcloud",
        "robot_perception.detection",
        "robot_perception.tracking",
        "robot_perception.fusion",
        "robot_perception.evaluation",
        "robot_perception.visualization",
    ],
)
def test_package_namespace_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
