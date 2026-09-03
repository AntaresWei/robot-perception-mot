"""Static and executable checks for the ROS2 adapter package."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_ros2_package_layout() -> None:
    package = ROOT / "ros2_ws" / "src" / "perception_tracker"
    assert (package / "package.xml").is_file()
    assert (package / "setup.py").is_file()
    assert (package / "perception_tracker" / "perception_tracker_node.py").is_file()
    assert (package / "launch" / "perception_tracker.launch.py").is_file()


def test_ros2_entrypoint_smoke() -> None:
    script = ROOT / "scripts" / "ros2_smoke_test.py"
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "smoke test passed" in result.stdout
