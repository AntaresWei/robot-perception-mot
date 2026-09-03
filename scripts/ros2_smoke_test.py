#!/usr/bin/env python3
"""Run the ROS2 package's dependency-light executable smoke test."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    package_src = Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "perception_tracker"
    module_path = package_src / "perception_tracker"
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(package_src) + ":" + env.get("PYTHONPATH", "")
    python = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
    result = subprocess.run([python, str(module_path / "perception_tracker_node.py"), "--smoke"], env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
