"""Command-line entry point for the replay-oriented perception ROS2 node.

ROS imports are intentionally delayed until the node is actually started.  This
keeps ``--smoke`` useful on machines where ROS2 is not sourced and keeps the
core package independent from ``rclpy``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _smoke_check() -> int:
    """Run a dependency-light executable check for packaging/entry-point errors."""

    package_root = Path(__file__).resolve().parents[1]
    required = (
        package_root / "package.xml",
        package_root / "setup.py",
        package_root / "launch" / "perception_tracker.launch.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("ROS2 package smoke test failed; missing: " + ", ".join(missing), file=sys.stderr)
        return 1
    print("perception_tracker package smoke test passed")
    return 0


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the KITTI replay perception ROS2 node")
    parser.add_argument("--smoke", action="store_true", help="check package files without starting ROS2")
    parsed, ros_args = parser.parse_known_args(args)
    if parsed.smoke:
        return _smoke_check()

    from perception_tracker.node import main as ros_main

    return ros_main(ros_args)


if __name__ == "__main__":  # pragma: no cover - exercised by the executable smoke test
    raise SystemExit(main())
