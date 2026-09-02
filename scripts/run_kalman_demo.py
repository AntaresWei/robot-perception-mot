#!/usr/bin/env python3
"""Compare noisy position measurements with a Kalman-filtered trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robot_perception.tracking import ConstantVelocityKalmanFilter


def _load_config(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read the Kalman demo configuration") from error
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict) or not isinstance(config.get("kalman"), dict):
        raise ValueError(f"configuration must contain a kalman mapping: {path}")
    return config["kalman"]


def _diagonal(config: object, size: int, name: str) -> np.ndarray:
    # YAML 用 diagonal 表达协方差，交给 KF 后统一转换为方阵。
    if not isinstance(config, dict) or "diagonal" not in config:
        raise ValueError(f"{name} must contain a diagonal list")
    values = np.asarray(config["diagonal"], dtype=np.float64)
    if values.shape != (size,):
        raise ValueError(f"{name}.diagonal must have {size} values")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "kalman.yaml")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "outputs" / "metrics" / "kalman_baseline.json")
    args = parser.parse_args()

    try:
        config = _load_config(args.config)
        trajectory = config.get("trajectory")
        if not isinstance(trajectory, dict):
            raise ValueError("kalman.trajectory mapping is required")
        num_steps = int(trajectory["num_steps"])
        dt = float(trajectory["dt"])
        if num_steps < 2 or dt <= 0.0:
            raise ValueError("trajectory.num_steps must be at least 2 and dt must be positive")
        initial_position = np.asarray(trajectory["initial_position"], dtype=np.float64)
        velocity = np.asarray(trajectory["velocity"], dtype=np.float64)
        measurement_std = np.asarray(trajectory["measurement_std"], dtype=np.float64)
        if initial_position.shape != (3,) or velocity.shape != (3,) or measurement_std.shape != (3,):
            raise ValueError("trajectory vectors must all have shape (3,)")
        if (measurement_std <= 0.0).any() or not np.isfinite(measurement_std).all():
            raise ValueError("measurement_std must be finite and positive")

        steps = np.arange(num_steps, dtype=np.float64)
        truth = initial_position + steps[:, None] * dt * velocity
        # 合成观测只在 truth 上加入高斯噪声，用于比较滤波前后的定位误差。
        rng = np.random.default_rng(int(trajectory["seed"]))
        measurements = truth + rng.normal(0.0, measurement_std, size=truth.shape)
        measurement_noise = np.square(measurement_std)
        kalman = ConstantVelocityKalmanFilter(
            measurements[0],
            initial_velocity=np.zeros(3),
            initial_covariance=_diagonal(config.get("initial_covariance"), 6, "initial_covariance"),
            process_noise=_diagonal(config.get("process_noise"), 6, "process_noise"),
            measurement_noise=measurement_noise,
            covariance_floor=float(config.get("covariance_floor", 1e-12)),
        )
        filtered_positions = [kalman.state[:3]]
        for measurement in measurements[1:]:
            kalman.predict(dt=dt)
            filtered_positions.append(kalman.update(measurement)[:3])
        filtered = np.asarray(filtered_positions)
        raw_rmse = float(np.sqrt(np.mean((measurements - truth) ** 2)))
        filtered_rmse = float(np.sqrt(np.mean((filtered - truth) ** 2)))
        result = {
            "dataset": "synthetic_constant_velocity",
            "num_steps": num_steps,
            "dt": dt,
            "seed": int(trajectory["seed"]),
            "raw_measurement_rmse": raw_rmse,
            "kalman_filtered_rmse": filtered_rmse,
            "improvement": raw_rmse - filtered_rmse,
            "config": str(args.config),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    print(f"Saved metrics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
