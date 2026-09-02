"""Self-contained 3D constant-velocity Kalman filter."""

from __future__ import annotations

from typing import Any

import numpy as np


STATE_DIM = 6
MEASUREMENT_DIM = 3


def _as_vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    return vector


def _as_covariance(value: Any, size: int, name: str) -> np.ndarray:
    # 允许标量、对角向量或完整矩阵，统一转换为方阵后再做 PSD 校验。
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        if not np.isfinite(array).all() or float(array) < 0.0:
            raise ValueError(f"{name} scalar must be finite and non-negative")
        matrix = np.eye(size, dtype=np.float64) * float(array)
    elif array.shape == (size,):
        if not np.isfinite(array).all() or (array < 0.0).any():
            raise ValueError(f"{name} diagonal must be finite and non-negative")
        matrix = np.diag(array)
    elif array.shape == (size, size):
        matrix = array.copy()
    else:
        raise ValueError(f"{name} must be scalar, ({size},), or ({size}, {size}), got {array.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min(initial=0.0) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return (matrix + matrix.T) * 0.5


def _stabilize_covariance(covariance: np.ndarray, floor: float) -> np.ndarray:
    # 浮点误差可能破坏对称性或产生极小负特征值；谱分解后设置下限可恢复稳定性。
    symmetric = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if not np.isfinite(eigenvalues).all():
        raise FloatingPointError("covariance eigenvalues contain NaN or Inf values")
    stabilized = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return (stabilized + stabilized.T) * 0.5


class ConstantVelocityKalmanFilter:
    """Estimate 3D position and velocity with a constant-velocity model.

    State order is ``[px, py, pz, vx, vy, vz]`` and measurements are
    ``[px, py, pz]``. Methods mutate the filter state and return a copy.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        *,
        initial_velocity: np.ndarray | None = None,
        initial_covariance: np.ndarray | float = 1.0,
        process_noise: np.ndarray | float = 1e-2,
        measurement_noise: np.ndarray | float = 1e-1,
        covariance_floor: float = 1e-12,
    ) -> None:
        if covariance_floor <= 0.0 or not np.isfinite(covariance_floor):
            raise ValueError("covariance_floor must be finite and positive")
        position = _as_vector(initial_position, 3, "initial_position")
        velocity = np.zeros(3, dtype=np.float64) if initial_velocity is None else _as_vector(initial_velocity, 3, "initial_velocity")
        self._state = np.concatenate((position, velocity))
        self._covariance = _as_covariance(initial_covariance, STATE_DIM, "initial_covariance")
        self._process_noise = _as_covariance(process_noise, STATE_DIM, "process_noise")
        self._measurement_noise = _as_covariance(measurement_noise, MEASUREMENT_DIM, "measurement_noise")
        self._covariance_floor = float(covariance_floor)
        self._measurement_matrix = np.hstack((np.eye(3), np.zeros((3, 3))))
        self._transition_matrix = np.eye(STATE_DIM)

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    @property
    def measurement_matrix(self) -> np.ndarray:
        return self._measurement_matrix.copy()

    @property
    def transition_matrix(self) -> np.ndarray:
        return self._transition_matrix.copy()

    @property
    def process_noise(self) -> np.ndarray:
        return self._process_noise.copy()

    @property
    def measurement_noise(self) -> np.ndarray:
        return self._measurement_noise.copy()

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Advance the state by ``dt`` seconds and return the predicted state."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        transition = np.eye(STATE_DIM, dtype=np.float64)
        # 常速度模型只有位置对速度的 dt 耦合，速度本身保持不变。
        transition[:3, 3:] = np.eye(3) * float(dt)
        predicted_state = transition @ self._state
        predicted_covariance = transition @ self._covariance @ transition.T + self._process_noise
        if not np.isfinite(predicted_state).all() or not np.isfinite(predicted_covariance).all():
            raise FloatingPointError("prediction produced NaN or Inf values")
        self._transition_matrix = transition
        self._state = predicted_state
        self._covariance = _stabilize_covariance(predicted_covariance, self._covariance_floor)
        return self.state

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Correct the state with one 3D position measurement."""

        observation = _as_vector(measurement, MEASUREMENT_DIM, "measurement")
        innovation = observation - self._measurement_matrix @ self._state
        innovation_covariance = (
            self._measurement_matrix @ self._covariance @ self._measurement_matrix.T
            + self._measurement_noise
        )
        if not np.isfinite(innovation_covariance).all():
            raise FloatingPointError("update produced NaN or Inf innovation covariance")
        try:
            cross_covariance = self._covariance @ self._measurement_matrix.T
            # 解 S X = (P H^T)^T，避免显式求逆；最后转置得到 K=P H^T S^-1。
            gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError("innovation covariance is singular") from error

        identity = np.eye(STATE_DIM)
        residual_projection = identity - gain @ self._measurement_matrix
        updated_state = self._state + gain @ innovation
        # Joseph 形式比 (I-KH)P 更能抵抗舍入误差，适合长期迭代。
        updated_covariance = (
            residual_projection @ self._covariance @ residual_projection.T
            + gain @ self._measurement_noise @ gain.T
        )
        if not np.isfinite(updated_state).all() or not np.isfinite(updated_covariance).all():
            raise FloatingPointError("update produced NaN or Inf values")
        self._state = updated_state
        self._covariance = _stabilize_covariance(updated_covariance, self._covariance_floor)
        return self.state


KalmanFilter = ConstantVelocityKalmanFilter
