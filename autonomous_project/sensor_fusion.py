"""Sensor-fusion and localization primitives.

The filter tracks a 2D constant-velocity vehicle state:

    state = [x, y, vx, vy]

This module intentionally keeps the math transparent so it can be discussed in
interviews and tested without heavyweight robotics dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FilterConfig:
    """Noise configuration for the 2D Kalman Filter."""

    process_var: float = 0.05
    position_var: float = 0.25
    velocity_var: float = 0.10
    initial_covariance: float = 1.0


class KalmanFilter2D:
    """2D constant-velocity Kalman Filter for vehicle localization."""

    state_dim = 4

    def __init__(self, config: FilterConfig | None = None):
        self.config = config or FilterConfig()
        self.x = np.zeros(self.state_dim, dtype=float)
        self.P = np.eye(self.state_dim, dtype=float) * self.config.initial_covariance
        self.Q = np.eye(self.state_dim, dtype=float) * self.config.process_var
        self.R = np.diag(
            [
                self.config.position_var,
                self.config.position_var,
                self.config.velocity_var,
                self.config.velocity_var,
            ]
        )

    @staticmethod
    def transition_matrix(dt: float) -> np.ndarray:
        """Return the constant-velocity transition matrix for time step ``dt``."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        """Reset state and covariance."""
        self.x = np.zeros(self.state_dim, dtype=float)
        if initial_state is not None:
            self.x = self._validate_vector(initial_state, name="initial_state")
        self.P = np.eye(self.state_dim, dtype=float) * self.config.initial_covariance

    def predict(self, dt: float) -> np.ndarray:
        """Run the prediction step and return the predicted state."""
        F = self.transition_matrix(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, measurement: np.ndarray, measurement_matrix: np.ndarray | None = None) -> np.ndarray:
        """Fuse a measurement and return the updated state.

        Parameters
        ----------
        measurement:
            Either a 4D vector ``[x, y, vx, vy]`` or a vector matching the
            provided measurement matrix.
        measurement_matrix:
            Optional matrix mapping state space into measurement space. Defaults
            to identity for full-state measurements.
        """
        z = np.asarray(measurement, dtype=float)
        H = measurement_matrix if measurement_matrix is not None else np.eye(self.state_dim)
        H = np.asarray(H, dtype=float)
        if H.shape[1] != self.state_dim:
            raise ValueError("measurement_matrix must have four state columns")
        if z.shape != (H.shape[0],):
            raise ValueError(f"measurement must have shape {(H.shape[0],)}")

        R = self.R if H.shape[0] == self.state_dim else np.eye(H.shape[0]) * self.config.position_var
        innovation = z - H @ self.x
        innovation_cov = H @ self.P @ H.T + R
        kalman_gain = self.P @ H.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + kalman_gain @ innovation

        identity = np.eye(self.state_dim)
        self.P = (identity - kalman_gain @ H) @ self.P @ (identity - kalman_gain @ H).T + kalman_gain @ R @ kalman_gain.T
        return self.x.copy()

    def step(self, measurement: np.ndarray, dt: float) -> np.ndarray:
        """Convenience method: predict then update."""
        self.predict(dt)
        return self.update(measurement)

    @staticmethod
    def _validate_vector(value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (KalmanFilter2D.state_dim,):
            raise ValueError(f"{name} must have shape {(KalmanFilter2D.state_dim,)}")
        return vector


ExtendedKalmanFilter = KalmanFilter2D
