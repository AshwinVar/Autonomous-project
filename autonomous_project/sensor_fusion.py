"""Extended Kalman Filter for constant-velocity motion in 2-D."""
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, process_var: float = 0.05, meas_var: float = 0.01):
        # state: [x, y, vx, vy]
        self.x = np.zeros(4)
        self.P = np.eye(4) * 0.1
        self.Q = np.eye(4) * process_var
        self.R = np.eye(4) * meas_var

    def predict(self, dt: float) -> None:
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        H = np.eye(4)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
