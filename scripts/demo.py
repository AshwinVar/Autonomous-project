"""Small EKF + planner demo."""
import time
import numpy as np
from autonomous_project.sensor_fusion import ExtendedKalmanFilter
from autonomous_project.planner import DQNPlanner

def main():
    ekf = ExtendedKalmanFilter()
    planner = DQNPlanner()
    dt = 0.1
    for t in range(50):
        ekf.predict(dt)
        z = ekf.x + np.random.randn(4) * 0.05        # fake noisy measurement
        ekf.update(z)
        action = planner.act(ekf.x, eps=0.2)
        print(f"t={t*dt:4.1f}s  pos=({ekf.x[0]:+.2f}, {ekf.x[1]:+.2f})  action={action}")
        time.sleep(0.05)

if __name__ == "__main__":
    main()
