from autonomous_project.sensor_fusion import ExtendedKalmanFilter
import numpy as np

def test_ekf_shapes():
    ekf = ExtendedKalmanFilter()
    ekf.predict(0.1)
    ekf.update(np.zeros(4))
    assert ekf.x.shape == (4,)
    assert ekf.P.shape == (4, 4)
