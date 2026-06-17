import numpy as np
import pytest

from autonomous_project.sensor_fusion import KalmanFilter2D


def test_transition_matrix_constant_velocity():
    F = KalmanFilter2D.transition_matrix(0.5)
    assert F.shape == (4, 4)
    assert F[0, 2] == 0.5
    assert F[1, 3] == 0.5


def test_predict_moves_position_using_velocity():
    kf = KalmanFilter2D()
    kf.reset(np.array([0.0, 0.0, 2.0, -1.0]))
    predicted = kf.predict(dt=2.0)
    assert np.allclose(predicted[:2], [4.0, -2.0])


def test_update_reduces_large_measurement_error():
    kf = KalmanFilter2D()
    kf.reset(np.array([0.0, 0.0, 0.0, 0.0]))
    kf.predict(dt=1.0)
    updated = kf.update(np.array([10.0, 0.0, 0.0, 0.0]))
    assert 0.0 < updated[0] < 10.0


def test_invalid_dt_raises():
    with pytest.raises(ValueError):
        KalmanFilter2D.transition_matrix(0.0)
