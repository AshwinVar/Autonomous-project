import numpy as np
import pytest

from autonomous_project.perception import LightweightPerceptionModel


def test_forward_shape():
    model = LightweightPerceptionModel()
    batch = np.ones((5, 3, 16, 16), dtype=float)
    logits = model.forward(batch)
    assert logits.shape == (5, 3)


def test_obstacle_risk_range():
    model = LightweightPerceptionModel()
    batch = np.ones((4, 3, 16, 16), dtype=float)
    risk = model.obstacle_risk(batch)
    assert risk.shape == (4,)
    assert np.all(risk >= 0.0)
    assert np.all(risk <= 1.0)


def test_wrong_channel_count_raises():
    model = LightweightPerceptionModel()
    with pytest.raises(ValueError):
        model.forward(np.ones((1, 1, 16, 16)))
