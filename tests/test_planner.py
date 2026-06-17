import numpy as np
import pytest

from autonomous_project.planner import PlannerAction, SafetyAwarePlanner


def test_keep_lane_when_risk_and_speed_are_low():
    planner = SafetyAwarePlanner()
    state = np.array([0.0, 0.0, 5.0, 0.0])
    assert planner.choose_action(state, obstacle_risk=0.1) == PlannerAction.KEEP_LANE


def test_stop_when_risk_is_high():
    planner = SafetyAwarePlanner()
    state = np.array([0.0, 0.0, 5.0, 0.0])
    assert planner.choose_action(state, obstacle_risk=0.95) == PlannerAction.STOP


def test_slow_down_when_speed_is_high():
    planner = SafetyAwarePlanner()
    state = np.array([0.0, 0.0, 30.0, 0.0])
    assert planner.choose_action(state, obstacle_risk=0.1) == PlannerAction.SLOW_DOWN


def test_invalid_risk_raises():
    planner = SafetyAwarePlanner()
    with pytest.raises(ValueError):
        planner.choose_action(np.zeros(4), obstacle_risk=1.5)
