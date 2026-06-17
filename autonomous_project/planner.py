"""Safety-aware planning primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class PlannerAction(str, Enum):
    KEEP_LANE = "KEEP_LANE"
    SLOW_DOWN = "SLOW_DOWN"
    STOP = "STOP"


@dataclass(frozen=True)
class PlannerConfig:
    max_speed_mps: float = 13.5
    medium_risk_threshold: float = 0.45
    high_risk_threshold: float = 0.70


class SafetyAwarePlanner:
    """Interpretable planner for risk-aware driving decisions.

    This is intentionally rule-based. For a compact portfolio repo, a tested and
    interpretable baseline is more credible than claiming deep RL without a real
    environment, reward function and training evidence.
    """

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()

    def choose_action(self, state: np.ndarray, obstacle_risk: float) -> PlannerAction:
        """Choose an action from vehicle state and risk estimate."""
        vehicle_state = np.asarray(state, dtype=float)
        if vehicle_state.shape != (4,):
            raise ValueError("state must have shape [x, y, vx, vy]")
        if not 0.0 <= obstacle_risk <= 1.0:
            raise ValueError("obstacle_risk must be between 0 and 1")

        speed = float(np.linalg.norm(vehicle_state[2:4]))
        if obstacle_risk >= self.config.high_risk_threshold:
            return PlannerAction.STOP
        if obstacle_risk >= self.config.medium_risk_threshold or speed > self.config.max_speed_mps:
            return PlannerAction.SLOW_DOWN
        return PlannerAction.KEEP_LANE

    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        """Return an integer action id for old demos.

        0 = KEEP_LANE, 1 = SLOW_DOWN, 2 = STOP.
        ``eps`` is accepted for backward compatibility and ignored.
        """
        del eps
        action = self.choose_action(state, obstacle_risk=0.25)
        return list(PlannerAction).index(action)


DQNPlanner = SafetyAwarePlanner
