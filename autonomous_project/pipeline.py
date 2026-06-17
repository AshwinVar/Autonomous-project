"""End-to-end autonomy demo pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import numpy as np

from .data import SyntheticDrive, generate_synthetic_drive
from .metrics import action_summary, rmse
from .perception import LightweightPerceptionModel
from .planner import PlannerAction, SafetyAwarePlanner
from .sensor_fusion import KalmanFilter2D


@dataclass(frozen=True)
class PipelineResult:
    timestamps: np.ndarray
    ground_truth: np.ndarray
    measurements: np.ndarray
    estimates: np.ndarray
    risk_scores: np.ndarray
    actions: list[PlannerAction]
    localization_rmse_m: float

    @property
    def action_summary(self) -> dict[str, int]:
        return action_summary(action.value for action in self.actions)


def run_autonomy_pipeline(drive: SyntheticDrive | None = None, dt: float = 0.1) -> PipelineResult:
    """Run localization, perception-risk scoring and planning."""
    drive = drive or generate_synthetic_drive(dt=dt)
    filter_2d = KalmanFilter2D()
    perception = LightweightPerceptionModel()
    planner = SafetyAwarePlanner()

    estimates: list[np.ndarray] = []
    actions: list[PlannerAction] = []
    risk_scores = perception.obstacle_risk(drive.image_batch)

    filter_2d.reset(drive.measurements[0])
    for measurement, risk in zip(drive.measurements, risk_scores):
        estimate = filter_2d.step(measurement, dt=dt)
        action = planner.choose_action(estimate, float(risk))
        estimates.append(estimate)
        actions.append(action)

    estimates_array = np.vstack(estimates)
    error = rmse(estimates_array[:, :2], drive.ground_truth[:, :2])
    return PipelineResult(
        timestamps=drive.timestamps,
        ground_truth=drive.ground_truth,
        measurements=drive.measurements,
        estimates=estimates_array,
        risk_scores=risk_scores,
        actions=actions,
        localization_rmse_m=error,
    )


def write_trajectory_csv(result: PipelineResult, output_path: str | Path) -> Path:
    """Write trajectory and planning output to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["t", "gt_x", "gt_y", "meas_x", "meas_y", "est_x", "est_y", "risk_score", "action"])
        for idx, timestamp in enumerate(result.timestamps):
            writer.writerow([
                f"{timestamp:.3f}",
                f"{result.ground_truth[idx, 0]:.6f}",
                f"{result.ground_truth[idx, 1]:.6f}",
                f"{result.measurements[idx, 0]:.6f}",
                f"{result.measurements[idx, 1]:.6f}",
                f"{result.estimates[idx, 0]:.6f}",
                f"{result.estimates[idx, 1]:.6f}",
                f"{result.risk_scores[idx]:.6f}",
                result.actions[idx].value,
            ])
    return path
