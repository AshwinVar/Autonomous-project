"""Run localization on a processed KITTI trajectory artifact.

Usage:
    python scripts/run_kitti_demo.py --trajectory processed_kitti/2011_09_26_drive_0001_sync_trajectory.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import numpy as np

from autonomous_project.metrics import rmse
from autonomous_project.sensor_fusion import KalmanFilter2D
from autonomous_project.visualization import save_trajectory_plot
from autonomous_project.pipeline import PipelineResult
from autonomous_project.planner import PlannerAction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kalman localization on processed KITTI OXTS trajectory data.")
    parser.add_argument("--trajectory", required=True, help="Path to processed KITTI .npz trajectory file.")
    parser.add_argument("--out", default="outputs/kitti_localization.csv", help="Output CSV path.")
    parser.add_argument("--plot", default="outputs/kitti_localization.png", help="Output plot path.")
    parser.add_argument("--dt", type=float, default=0.1, help="Assumed frame time step in seconds. KITTI synced raw is commonly 10 Hz.")
    parser.add_argument("--measurement-std", type=float, default=0.75, help="Synthetic measurement noise added for filter demonstration.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for measurement noise.")
    return parser.parse_args()


def load_kitti_state(npz_path: str | Path, dt: float) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    xy = np.asarray(data["xy"], dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("trajectory file must contain xy with shape [N, 2]")

    if "velocity" in data.files:
        velocity = np.asarray(data["velocity"], dtype=float)
        if velocity.ndim == 2 and velocity.shape[1] >= 2 and velocity.shape[0] == xy.shape[0]:
            vx_vy = velocity[:, :2]
        else:
            vx_vy = np.gradient(xy, dt, axis=0)
    else:
        vx_vy = np.gradient(xy, dt, axis=0)

    return np.column_stack([xy, vx_vy])


def run_filter(reference_state: np.ndarray, dt: float, measurement_std: float, seed: int):
    rng = np.random.default_rng(seed)
    measurements = reference_state.copy()
    measurements[:, :2] += rng.normal(0.0, measurement_std, size=measurements[:, :2].shape)
    measurements[:, 2:4] += rng.normal(0.0, measurement_std * 0.2, size=measurements[:, 2:4].shape)

    kf = KalmanFilter2D()
    kf.reset(measurements[0])
    estimates = []
    for measurement in measurements:
        estimates.append(kf.step(measurement, dt=dt))
    return measurements, np.vstack(estimates)


def write_csv(path: str | Path, reference_state: np.ndarray, measurements: np.ndarray, estimates: np.ndarray) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "ref_x", "ref_y", "meas_x", "meas_y", "est_x", "est_y"])
        for i in range(reference_state.shape[0]):
            writer.writerow([
                i,
                f"{reference_state[i, 0]:.6f}",
                f"{reference_state[i, 1]:.6f}",
                f"{measurements[i, 0]:.6f}",
                f"{measurements[i, 1]:.6f}",
                f"{estimates[i, 0]:.6f}",
                f"{estimates[i, 1]:.6f}",
            ])
    return out_path


def main() -> None:
    args = parse_args()
    reference_state = load_kitti_state(args.trajectory, dt=args.dt)
    measurements, estimates = run_filter(reference_state, dt=args.dt, measurement_std=args.measurement_std, seed=args.seed)
    error = rmse(estimates[:, :2], reference_state[:, :2])
    csv_path = write_csv(args.out, reference_state, measurements, estimates)

    plot_path = None
    try:
        result = PipelineResult(
            timestamps=np.arange(reference_state.shape[0]) * args.dt,
            ground_truth=reference_state,
            measurements=measurements,
            estimates=estimates,
            risk_scores=np.zeros(reference_state.shape[0]),
            actions=[PlannerAction.KEEP_LANE for _ in range(reference_state.shape[0])],
            localization_rmse_m=error,
        )
        plot_path = save_trajectory_plot(result, args.plot)
    except RuntimeError as exc:
        print(f"Plot skipped: {exc}")

    print("KITTI localization demo complete")
    print(f"Frames: {reference_state.shape[0]}")
    print(f"Localization RMSE: {error:.3f} m")
    print(f"Wrote: {csv_path}")
    if plot_path is not None:
        print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
