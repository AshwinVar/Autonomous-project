"""Run the complete local autonomy demo."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autonomous_project.pipeline import run_autonomy_pipeline, write_trajectory_csv
from autonomous_project.visualization import save_trajectory_plot


def main() -> None:
    output_dir = Path("outputs")
    result = run_autonomy_pipeline()
    csv_path = write_trajectory_csv(result, output_dir / "trajectory.csv")

    plot_path = None
    try:
        plot_path = save_trajectory_plot(result, output_dir / "trajectory.png")
    except RuntimeError as exc:
        print(f"Plot skipped: {exc}")

    print("Autonomous systems demo complete")
    print(f"Samples: {len(result.timestamps)}")
    print(f"Localization RMSE: {result.localization_rmse_m:.3f} m")
    print(f"Action summary: {result.action_summary}")
    print(f"Wrote: {csv_path}")
    if plot_path is not None:
        print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
