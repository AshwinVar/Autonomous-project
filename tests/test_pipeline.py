from pathlib import Path

from autonomous_project.data import generate_synthetic_drive
from autonomous_project.pipeline import run_autonomy_pipeline, write_trajectory_csv


def test_pipeline_returns_expected_shapes():
    drive = generate_synthetic_drive(samples=20, dt=0.1)
    result = run_autonomy_pipeline(drive=drive, dt=0.1)
    assert result.estimates.shape == (20, 4)
    assert len(result.actions) == 20
    assert result.localization_rmse_m >= 0.0


def test_write_trajectory_csv(tmp_path: Path):
    drive = generate_synthetic_drive(samples=5, dt=0.1)
    result = run_autonomy_pipeline(drive=drive, dt=0.1)
    path = write_trajectory_csv(result, tmp_path / "trajectory.csv")
    assert path.exists()
    assert "gt_x" in path.read_text(encoding="utf-8")
