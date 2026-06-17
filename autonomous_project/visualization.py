"""Visualization helpers for trajectory outputs."""

from __future__ import annotations

from pathlib import Path
from .pipeline import PipelineResult


def save_trajectory_plot(result: PipelineResult, output_path: str | Path) -> Path:
    """Save a trajectory comparison plot.

    Matplotlib is imported lazily so the core package remains lightweight.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to save trajectory plots: pip install matplotlib") from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(result.ground_truth[:, 0], result.ground_truth[:, 1], label="ground truth")
    plt.scatter(result.measurements[:, 0], result.measurements[:, 1], s=8, alpha=0.35, label="noisy measurement")
    plt.plot(result.estimates[:, 0], result.estimates[:, 1], label="kalman estimate")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Autonomous Systems Localization Demo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path
