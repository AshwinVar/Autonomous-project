"""Metrics for localization and planning demos."""

from __future__ import annotations

from collections import Counter
from typing import Iterable
import numpy as np


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Return root mean squared error between arrays."""
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have the same shape")
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def path_length_xy(path_xy: np.ndarray) -> float:
    """Return cumulative path length for ``[x, y]`` coordinates."""
    points = np.asarray(path_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("path_xy must have shape [N, 2]")
    deltas = np.diff(points, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def action_summary(actions: Iterable[str]) -> dict[str, int]:
    """Count planner action strings."""
    return dict(Counter(str(action) for action in actions))
