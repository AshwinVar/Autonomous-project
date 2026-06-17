"""Synthetic driving data generation for reproducible demos and tests."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SyntheticDrive:
    timestamps: np.ndarray
    ground_truth: np.ndarray
    measurements: np.ndarray
    image_batch: np.ndarray


def generate_synthetic_drive(
    samples: int = 120,
    dt: float = 0.1,
    measurement_std: float = 0.35,
    seed: int = 42,
) -> SyntheticDrive:
    """Generate a deterministic 2D driving trajectory with noisy measurements."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if dt <= 0:
        raise ValueError("dt must be positive")

    rng = np.random.default_rng(seed)
    timestamps = np.arange(samples, dtype=float) * dt

    vx = 7.5 + 0.8 * np.sin(0.35 * timestamps)
    vy = 0.6 * np.cos(0.25 * timestamps)
    x = np.cumsum(vx * dt)
    y = np.cumsum(vy * dt)
    ground_truth = np.column_stack([x, y, vx, vy])

    noise = rng.normal(0.0, measurement_std, size=ground_truth.shape)
    noise[:, 2:4] *= 0.35
    measurements = ground_truth + noise

    image_batch = rng.normal(0.5, 0.15, size=(samples, 3, 16, 16)).clip(0.0, 1.0)
    image_batch[int(samples * 0.70) :, :, 6:10, 6:10] += 0.35
    image_batch = image_batch.clip(0.0, 1.0)

    return SyntheticDrive(
        timestamps=timestamps,
        ground_truth=ground_truth,
        measurements=measurements,
        image_batch=image_batch,
    )
