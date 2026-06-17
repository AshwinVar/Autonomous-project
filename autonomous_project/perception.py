"""Lightweight perception-risk scoring.

The model is deliberately dependency-minimal. It does not claim to be a trained
production CNN; it provides deterministic image/tensor feature extraction and a
risk score that can feed the planner during demos and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PerceptionConfig:
    input_channels: int = 3
    kernel_size: int = 3
    num_classes: int = 3
    seed: int = 7


class LightweightPerceptionModel:
    """Small NumPy perception model producing logits and obstacle risk."""

    def __init__(self, config: PerceptionConfig | None = None):
        self.config = config or PerceptionConfig()
        rng = np.random.default_rng(self.config.seed)
        self.kernel = rng.normal(0.0, 0.03, size=(self.config.input_channels, self.config.kernel_size, self.config.kernel_size))
        self.classifier = rng.normal(0.0, 0.05, size=(self.config.num_classes, self.config.input_channels))

    def forward(self, image_batch: np.ndarray) -> np.ndarray:
        """Return class logits for a batch of images shaped ``[B, C, H, W]``."""
        x = self._validate_batch(image_batch)
        features = self._conv2d(x, self.kernel)
        pooled = features.mean(axis=(2, 3))
        return pooled @ self.classifier.T

    def obstacle_risk(self, image_batch: np.ndarray) -> np.ndarray:
        """Return a normalized risk score between 0 and 1 for each image.

        The score combines model logits with central-image intensity. The central
        term gives the demo deterministic high-risk frames when the synthetic
        generator injects an obstacle-like patch.
        """
        x = self._validate_batch(image_batch)
        logits = self.forward(x)
        probabilities = self._softmax(logits)
        model_risk = probabilities[:, -1]

        _, _, height, width = x.shape
        row_start = max(height // 2 - 2, 0)
        row_end = min(height // 2 + 2, height)
        col_start = max(width // 2 - 2, 0)
        col_end = min(width // 2 + 2, width)
        central_intensity = x[:, :, row_start:row_end, col_start:col_end].mean(axis=(1, 2, 3))

        risk = 0.25 * model_risk + 1.4 * (central_intensity - 0.50) + 0.25
        return np.clip(risk, 0.0, 1.0)

    @staticmethod
    def _conv2d(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        k = kernel.shape[-1]
        batch, channels, height, width = x.shape
        out_h = height - k + 1
        out_w = width - k + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("input image is smaller than the convolution kernel")
        out = np.empty((batch, channels, out_h, out_w), dtype=float)
        for row in range(out_h):
            for col in range(out_w):
                patch = x[:, :, row : row + k, col : col + k]
                out[:, :, row, col] = (patch * kernel).sum(axis=(2, 3))
        return np.tanh(out)

    def _validate_batch(self, image_batch: np.ndarray) -> np.ndarray:
        x = np.asarray(image_batch, dtype=float)
        if x.ndim != 4:
            raise ValueError("image_batch must have shape [B, C, H, W]")
        if x.shape[1] != self.config.input_channels:
            raise ValueError(f"expected {self.config.input_channels} channels")
        return x

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
