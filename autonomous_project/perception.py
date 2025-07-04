"""Very small CNN-like network in pure NumPy."""
import numpy as np

class DummyCNN:
    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        self.w = np.random.randn(input_channels, 3, 3) * 0.01
        self.fc = np.random.randn(num_classes, input_channels) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_conv = self._conv2d(x, self.w)
        x_pool = x_conv.mean(axis=(2, 3))          # global average pool
        logits = self.fc @ x_pool
        return logits

    @staticmethod
    def _conv2d(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        k = w.shape[-1]
        B, C, H, W = x.shape
        out_h = H - k + 1
        out_w = W - k + 1
        out = np.empty((B, C, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, :, i:i+k, j:j+k]
                out[:, :, i, j] = (patch * w).sum(axis=(2, 3))
        return np.tanh(out)
