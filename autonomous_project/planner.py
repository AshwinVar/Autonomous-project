"""Minimal DQN-style planner (NumPy)."""
import numpy as np
from collections import deque
import random

class DQNPlanner:
    def __init__(self, state_dim: int = 4, action_dim: int = 3,
                 hidden_dim: int = 64, lr: float = 1e-3, gamma: float = 0.95,
                 buffer_size: int = 10_000, batch_size: int = 64):
        self.w1 = np.random.randn(hidden_dim, state_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(action_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(action_dim)
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.action_dim = action_dim

    def act(self, state: np.ndarray, eps: float = 0.1) -> int:
        if random.random() < eps:
            return random.randrange(self.action_dim)
        q = self._forward(state)
        return int(np.argmax(q))

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for s, a, r, s_next, done in batch:
            target = r
            if not done:
                target += self.gamma * np.max(self._forward(s_next))
            q = self._forward(s)
            td = target - q[a]
            grad_out = np.zeros_like(q)
            grad_out[a] = td
            self._backward(s, grad_out)

    # ---- internals ---------------------------------------------------------
    def _forward(self, s: np.ndarray) -> np.ndarray:
        z1 = self.w1 @ s + self.b1
        h1 = np.tanh(z1)
        z2 = self.w2 @ h1 + self.b2
        return z2

    def _backward(self, s: np.ndarray, grad_out: np.ndarray):
        z1 = self.w1 @ s + self.b1
        h1 = np.tanh(z1)
        grad_w2 = np.outer(grad_out, h1)
        grad_b2 = grad_out
        grad_h1 = self.w2.T @ grad_out
        grad_z1 = grad_h1 * (1 - np.tanh(z1) ** 2)
        grad_w1 = np.outer(grad_z1, s)
        grad_b1 = grad_z1
        self.w2 += self.lr * grad_w2
        self.b2 += self.lr * grad_b2
        self.w1 += self.lr * grad_w1
        self.b1 += self.lr * grad_b1
