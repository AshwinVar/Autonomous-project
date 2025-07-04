"""Evaluate a saved planner checkpoint."""
import sys, os
import numpy as np
from autonomous_project.planner import DQNPlanner

ckpt = "model_checkpoints/planner.npy"
if not os.path.exists(ckpt):
    sys.exit("run scripts/train.py first")

weights = np.load(ckpt, allow_pickle=True).item()
planner = DQNPlanner()
planner.w1, planner.b1 = weights["w1"], weights["b1"]
planner.w2, planner.b2 = weights["w2"], weights["b2"]

s = np.zeros(4)
total_r = 0.0
for _ in range(100):
    a = planner.act(s, eps=0.0)
    s, r, _ = (s + np.random.randn(4)*0.05,
               -np.linalg.norm(s[:2]),
               False)
    total_r += r
print(f"avg reward over 100 steps: {total_r/100:+.3f}")
