"""Offline training loop for the planner."""
import os
import numpy as np
from autonomous_project.planner import DQNPlanner

def env_step(state, action):
    next_state = state + np.random.randn(*state.shape) * 0.1
    reward = -np.linalg.norm(next_state[:2])          # keep close to origin
    return next_state, reward, False

def main():
    planner = DQNPlanner()
    for ep in range(200):
        s = np.random.randn(4)
        for _ in range(30):
            a = planner.act(s)
            s_next, r, done = env_step(s, a)
            planner.remember(s, a, r, s_next, done)
            planner.train_step()
            s = s_next
        if (ep + 1) % 50 == 0:
            print(f"episode {ep+1}/200")

    os.makedirs("model_checkpoints", exist_ok=True)
    np.save("model_checkpoints/planner.npy",
            {"w1": planner.w1, "b1": planner.b1,
             "w2": planner.w2, "b2": planner.b2})
    print("weights saved to model_checkpoints/planner.npy")

if __name__ == "__main__":
    main()
