# Architecture Notes

## Localization

The localization baseline uses a constant-velocity Kalman Filter over state:

```text
x = [position_x, position_y, velocity_x, velocity_y]
```

The prediction step propagates position using velocity and elapsed time. The update step fuses a noisy measurement vector with the predicted state.

This gives a clear baseline for discussing:

- motion models
- process noise
- measurement noise
- covariance growth during prediction
- covariance reduction after measurement updates

## Perception

The perception model is intentionally lightweight and dependency-minimal. It provides deterministic tensor operations and a risk score that can be consumed by the planner. This keeps the repo fast to run in CI while leaving room for future upgrades to PyTorch, TensorFlow, YOLO, semantic segmentation or camera/LiDAR fusion.

## Planning

The planner is rule-based and safety-aware. It is deliberately interpretable:

- stop when obstacle risk is high
- slow down for medium risk or excess speed
- keep lane otherwise

This is more credible for a small portfolio repo than claiming a full DQN policy without a real environment, reward design and training evidence.

## KITTI Integration

The KITTI pipeline parses OXTS packets and converts latitude/longitude into a local metric frame. It outputs a compact `.npz` trajectory file that can be used by later localization experiments.
