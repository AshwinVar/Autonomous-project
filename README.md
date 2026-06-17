# Autonomous Systems Demo: Localization, Perception & Safety-Aware Planning

Autonomous systems project showing a complete, testable pipeline for:

- **Sensor fusion / localization** using a Kalman Filter over 2D vehicle state.
- **Lightweight perception scoring** using a NumPy CNN-style feature extractor.
- **Safety-aware planning** using interpretable action selection.
- **KITTI Raw data preparation** for OXTS/GPS trajectory conversion.
- **CI-ready testing** with pytest and GitHub Actions.

This repository is designed as a portfolio project for robotics, autonomous systems, ML, perception and vehicle-software roles.

---

## 1. Why this project matters

This repo demonstrates the engineering layer around autonomy code:

- deterministic simulation
- clean package structure
- reproducible command-line scripts
- test coverage
- real-world dataset preparation path
- metrics and visualization outputs
- documented architecture and assumptions

The default demo runs without external datasets.

## 2. Architecture

```text
Synthetic trajectory / KITTI OXTS
        |
        v
Noisy GPS-like position + velocity measurements
        |
        v
KalmanFilter2D localization
        |
        +--> trajectory RMSE metrics
        |
        v
LightweightPerceptionModel obstacle-risk score
        |
        v
SafetyAwarePlanner action decision
        |
        v
CSV + optional trajectory plot
```

Core modules:

| Module | Purpose |
|---|---|
| `autonomous_project.sensor_fusion` | 2D constant-velocity Kalman Filter for state estimation. |
| `autonomous_project.perception` | Lightweight NumPy perception model for feature/logit generation. |
| `autonomous_project.planner` | Safety-aware planner with interpretable actions. |
| `autonomous_project.data` | Synthetic driving scenario generation. |
| `autonomous_project.pipeline` | End-to-end localization and planning pipeline. |
| `autonomous_project.kitti` | KITTI Raw OXTS indexing and trajectory conversion utilities. |
| `autonomous_project.metrics` | RMSE, path length and action summary metrics. |
| `autonomous_project.visualization` | Optional matplotlib trajectory plots. |

---

## 3. Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
python scripts/demo.py
pytest
```

Expected demo output:

```text
Autonomous systems demo complete
Samples: 120
Localization RMSE: ... m
Action summary: {'KEEP_LANE': ..., 'SLOW_DOWN': ..., 'STOP': ...}
Wrote: outputs/trajectory.csv
Wrote: outputs/trajectory.png
```

The demo uses a synthetic driving trajectory, noisy measurements and a Kalman Filter estimate. It writes outputs to the `outputs/` folder.

---

## 4. Run the KITTI preparation pipeline

KITTI data is not included in this repository.

Download KITTI Raw data from the official KITTI website and place a sequence like this:

```text
raw_dataset/
└── 2011_09_26/
    └── 2011_09_26_drive_0001_sync/
        └── oxts/
            └── data/
                ├── 0000000000.txt
                ├── 0000000001.txt
                └── ...
```

Then run:

```bash
python scripts/prepare_kitti.py \
  --raw-root raw_dataset \
  --date 2011_09_26 \
  --drive 2011_09_26_drive_0001_sync \
  --out processed_kitti
```

Expected output:

```text
Prepared KITTI trajectory
Wrote: processed_kitti/2011_09_26_drive_0001_sync_trajectory.npz
Wrote: processed_kitti/manifest.json
```

The generated `.npz` contains local `x/y` positions and available yaw/velocity metadata parsed from KITTI OXTS packets.

---

## 5. Run localization on processed KITTI data

After preparing the KITTI trajectory, run:

```bash
python scripts/run_kitti_demo.py \
  --trajectory processed_kitti/2011_09_26_drive_0001_sync_trajectory.npz \
  --out outputs/kitti_localization.csv \
  --plot outputs/kitti_localization.png
```

Expected output:

```text
KITTI localization demo complete
Frames: ...
Localization RMSE: ... m
Wrote: outputs/kitti_localization.csv
Wrote: outputs/kitti_localization.png
```

This runner loads the processed OXTS trajectory, creates noisy GPS-like measurements, runs the Kalman Filter and exports a CSV plus plot.

---

## 6. Technical scope

This is a compact autonomy engineering demo, not a full production self-driving stack.

Implemented:

- 2D vehicle state estimation: `[x, y, vx, vy]`
- predict/update localization cycle
- synthetic measurement generation
- perception-like risk scoring
- safety-aware rule-based planning
- KITTI OXTS pose extraction
- processed KITTI localization runner
- automated tests
- CI workflow

Not included:

- full ROS2 node graph
- deep-learning training on KITTI images
- LiDAR point cloud segmentation
- production-grade MPC
- HD map routing

Those would be natural next upgrades.

---

## 7. Results produced by the demo

After running `python scripts/demo.py`, the project generates:

```text
outputs/trajectory.csv
outputs/trajectory.png
```

After running `python scripts/run_kitti_demo.py`, the project generates:

```text
outputs/kitti_localization.csv
outputs/kitti_localization.png
```

The CSV includes reference position, noisy measurement and Kalman estimate. The plot compares reference trajectory, noisy measurements and estimated trajectory.

---

## 8. Repository commands

```bash
# Run local synthetic demo
python scripts/demo.py

# Prepare KITTI OXTS trajectory
python scripts/prepare_kitti.py --raw-root raw_dataset --date 2011_09_26 --drive 2011_09_26_drive_0001_sync --out processed_kitti

# Run KITTI localization demo
python scripts/run_kitti_demo.py --trajectory processed_kitti/2011_09_26_drive_0001_sync_trajectory.npz

# Run tests
pytest

# Run only localization tests
pytest tests/test_sensor_fusion.py

# Install package locally
pip install -e .
```
