# Autonomous-Project 🚗🤖  
Sensor-fusion & decision-making for autonomous vehicles  
[![CI](https://img.shields.io/github/actions/workflow/status/AshwinVar/Autonomous-project/ci.yml?branch=main)](./.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Quick start:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/demo.py
```
## Dataset
This project is evaluated on the **KITTI Raw** sequences from Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago.

> **Download** – Visit the official site: <http://www.cvlibs.net/datasets/kitti/>  
> Select **Raw data synchronous** and grab the date sequences '2011_09_26'.

Expected folder layout:
raw_dataset/
└── 2011_09_26/
├── image_00/
├── velodyne_points/
├── oxts/
└── calib.txt


Run `python scripts/prepare_kitti.py` (to-do) to convert the raw logs into the NumPy/TFRecord format used by the training pipeline.  
*Note: The dataset is **not** included in this repository to respect the original licence.*


