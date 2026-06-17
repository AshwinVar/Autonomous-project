"""KITTI Raw OXTS utilities.

The KITTI Raw OXTS packet format stores GPS/IMU values as one text file per
frame. This module parses the fields needed for a simple local trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np

EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class OxtsPacket:
    lat: float
    lon: float
    alt: float
    roll: float
    pitch: float
    yaw: float
    vf: float
    vl: float
    vu: float


@dataclass(frozen=True)
class KittiTrajectory:
    xy: np.ndarray
    yaw: np.ndarray
    velocity: np.ndarray
    source_files: list[str]


def parse_oxts_file(path: str | Path) -> OxtsPacket:
    """Parse one KITTI OXTS packet file."""
    values = [float(value) for value in Path(path).read_text(encoding="utf-8").split()]
    if len(values) < 11:
        raise ValueError(f"OXTS file has too few fields: {path}")
    return OxtsPacket(
        lat=values[0], lon=values[1], alt=values[2], roll=values[3], pitch=values[4], yaw=values[5],
        vf=values[8], vl=values[9], vu=values[10],
    )


def geodetic_to_local_xy(lat_lon_alt: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude values to local metric x/y coordinates."""
    data = np.asarray(lat_lon_alt, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("lat_lon_alt must have shape [N, >=2]")

    lat0 = math.radians(float(data[0, 0]))
    lon0 = math.radians(float(data[0, 1]))
    lat = np.radians(data[:, 0])
    lon = np.radians(data[:, 1])

    x = (lon - lon0) * math.cos(lat0) * EARTH_RADIUS_M
    y = (lat - lat0) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def load_oxts_trajectory(oxts_data_dir: str | Path) -> KittiTrajectory:
    """Load all OXTS packets from a KITTI ``oxts/data`` directory."""
    directory = Path(oxts_data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"OXTS data directory not found: {directory}")

    files = sorted(directory.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No OXTS .txt files found in: {directory}")

    packets = [parse_oxts_file(file_path) for file_path in files]
    lat_lon_alt = np.array([[p.lat, p.lon, p.alt] for p in packets], dtype=float)
    xy = geodetic_to_local_xy(lat_lon_alt)
    yaw = np.array([p.yaw for p in packets], dtype=float)
    velocity = np.array([[p.vf, p.vl, p.vu] for p in packets], dtype=float)
    return KittiTrajectory(xy=xy, yaw=yaw, velocity=velocity, source_files=[str(file_path) for file_path in files])


def prepare_kitti_sequence(raw_root: str | Path, date: str, drive: str, out_dir: str | Path) -> tuple[Path, Path]:
    """Convert a KITTI Raw OXTS sequence into a compact trajectory artifact."""
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    oxts_data = raw_root / date / drive / "oxts" / "data"
    trajectory = load_oxts_trajectory(oxts_data)

    out_dir.mkdir(parents=True, exist_ok=True)
    sequence_name = drive.replace("/", "_")
    npz_path = out_dir / f"{sequence_name}_trajectory.npz"
    manifest_path = out_dir / "manifest.json"

    np.savez_compressed(
        npz_path,
        xy=trajectory.xy,
        yaw=trajectory.yaw,
        velocity=trajectory.velocity,
        source_files=np.array(trajectory.source_files, dtype=object),
    )

    manifest = {
        "raw_root": str(raw_root),
        "date": date,
        "drive": drive,
        "frames": int(trajectory.xy.shape[0]),
        "trajectory_file": str(npz_path),
        "fields": ["xy", "yaw", "velocity", "source_files"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return npz_path, manifest_path
