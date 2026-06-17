"""Prepare KITTI Raw OXTS trajectory data."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

from autonomous_project.kitti import prepare_kitti_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert KITTI Raw OXTS packets into trajectory arrays.")
    parser.add_argument("--raw-root", default="raw_dataset", help="Root directory containing KITTI Raw downloads.")
    parser.add_argument("--date", required=True, help="KITTI date folder, e.g. 2011_09_26.")
    parser.add_argument("--drive", required=True, help="Drive folder, e.g. 2011_09_26_drive_0001_sync.")
    parser.add_argument("--out", default="processed_kitti", help="Output directory for processed trajectory files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path, manifest_path = prepare_kitti_sequence(
        raw_root=args.raw_root,
        date=args.date,
        drive=args.drive,
        out_dir=args.out,
    )
    print("Prepared KITTI trajectory")
    print(f"Wrote: {npz_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
