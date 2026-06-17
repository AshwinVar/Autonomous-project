from pathlib import Path
import numpy as np

from autonomous_project.kitti import geodetic_to_local_xy, load_oxts_trajectory, prepare_kitti_sequence


def _write_oxts(path: Path, lat: float, lon: float):
    values = [lat, lon, 100.0, 0.0, 0.0, 0.1, 0.0, 0.0, 5.0, 0.1, 0.0, 0, 0, 0]
    path.write_text(" ".join(str(v) for v in values), encoding="utf-8")


def test_geodetic_to_local_xy_origin_is_zero():
    points = np.array([[52.0, 13.0, 0.0], [52.0001, 13.0001, 0.0]])
    xy = geodetic_to_local_xy(points)
    assert xy.shape == (2, 2)
    assert np.allclose(xy[0], [0.0, 0.0])
    assert np.linalg.norm(xy[1]) > 0.0


def test_prepare_kitti_sequence(tmp_path: Path):
    raw_root = tmp_path / "raw_dataset"
    data_dir = raw_root / "2011_09_26" / "2011_09_26_drive_0001_sync" / "oxts" / "data"
    data_dir.mkdir(parents=True)
    _write_oxts(data_dir / "0000000000.txt", 52.0, 13.0)
    _write_oxts(data_dir / "0000000001.txt", 52.0001, 13.0001)

    trajectory = load_oxts_trajectory(data_dir)
    assert trajectory.xy.shape == (2, 2)

    npz_path, manifest_path = prepare_kitti_sequence(
        raw_root=raw_root,
        date="2011_09_26",
        drive="2011_09_26_drive_0001_sync",
        out_dir=tmp_path / "processed",
    )
    assert npz_path.exists()
    assert manifest_path.exists()
