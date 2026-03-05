from __future__ import annotations

"""Monocular visual odometry core for Task2.

Given a frame sequence and camera intrinsics, this module estimates camera
trajectory by:
1) ORB detection/description on adjacent frames,
2) descriptor matching + ratio test,
3) Essential matrix estimation with RANSAC,
4) relative pose recovery,
5) global pose accumulation.

Because pure monocular VO has unknown global scale, this implementation uses
adjacent GT translation magnitude when GT is available. That keeps focus on
relative motion estimation quality while avoiding scale ambiguity.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
import pandas as pd


def _orthonormalize_rotation(r: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3) via SVD."""
    u, _, vt = np.linalg.svd(r)
    r_ortho = u @ vt
    if np.linalg.det(r_ortho) < 0:
        u[:, -1] *= -1.0
        r_ortho = u @ vt
    return r_ortho


def _write_dict_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write dict rows to a CSV with explicit headers."""
    with open(path, 'w', encoding = 'utf-8', newline ='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_mean(values: list[float]) -> float:
    """Return mean or NaN if list is empty."""
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _strip_row_keys(row: dict[str, str]) -> dict[str, str]:
    """Normalize CSV keys by trimming leading/trailing spaces."""
    return {str(k):v for k,v in row.items()}


def _get_first_key(row: dict[str, str], candidates: list[str]) -> str | None:
    """Return first non-empty value from candidate keys."""
    for key in candidates:
        if key in row:
            value = row[key]
            if value is not None and str(value).strip() != "":
                return str(value).strip()
    return None

def _read_intrinsics(path: Path) -> np.ndarray:
    """Load a 3x3 camera intrinsic matrix from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Intrinsics file {path} does not contain a YAML mapping")

    if "intrinsics" in data:
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64).reshape(-1)
        if intrinsics.shape != (4,):
            raise ValueError(f"Invalid intrinsics shape in {path}: {intrinsics.shape}, expected (4,)")
        fu, fv, cu, cv = intrinsics.tolist()
        return np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)

    required = ("fx", "fy", "cx", "cy")
    if all(key in data for key in required):
        return np.array(
            [[float(data["fx"]), 0.0, float(data["cx"])], [0.0, float(data["fy"]), float(data["cy"])], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    raise ValueError(f"Intrinsics file {path} must contain intrinsics[fu,fv,cu,cv], or fx/fy/cx/cy")

# csv read helpers that support flexible key names, used for GT pose loading and output writing
def _read_distortion_coeffs(path: Path) -> np.ndarray | None:
    """Load camera distortion coefficients from YAML when available."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return None

    dist_raw = None
    if "distortion_coefficients" in data:
        dist_raw = data["distortion_coefficients"]

    if dist_raw is None:
        return None

    dist = np.asarray(dist_raw, dtype=np.float64).reshape(-1, 1)

    if dist.size == 0:
        return None
    return dist


def _read_gt_poses(path:Path) -> dict[int,np.ndarray]:
    '''Load GT poses from CSV, supporting flexible key names. 
        Returns dict of frame->(x,y,z).'''
    with open(path, 'r', encoding = 'utf-8') as f:
        reader = csv.DictReader(f)
        gt_poses = {}
        for row in reader:
            row = _strip_row_keys(row)
            frame_str = _get_first_key(row, ['#timestamp', 'timestamp', 'frame'])
            gt_poses_x = _get_first_key(row, ['p_RS_R_x [m]', 'p_RS_R_x'])
            gt_poses_y = _get_first_key(row, ['p_RS_R_y [m]', 'p_RS_R_y'])
            gt_poses_z = _get_first_key(row, ['p_RS_R_z [m]', 'p_RS_R_z'])
            if frame_str is None or gt_poses_x is None or gt_poses_y is None or gt_poses_z is None:
                continue
            gt_poses[int(frame_str)] = np.array([float(gt_poses_x), float(gt_poses_y), float(gt_poses_z)], dtype=np.float64)
    return gt_poses