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
    with open(path, "w", encoding="utf-8", newline="") as f:
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
    """Normalize CSV keys/values by trimming leading/trailing spaces."""
    out: dict[str, str] = {}
    for k, v in row.items():
        kk = str(k).strip()
        if v is None:
            out[kk] = v
        else:
            out[kk] = str(v).strip()
    return out


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
            [
                [float(data["fx"]), 0.0, float(data["cx"])],
                [0.0, float(data["fy"]), float(data["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    raise ValueError(f"Intrinsics file {path} must contain intrinsics[fu,fv,cu,cv], or fx/fy/cx/cy")


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


def _read_gt_poses(path: Path) -> dict[int, np.ndarray]:
    """Load GT poses from CSV, supporting flexible key names.

    Returns:
        Dict mapping frame index -> np.ndarray shape (3,) for (x,y,z) in meters.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        gt_poses: dict[int, np.ndarray] = {}
        for row in reader:
            row = _strip_row_keys(row)
            frame_str = _get_first_key(row, ["#timestamp", "timestamp", "frame"])
            gt_poses_x = _get_first_key(row, ["p_RS_R_x [m]", "p_RS_R_x"])
            gt_poses_y = _get_first_key(row, ["p_RS_R_y [m]", "p_RS_R_y"])
            gt_poses_z = _get_first_key(row, ["p_RS_R_z [m]", "p_RS_R_z"])
            if frame_str is None or gt_poses_x is None or gt_poses_y is None or gt_poses_z is None:
                continue
            gt_poses[int(frame_str)] = np.array(
                [float(gt_poses_x), float(gt_poses_y), float(gt_poses_z)], dtype=np.float64
            )
    return gt_poses


def _read_gt_velocities(path: Path) -> dict[int, np.ndarray]:
    """Load GT velocities from CSV, supporting flexible key names.

    Returns:
        Dict mapping frame index -> np.ndarray shape (3,) for (vx,vy,vz) in m/s.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        gt_velocities: dict[int, np.ndarray] = {}
        for row in reader:
            row = _strip_row_keys(row)
            frame_str = _get_first_key(row, ["#timestamp", "timestamp", "frame"])
            gt_velocities_x = _get_first_key(row, ["v_RS_R_x [m s^-1]", "v_RS_R_x", " v_RS_R_x [m s^-1]"])
            gt_velocities_y = _get_first_key(row, ["v_RS_R_y [m s^-1]", "v_RS_R_y", " v_RS_R_y [m s^-1]"])
            gt_velocities_z = _get_first_key(row, ["v_RS_R_z [m s^-1]", "v_RS_R_z", " v_RS_R_z [m s^-1]"])
            if frame_str is None or gt_velocities_x is None or gt_velocities_y is None or gt_velocities_z is None:
                continue
            gt_velocities[int(frame_str)] = np.array(
                [float(gt_velocities_x), float(gt_velocities_y), float(gt_velocities_z)], dtype=np.float64
            )
    return gt_velocities


def est_vo_pair(
    gt_poses: dict,
    gt_velocities: dict,
    intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None,
    curr_image_dir: Path,
    next_image_dir: Path,
) -> dict:
    """Run monocular VO on a pair of images with optional GT scale.

    Args:
        gt_poses: Dict mapping frame index -> GT translation (x,y,z) in meters.
        gt_velocities: Dict mapping frame index -> GT velocity (vx,vy,vz) in meters per second.
        intrinsics: 3x3 camera intrinsic matrix.
        dist_coeffs: Optional distortion coefficients for undistortion.
        curr_image_dir: Path to current image file named as its frame index (e.g. 0.jpg).
        next_image_dir: Path to next image file named as its frame index (e.g. 1.jpg).

    Returns:
        Dict with keys:
            status: "success" or "failure"
            reason: failure reason when status=="failure" (optional)
            curr_ts, next_ts: frame indices
            kpts1_inliers, kpts2_inliers: inlier matched keypoints in pixel coords
            R: 3x3 rotation
            t: 3x1 translation (scaled if scale available)
    """
    delta_t = 0.05  # seconds between adjacent frames (dataset-specific)

    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    )
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    curr_img = cv2.imread(str(curr_image_dir), cv2.IMREAD_GRAYSCALE)
    next_img = cv2.imread(str(next_image_dir), cv2.IMREAD_GRAYSCALE)

    frame_i = int(curr_image_dir.stem)
    frame_j = int(next_image_dir.stem)

    curr_gt_poses = gt_poses.get(frame_i, None)
    next_gt_poses = gt_poses.get(frame_j, None)

    curr_velo = gt_velocities.get(frame_i, None)
    next_velo = gt_velocities.get(frame_j, None)

    if curr_img is None or next_img is None:
        raise ValueError(f"Failed to read images from {curr_image_dir} or {next_image_dir}")

    kpts1, descs1 = orb.detectAndCompute(curr_img, None)
    kpts2, descs2 = orb.detectAndCompute(next_img, None)

    if descs1 is None or descs2 is None:
        return {
            "status": "failure",
            "reason": "descriptor is None",
            "curr_ts": frame_i,
            "next_ts": frame_j,
            "kpts1_inliers": [],
            "kpts2_inliers": [],
            "R": np.eye(3, dtype=np.float64).tolist(),
            "t": np.zeros((3, 1), dtype=np.float64).tolist(),
        }

    matches = bf.knnMatch(descs1, descs2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance <= 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        return {
            "status": "failure",
            "reason": "good matches are not enough (<8)",
            "curr_ts": frame_i,
            "next_ts": frame_j,
            "kpts1_inliers": [],
            "kpts2_inliers": [],
            "R": np.eye(3, dtype=np.float64).tolist(),
            "t": np.zeros((3, 1), dtype=np.float64).tolist(),
        }

    src_pts = np.float32([kpts1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    tgt_pts = np.float32([kpts2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Normalize points using camera intrinsics + distortion
    src_pts_norm = cv2.undistortPoints(src_pts, intrinsics, dist_coeffs).reshape(-1, 2)
    tgt_pts_norm = cv2.undistortPoints(tgt_pts, intrinsics, dist_coeffs).reshape(-1, 2)

    # Estimate Essential matrix with RANSAC in normalized coordinates
    E, mask = cv2.findEssentialMat(
        points1=src_pts_norm,
        points2=tgt_pts_norm,
        focal=1.0,
        pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=0.99,
        threshold=0.003,  # normalized coords; 0.001 is often too strict
    )

    if E is None or mask is None:
        return {
            "status": "failure",
            "reason": "E/mask is None (findEssentialMat failed)",
            "curr_ts": frame_i,
            "next_ts": frame_j,
            "kpts1_inliers": [],
            "kpts2_inliers": [],
            "R": np.eye(3, dtype=np.float64).tolist(),
            "t": np.zeros((3, 1), dtype=np.float64).tolist(),
        }

    # Handle multi-solution E (OpenCV may return 3x(3n))
    if E.shape != (3, 3):
        E = E[:, :3].copy()

    pose_inliers, R, t, recover_mask = cv2.recoverPose(
        E,
        src_pts_norm,
        tgt_pts_norm,
        focal=1.0,
        pp=(0.0, 0.0),
        mask=mask,
    )

    R = _orthonormalize_rotation(R)

    # Use recoverPose mask (cheirality-consistent) when available
    final_mask = recover_mask if recover_mask is not None else mask
    inlier_sel = (final_mask.ravel().astype(np.uint8) == 1)

    # If not enough inliers, fail early
    if pose_inliers < 20:
        return {
            "status": "failure",
            "reason": "not enough inliers",
            "curr_ts": frame_i,
            "next_ts": frame_j,
            "kpts1_inliers": src_pts[inlier_sel].tolist(),
            "kpts2_inliers": tgt_pts[inlier_sel].tolist(),
            "R": R.tolist(),
            "t": t.tolist(),
        }

    # Scale estimation priority:
    # 1) velocity-based scale if both velocities exist
    # 2) GT translation magnitude if both GT poses exist
    # 3) otherwise mark as failure (unscaled)
    scale = None
    if (curr_velo is not None) and (next_velo is not None):
        # Average speed * delta_t as a rough translation magnitude
        scale = float(0.5 * (np.linalg.norm(curr_velo) + np.linalg.norm(next_velo)) * delta_t)
    elif (curr_gt_poses is not None) and (next_gt_poses is not None):
        scale = float(np.linalg.norm(next_gt_poses - curr_gt_poses))

    if scale is None:
        return {
            "status": "failure",
            "reason": "no velocity and no GT scale",
            "curr_ts": frame_i,
            "next_ts": frame_j,
            "kpts1_inliers": src_pts[inlier_sel].tolist(),
            "kpts2_inliers": tgt_pts[inlier_sel].tolist(),
            "R": R.tolist(),
            "t": t.tolist(),
        }

    t_scaled = scale * t
    return {
        "status": "success",
        "curr_ts": frame_i,
        "next_ts": frame_j,
        "kpts1_inliers": src_pts[inlier_sel].tolist(),
        "kpts2_inliers": tgt_pts[inlier_sel].tolist(),
        "R": R.tolist(),
        "t": t_scaled.tolist(),
    }