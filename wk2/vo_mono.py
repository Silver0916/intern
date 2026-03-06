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
            frame_str = _get_first_key(row, ["#timestamp", "#timestamp [ns]", "timestamp", "timestamp [ns]", "frame"])
            gt_poses_x = _get_first_key(row, ["p_RS_R_x [m]", "p_RS_R_x", "tx"])
            gt_poses_y = _get_first_key(row, ["p_RS_R_y [m]", "p_RS_R_y", "ty"])
            gt_poses_z = _get_first_key(row, ["p_RS_R_z [m]", "p_RS_R_z", "tz"])
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
            frame_str = _get_first_key(row, ["#timestamp", "#timestamp [ns]", "timestamp", "timestamp [ns]", "frame"])
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
    max_features: int = 2000,
    ratio_test: float = 0.7,
    min_inliers: int = 15,
    min_parallax_px: float = 0.5,
    fallback_scale: float = 1.0,
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
        nfeatures=max_features,
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
        if m.distance <= ratio_test * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        return {
            "status": "failure",
            "reason": "good matches are not enough (<8)",
            "curr_ts": frame_i,
            "next_ts": frame_j,
            "good_matches": len(good_matches),
            "essential_inliers": 0,
            "pose_inliers": 0,
            "median_parallax_px": 0.0,
            "scale": 0.0,
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
            "good_matches": len(good_matches),
            "essential_inliers": 0,
            "pose_inliers": 0,
            "median_parallax_px": 0.0,
            "scale": 0.0,
            "kpts1_inliers": [],
            "kpts2_inliers": [],
            "R": np.eye(3, dtype=np.float64).tolist(),
            "t": np.zeros((3, 1), dtype=np.float64).tolist(),
        }

    # Handle multi-solution E (OpenCV may return 3x(3n))
    if E.shape != (3, 3):
        E = E[:, :3].copy()
    essential_inliers = int((mask.ravel().astype(np.uint8) > 0).sum())

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
    inlier_sel = final_mask.ravel().astype(np.uint8) > 0
    flow = np.linalg.norm(tgt_pts[inlier_sel] - src_pts[inlier_sel], axis=1)
    median_parallax_px_val = float(np.median(flow)) if flow.size > 0 else 0.0

    # If not enough inliers, fail early
    if pose_inliers < min_inliers:
        return {
            "status": "failure",
            "reason": "not enough inliers",
            "curr_ts": frame_i,
            "next_ts": frame_j,
            "good_matches": len(good_matches),
            "essential_inliers": essential_inliers,
            "pose_inliers": int(pose_inliers),
            "median_parallax_px": median_parallax_px_val,
            "scale": 0.0,
            "kpts1_inliers": src_pts[inlier_sel].tolist(),
            "kpts2_inliers": tgt_pts[inlier_sel].tolist(),
            "R": R.tolist(),
            "t": t.tolist(),
        }
    if median_parallax_px_val < min_parallax_px:
        return {
            "status": "failure",
            "reason": f"Parallax too low ({median_parallax_px_val:.3f} px < {min_parallax_px:.3f} px)",
            "curr_ts": frame_i,
            "next_ts": frame_j,
            "good_matches": len(good_matches),
            "essential_inliers": essential_inliers,
            "pose_inliers": int(pose_inliers),
            "median_parallax_px": median_parallax_px_val,
            "scale": 0.0,
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
    scale_source = "none"
    if (curr_velo is not None) and (next_velo is not None):
        # Average speed * delta_t as a rough translation magnitude
        scale = float(0.5 * (np.linalg.norm(curr_velo) + np.linalg.norm(next_velo)) * delta_t)
        scale_source = "velocity"
    elif (curr_gt_poses is not None) and (next_gt_poses is not None):
        scale = float(np.linalg.norm(next_gt_poses - curr_gt_poses))
        scale_source = "gt"

    if scale is None or (not np.isfinite(scale)) or float(scale) <= 1e-12:
        scale = float(fallback_scale)
        scale_source = "fallback"
        if (not np.isfinite(scale)) or float(scale) <= 1e-12:
            scale = 1.0
            scale_source = "unit"

    t_scaled = scale * t
    return {
        "status": "success",
        "curr_ts": frame_i,
        "next_ts": frame_j,
        "good_matches": len(good_matches),
        "essential_inliers": essential_inliers,
        "pose_inliers": int(pose_inliers),
        "median_parallax_px": median_parallax_px_val,
        "scale": float(scale),
        "scale_source": scale_source,
        "kpts1_inliers": src_pts[inlier_sel].tolist(),
        "kpts2_inliers": tgt_pts[inlier_sel].tolist(),
        "R": R.tolist(),
        "t": t_scaled.tolist(),
    }


def _sorted_frame_paths(frames_dir: Path) -> list[Path]:
    frame_paths = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg"))
    if not frame_paths:
        return []

    def _sort_key(path: Path) -> tuple[int, int | str]:
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.name)

    return sorted(frame_paths, key=_sort_key)


def _write_frame_index_csv(path: Path, frame_paths: list[Path]) -> None:
    rows: list[dict] = []
    for idx, frame_path in enumerate(frame_paths):
        timestamp_ns = ""
        try:
            timestamp_ns = str(int(frame_path.stem))
        except ValueError:
            pass
        rows.append({"frame": idx, "filename": frame_path.name, "timestamp_ns": timestamp_ns})
    _write_dict_rows(path, ["frame", "filename", "timestamp_ns"], rows)


def _write_aligned_gt_csv(path: Path, gt_positions: dict[int, np.ndarray]) -> None:
    if not gt_positions:
        return
    rows: list[dict] = []
    for frame in sorted(gt_positions.keys()):
        pos = gt_positions[frame]
        rows.append(
            {
                "frame": int(frame),
                "tx": f"{float(pos[0]):.8f}",
                "ty": f"{float(pos[1]):.8f}",
                "tz": f"{float(pos[2]):.8f}",
            }
        )
    _write_dict_rows(path, ["frame", "tx", "ty", "tz"], rows)


def run_monocular_vo(
    frames_dir: Path,
    intrinsics_yaml: Path,
    gt_csv: Path | None = None,
    out_dir: Path | None = None,
    max_features: int = 2000,
    ratio_test: float = 0.7,
    min_inliers: int = 15,
    min_parallax_px: float = 0.5,
    frame_step: int = 1,
) -> dict:
    """Run sequence VO using est_vo_pair and write Task2-compatible CSV outputs."""
    frames_dir = Path(frames_dir)
    intrinsics_yaml = Path(intrinsics_yaml)
    gt_csv_path = Path(gt_csv) if gt_csv is not None else None

    frame_paths_all = _sorted_frame_paths(frames_dir)
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")
    frame_paths = frame_paths_all[::frame_step]
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(frame_paths)} from {frames_dir}")

    out_dir = Path(out_dir) if out_dir is not None else frames_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    est_csv = out_dir / "estimated_poses.csv"
    pair_csv = out_dir / "vo_pair_metrics.csv"
    summary_csv = out_dir / "metrics_summary.csv"
    frame_index_csv = out_dir / "frame_index.csv"
    aligned_gt_csv = out_dir / "gt_poses_aligned.csv"

    intrinsics = _read_intrinsics(intrinsics_yaml)
    dist_coeffs = _read_distortion_coeffs(intrinsics_yaml)

    gt_poses: dict[int, np.ndarray] = {}
    gt_velocities: dict[int, np.ndarray] = {}
    if gt_csv_path is not None and gt_csv_path.exists():
        gt_poses = _read_gt_poses(gt_csv_path)
        gt_velocities = _read_gt_velocities(gt_csv_path)

    aligned_gt_positions: dict[int, np.ndarray] = {}
    for idx, frame_path in enumerate(frame_paths):
        frame_keys: list[int] = [idx]
        try:
            frame_keys.insert(0, int(frame_path.stem))
        except ValueError:
            pass
        for key in frame_keys:
            if key in gt_poses:
                aligned_gt_positions[idx] = gt_poses[key]
                break

    _write_frame_index_csv(frame_index_csv, frame_paths)
    _write_aligned_gt_csv(aligned_gt_csv, aligned_gt_positions)

    r_cw = np.eye(3, dtype=np.float64)
    t_cw = np.zeros((3, 1), dtype=np.float64)
    c_w_prev = np.zeros(3, dtype=np.float64)

    est_rows: list[dict] = [
        {
            "frame": 0,
            "tx": "0.00000000",
            "ty": "0.00000000",
            "tz": "0.00000000",
            "status": "init",
            "reason": "",
            "good_matches": 0,
            "pose_inliers": 0,
            "scale": "0.00000000",
            "median_parallax_px": "0.00000000",
        }
    ]
    pair_rows: list[dict] = []
    failure_reasons: list[str] = []
    good_matches_list: list[float] = []
    scale_source_counter: Counter[str] = Counter()
    last_valid_scale = 1.0

    for idx in range(1, len(frame_paths)):
        pair_result = est_vo_pair(
            gt_poses=gt_poses,
            gt_velocities=gt_velocities,
            intrinsics=intrinsics,
            dist_coeffs=dist_coeffs,
            curr_image_dir=frame_paths[idx - 1],
            next_image_dir=frame_paths[idx],
            max_features=max_features,
            ratio_test=ratio_test,
            min_inliers=min_inliers,
            min_parallax_px=min_parallax_px,
            fallback_scale=last_valid_scale,
        )

        status = str(pair_result["status"])
        reason = str(pair_result.get("reason", ""))
        good_matches = int(pair_result.get("good_matches", 0))
        essential_inliers = int(pair_result.get("essential_inliers", 0))
        pose_inliers = int(pair_result.get("pose_inliers", 0))
        median_parallax_px_val = float(pair_result.get("median_parallax_px", 0.0))
        scale = float(pair_result.get("scale", 0.0))
        scale_source = str(pair_result.get("scale_source", "none"))

        if status == "success":
            if np.isfinite(scale) and scale > 1e-12:
                last_valid_scale = scale
            scale_source_counter[scale_source] += 1
            r_rel = np.asarray(pair_result["R"], dtype=np.float64).reshape(3, 3)
            t_rel = np.asarray(pair_result["t"], dtype=np.float64).reshape(3, 1)
            r_cw = _orthonormalize_rotation(r_rel @ r_cw)
            t_cw = r_rel @ t_cw + t_rel
            c_w_prev = (-r_cw.T @ t_cw).reshape(3)
        else:
            failure_reasons.append(reason)

        pair_rows.append(
            {
                "frame_prev": idx - 1,
                "frame_curr": idx,
                "good_matches": good_matches,
                "essential_inliers": essential_inliers,
                "pose_inliers": pose_inliers,
                "scale": f"{scale:.8f}",
                "median_parallax_px": f"{median_parallax_px_val:.8f}",
                "status": status,
                "reason": reason,
            }
        )
        est_rows.append(
            {
                "frame": idx,
                "tx": f"{c_w_prev[0]:.8f}",
                "ty": f"{c_w_prev[1]:.8f}",
                "tz": f"{c_w_prev[2]:.8f}",
                "status": status,
                "reason": reason,
                "good_matches": good_matches,
                "pose_inliers": pose_inliers,
                "scale": f"{scale:.8f}",
                "median_parallax_px": f"{median_parallax_px_val:.8f}",
            }
        )
        good_matches_list.append(float(good_matches))

    _write_dict_rows(
        est_csv,
        ["frame", "tx", "ty", "tz", "status", "reason", "good_matches", "pose_inliers", "scale", "median_parallax_px"],
        est_rows,
    )
    _write_dict_rows(
        pair_csv,
        [
            "frame_prev",
            "frame_curr",
            "good_matches",
            "essential_inliers",
            "pose_inliers",
            "scale",
            "median_parallax_px",
            "status",
            "reason",
        ],
        pair_rows,
    )

    num_pairs = len(frame_paths) - 1
    num_failed = len([row for row in pair_rows if row["status"] == "failure"])
    num_success = num_pairs - num_failed

    stats: dict[str, float | int | str] = {
        "num_frames_total": len(frame_paths_all),
        "frame_step": frame_step,
        "num_frames": len(frame_paths),
        "num_pairs": num_pairs,
        "success_pairs": num_success,
        "failed_pairs": num_failed,
        "mean_good_matches": _safe_mean(good_matches_list),
        "gt_scale_used": int(len(gt_poses) > 0),
        "distortion_compensated": int(dist_coeffs is not None),
        "last_valid_scale": float(last_valid_scale),
    }

    if aligned_gt_positions:
        valid_gt_frames = sorted(set(aligned_gt_positions.keys()) & set(range(len(est_rows))))
        if valid_gt_frames:
            gt0 = aligned_gt_positions[valid_gt_frames[0]]
            est_positions = {
                int(row["frame"]): np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
                for row in est_rows
            }
            errors: list[float] = []
            for frame in valid_gt_frames:
                gt_local = aligned_gt_positions[frame] - gt0
                err = float(np.linalg.norm(est_positions[frame] - gt_local))
                errors.append(err)
            if errors:
                stats["mean_position_error"] = float(np.mean(errors))
                stats["final_position_error"] = float(errors[-1])

    reason_counts = Counter(failure_reasons)
    if scale_source_counter:
        stats["scale_source_counts"] = "; ".join(f"{k}:{v}" for k, v in sorted(scale_source_counter.items()))
    else:
        stats["scale_source_counts"] = ""
    if reason_counts:
        stats["failure_reason_counts"] = "; ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items()))
    else:
        stats["failure_reason_counts"] = ""

    summary_rows = [{"metric": k, "value": v} for k, v in stats.items()]
    _write_dict_rows(summary_csv, ["metric", "value"], summary_rows)

    return {
        "est_csv": est_csv,
        "pair_csv": pair_csv,
        "metrics_csv": summary_csv,
        "frame_index_csv": frame_index_csv,
        "aligned_gt_csv": aligned_gt_csv if aligned_gt_positions else None,
        "stats": stats,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run monocular VO on a frame sequence.")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing frame images (*.png/*.jpg).")
    parser.add_argument("--intrinsics_yaml", type=str, required=True, help="Camera intrinsics YAML path.")
    parser.add_argument("--gt_csv", type=str, default=None, help="GT poses CSV path for scale and metrics.")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory for CSV outputs.")
    parser.add_argument("--max_features", type=int, default=2000, help="ORB max features.")
    parser.add_argument("--ratio_test", type=float, default=0.7, help="Lowe ratio threshold.")
    parser.add_argument("--min_inliers", type=int, default=15, help="Minimum inliers accepted from recoverPose.")
    parser.add_argument(
        "--min_parallax_px",
        type=float,
        default=0.5,
        help="Minimum median pixel parallax on pose inliers to accept a frame-to-frame update.",
    )
    parser.add_argument("--frame_step", type=int, default=1, help="Use every Nth frame from frames_dir.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = run_monocular_vo(
        frames_dir=Path(args.frames_dir),
        intrinsics_yaml=Path(args.intrinsics_yaml),
        gt_csv=Path(args.gt_csv) if args.gt_csv else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        max_features=args.max_features,
        ratio_test=args.ratio_test,
        min_inliers=args.min_inliers,
        min_parallax_px=args.min_parallax_px,
        frame_step=args.frame_step,
    )
    print("VO completed:")
    print(f"  est_csv: {result['est_csv']}")
    print(f"  pair_csv: {result['pair_csv']}")
    print(f"  frame_index_csv: {result['frame_index_csv']}")
    if result["aligned_gt_csv"] is not None:
        print(f"  aligned_gt_csv: {result['aligned_gt_csv']}")
    print(f"  metrics_csv: {result['metrics_csv']}")
    print("  stats:")
    for key, value in result["stats"].items():
        print(f"    {key}: {value}")
