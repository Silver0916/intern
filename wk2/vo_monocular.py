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


def _load_gt_positions_for_frames(gt_csv: Path, frame_paths: list[Path]) -> dict[int, np.ndarray] | None:
    """Load GT positions keyed by frame index using exact timestamp matching."""
    if not gt_csv.exists():
        return None

    ts_to_pos: dict[int, np.ndarray] = {}

    with open(gt_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = _strip_row_keys(raw_row)
            ts = _get_first_key(row, ["#timestamp", "#timestamp [ns]", "timestamp", "timestamp [ns]"])
            px = _get_first_key(row, ["p_RS_R_x [m]", "p_RS_R_x"])
            py = _get_first_key(row, ["p_RS_R_y [m]", "p_RS_R_y"])
            pz = _get_first_key(row, ["p_RS_R_z [m]", "p_RS_R_z"])
            if ts is None or px is None or py is None or pz is None:
                continue
            ts_to_pos[int(ts)] = np.array([float(px), float(py), float(pz)], dtype=np.float64)

    if not ts_to_pos:
        return None

    aligned: dict[int, np.ndarray] = {}
    for frame_idx, path in enumerate(frame_paths):
        aligned[frame_idx] = ts_to_pos[int(path.stem)]
    return aligned


def _write_frame_index_csv(path: Path, frame_paths: list[Path]) -> None:
    """Write frame index to filename mapping for easier debugging."""
    rows: list[dict] = []
    for idx, frame_path in enumerate(frame_paths):
        timestamp_ns = ""
        try:
            timestamp_ns = str(int(frame_path.stem))
        except ValueError:
            pass
        rows.append({"frame": idx, "filename": frame_path.name, "timestamp_ns": timestamp_ns})
    _write_dict_rows(path, ["frame", "filename", "timestamp_ns"], rows)


def _write_aligned_gt_csv(path: Path, gt_positions: dict[int, np.ndarray] | None) -> None:
    """Write frame-indexed GT positions for evaluation/visualization."""
    if gt_positions is None:
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
    max_features: int = 1500,
    ratio_test: float = 0.75,
    min_inliers: int = 20,
    min_parallax_px: float = 1.0,
    frame_step: int = 1,
) -> dict:
    """Run frame-to-frame monocular VO and export trajectory/metrics CSVs.

    Coordinate conventions:
    - r_cw, t_cw map world points to camera coordinates: p_c = r_cw * p_w + t_cw
    - camera center in world frame: c_w = -r_cw^T * t_cw
    """
    frames_dir = Path(frames_dir)
    intrinsics_yaml = Path(intrinsics_yaml)
    gt_csv_path = Path(gt_csv) if gt_csv is not None else None

    frame_paths_all = sorted(frames_dir.glob("*.png"))
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")
    frame_paths = frame_paths_all[::frame_step]
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(frame_paths)} from {frames_dir}")

    out_dir = frames_dir.parent
    est_csv = out_dir / "estimated_poses.csv"
    pair_csv = out_dir / "vo_pair_metrics.csv"
    summary_csv = out_dir / "metrics_summary.csv"
    frame_index_csv = out_dir / "frame_index.csv"
    aligned_gt_csv = out_dir / "gt_poses_aligned.csv"

    k = _read_intrinsics(intrinsics_yaml)
    dist_coeffs = _read_distortion_coeffs(intrinsics_yaml)
    gt_positions = _load_gt_positions_for_frames(gt_csv_path, frame_paths) if gt_csv_path is not None else None

    _write_frame_index_csv(frame_index_csv, frame_paths)
    _write_aligned_gt_csv(aligned_gt_csv, gt_positions)

    orb = cv2.ORB_create(nfeatures=max_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

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

    for idx in range(1, len(frame_paths)):
        prev_img = cv2.imread(str(frame_paths[idx - 1]), cv2.IMREAD_GRAYSCALE)
        curr_img = cv2.imread(str(frame_paths[idx]), cv2.IMREAD_GRAYSCALE)

        if prev_img is None or curr_img is None:
            reason = "Failed to read one of the frame images"
            failure_reasons.append(reason)
            pair_rows.append(
                {
                    "frame_prev": idx - 1,
                    "frame_curr": idx,
                    "kp_prev": 0,
                    "kp_curr": 0,
                    "good_matches": 0,
                    "essential_inliers": 0,
                    "pose_inliers": 0,
                    "scale": "0.00000000",
                    "status": "failure",
                    "reason": reason,
                }
            )
            est_rows.append(
                {
                    "frame": idx,
                    "tx": f"{c_w_prev[0]:.8f}",
                    "ty": f"{c_w_prev[1]:.8f}",
                    "tz": f"{c_w_prev[2]:.8f}",
                    "status": "failure",
                    "reason": reason,
                    "good_matches": 0,
                    "pose_inliers": 0,
                    "scale": "0.00000000",
                }
            )
            good_matches_list.append(0.0)
            continue

        kps1, des1 = orb.detectAndCompute(prev_img, None)
        kps2, des2 = orb.detectAndCompute(curr_img, None)
        kp_prev = 0 if kps1 is None else len(kps1)
        kp_curr = 0 if kps2 is None else len(kps2)

        reason = ""
        good_matches: list[cv2.DMatch] = []
        essential_inliers = 0
        pose_inliers = 0
        scale = 0.0
        median_parallax_px = 0.0
        status = "success"

        if des1 is None or des2 is None or kp_prev < 8 or kp_curr < 8:
            status = "failure"
            reason = "Not enough descriptors/keypoints"
        else:
            knn_matches = matcher.knnMatch(des1, des2, k=2)
            for pair in knn_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio_test * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 8:
                status = "failure"
                reason = "Not enough good matches for Essential matrix"
            else:
                pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches])

                if dist_coeffs is not None:
                    pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), k, dist_coeffs).reshape(-1, 2)
                    pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), k, dist_coeffs).reshape(-1, 2)
                    e, e_mask = cv2.findEssentialMat(
                        pts1_norm,
                        pts2_norm,
                        focal=1.0,
                        pp=(0.0, 0.0),
                        method=cv2.RANSAC,
                        prob=0.999,
                        threshold=0.0015,
                    )
                else:
                    pts1_norm = pts1
                    pts2_norm = pts2
                    e, e_mask = cv2.findEssentialMat(
                        pts1_norm,
                        pts2_norm,
                        cameraMatrix=k,
                        method=cv2.RANSAC,
                        prob=0.999,
                        threshold=1.0,
                    )
                if e is None or e_mask is None:
                    status = "failure"
                    reason = "findEssentialMat failed"
                else:
                    if e.ndim == 2 and e.shape[0] >= 3 and e.shape[1] == 3 and e.shape[0] != 3:
                        # findEssentialMat can return stacked 3x3 candidates (3N x 3).
                        e = e[:3, :3]

                    essential_inliers = int(e_mask.ravel().astype(bool).sum())
                    if dist_coeffs is not None:
                        pose_inliers, r_rel, t_rel, pose_mask = cv2.recoverPose(
                            e,
                            pts1_norm,
                            pts2_norm,
                            np.eye(3, dtype=np.float64),
                            mask=e_mask,
                        )
                    else:
                        pose_inliers, r_rel, t_rel, pose_mask = cv2.recoverPose(e, pts1_norm, pts2_norm, k, mask=e_mask)
                    pose_inliers = int(pose_inliers)

                    if pose_inliers < min_inliers:
                        status = "failure"
                        reason = f"recoverPose inliers too low ({pose_inliers} < {min_inliers})"
                    else:
                        if pose_mask is None:
                            status = "failure"
                            reason = "recoverPose returned empty inlier mask"
                        else:
                            inlier_mask = pose_mask.ravel().astype(bool)
                            if int(inlier_mask.sum()) < min_inliers:
                                status = "failure"
                                reason = f"recoverPose valid mask too low ({int(inlier_mask.sum())} < {min_inliers})"
                            else:
                                flow = np.linalg.norm(pts2[inlier_mask] - pts1[inlier_mask], axis=1)
                                median_parallax_px = float(np.median(flow)) if flow.size > 0 else 0.0
                                if median_parallax_px < min_parallax_px:
                                    status = "failure"
                                    reason = f"Parallax too low ({median_parallax_px:.3f} px < {min_parallax_px:.3f} px)"

                    if status == "success":
                        if gt_positions is not None and (idx - 1) in gt_positions and idx in gt_positions:
                            scale = float(np.linalg.norm(gt_positions[idx] - gt_positions[idx - 1]))
                        else:
                            scale = 1.0

                        if not np.isfinite(scale) or scale <= 1e-12:
                            status = "failure"
                            reason = "Invalid translation scale"
                            scale = 0.0
                        else:
                            t_step = scale * t_rel
                            r_cw = _orthonormalize_rotation(r_rel @ r_cw)
                            t_cw = r_rel @ t_cw + t_step
                            c_w_prev = (-r_cw.T @ t_cw).reshape(3)

        if status == "failure":
            failure_reasons.append(reason)

        pair_rows.append(
            {
                "frame_prev": idx - 1,
                "frame_curr": idx,
                "kp_prev": kp_prev,
                "kp_curr": kp_curr,
                "good_matches": len(good_matches),
                "essential_inliers": essential_inliers,
                "pose_inliers": pose_inliers,
                "scale": f"{scale:.8f}",
                "median_parallax_px": f"{median_parallax_px:.8f}",
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
                "good_matches": len(good_matches),
                "pose_inliers": pose_inliers,
                "scale": f"{scale:.8f}",
                "median_parallax_px": f"{median_parallax_px:.8f}",
            }
        )
        good_matches_list.append(float(len(good_matches)))

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
            "kp_prev",
            "kp_curr",
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
        "gt_scale_used": int(gt_positions is not None),
        "distortion_compensated": int(dist_coeffs is not None),
    }

    if gt_positions is not None and len(gt_positions) > 0:
        valid_gt_frames = sorted(set(gt_positions.keys()) & set(range(len(est_rows))))
        if valid_gt_frames:
            gt0 = gt_positions[valid_gt_frames[0]]
            est_positions = {
                int(row["frame"]): np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
                for row in est_rows
            }
            errors: list[float] = []
            for frame in valid_gt_frames:
                gt_local = gt_positions[frame] - gt0
                err = float(np.linalg.norm(est_positions[frame] - gt_local))
                errors.append(err)
            if errors:
                stats["mean_position_error"] = float(np.mean(errors))
                stats["final_position_error"] = float(errors[-1])

    reason_counts = Counter(failure_reasons)
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
        "aligned_gt_csv": aligned_gt_csv if gt_positions is not None else None,
        "stats": stats,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run monocular VO on a frame sequence.")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing frame images (*.png).")
    parser.add_argument("--intrinsics_yaml", type=str, required=True, help="Camera intrinsics YAML path.")
    parser.add_argument("--gt_csv", type=str, default=None, help="GT poses CSV path for scale and metrics.")
    parser.add_argument("--max_features", type=int, default=1500, help="ORB max features.")
    parser.add_argument("--ratio_test", type=float, default=0.75, help="Lowe ratio threshold.")
    parser.add_argument("--min_inliers", type=int, default=20, help="Minimum inliers accepted from recoverPose.")
    parser.add_argument(
        "--min_parallax_px",
        type=float,
        default=1.0,
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
