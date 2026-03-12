from __future__ import annotations
from sys import path

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
from dataclasses import dataclass
import pandas as pd

@dataclass(slots = True)
class VO_config:

    """Configuration parameters for monocular VO."""
    max_features: int = 2000,
    ratio_test: float = 0.7,
    min_inliers: int = 15,
    min_parallax_px: float = 0.5,
    fallback_scale: float = 1.0,
    max_features = max_features
    ratio_test = ratio_test
    min_inliers = min_inliers
    min_parallax_px = min_parallax_px
    fallback_scale = fallback_scale

    def validate(self) -> None:
        if self.max_features <= 0:
            raise ValueError(f"max_features must be > 0, got {self.max_features}")
        if not (0.0 < self.ratio_test < 1.0):
            raise ValueError(f"ratio_test must be in (0,1), got {self.ratio_test}")
        if self.min_inliers < 1:
            raise ValueError(f"min_inliers must be >= 1, got {self.min_inliers}")
        if self.min_parallax_px < 0.0:
            raise ValueError(f"min_parallax_px must be >= 0, got {self.min_parallax_px}")
        if self.fallback_scale <= 0.0:
            raise ValueError(f"fallback_scale must be > 0, got {self.fallback_scale}")
        if self.delta_t <= 0.0:
            raise ValueError(f"delta_t must be > 0, got {self.delta_t}")




def _safe_mean(values: list[float]) -> float:
    """Return mean or NaN if list is empty."""
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))

class VOdataIO():
    def __init__(self):
        pass

    def read_intrinsics(self, path: Path):
        # Read camera intrinsics from YAML file and return as 3x3 matrix or None if not found
        with open(str(path), 'r', encoding = 'utf-8') as f:
            data = yaml.safe_load(f)
            intrinsics = data.get('intrinsics', None)
            intrinsics_matrix = np.array([[intrinsics[0], 0, intrinsics[2]], 
                                          [0, intrinsics[1], intrinsics[3]], 
                                          [0, 0, 1]], 
                                          dtype=np.float64) if intrinsics else None
            if intrinsics_matrix is None:
                raise ValueError(f"Intrinsics not found in {path}")
            return intrinsics_matrix
    
    def read_distortion_coeffs(self, path: Path):
        # Read distortion coefficients from YAML file and return as Nx1 array or None if not found
        with open(str(path),'r', encoding='utf-8') as f:
            data  =yaml.safe_load(f)
            distortion_coefficients = data.get('distortion_coefficients', None)
            distortion_coeffs_array = np.array(distortion_coefficients, dtype=np.float64) if distortion_coefficients else None
            if distortion_coeffs_array is None:
                raise ValueError(f"Distortion coefficients not found in {path}")
            return distortion_coeffs_array
        
    def read_gt_poses(self, path: Path) -> dict[int, np.ndarray]:
        # Read GT poses from CSV file and return as dict mapping frame index to 3D position array
        gt_poses = {}
        with open(str(path), 'r', encoding = 'utf-8', newline='') as f:
            df = pd.read_csv(f)

            df.columns = df.columns.str.strip()

            for _, row in df.iterrows():
                for key in ["frame", "#timestamp", "#timestamp [ns]", "timestamp", "timestamp [ns]"]:
                    if key in row and pd.notna(row[key]):
                        frame_idx = int(row[key])
                        break
                    else:
                        continue
                for key in ["tx", " p_RS_R_x [m]", "p_RS_R_x"]:
                    if key in row and pd.notna(row[key]):
                        tx = float(row[key])
                        break
                    else:
                        continue
                for key in ["ty", "p_RS_R_y [m]", "p_RS_R_y"]:
                    if key in row and pd.notna(row[key]):
                        ty  =float(row[key])
                        break
                    else:
                        continue
                for key in ["tz", "p_RS_R_z [m]", "p_RS_R_z"]:
                    if key in row and pd.notna(row[key]):
                        tz = float(row[key])
                        break
                    else:
                        continue
                gt_poses[frame_idx] = np.array([tx, ty, tz], dtype=np.float64)
        return gt_poses
    
    def read_velocities(self, path: Path) -> dict[int, np.ndarray]:
        with open(str(path), 'r', encoding='utf-8', newline='') as f:
            df = pd.read_csv(f)
            gt_velocities = {}
            df.columns = df.columns.str.strip()
            for _, row in df.iterrows():
                for key in ["frame", "#timestamp", "#timestamp [ns]", "timestamp", "timestamp [ns]"]:
                    if key in row and pd.notna(row[key]):
                        frame_idx = int(row[key])
                        break
                    else:
                        continue
                for key in ["vx", " v_RS_R_x [m s^-1]", "v_RS_R_x"]:
                    if key in row and pd.notna(row[key]):
                        vx = float(row[key])
                        break
                    else:
                        continue
                for key in ["vy", "v_RS_R_y [m s^-1]", "v_RS_R_y"]:
                    if key in row and pd.notna(row[key]):
                        vy = float(row[key])
                        break
                    else:
                        continue
                for key in ["vz", "v_RS_R_z [m s^-1]", "v_RS_R_z"]:
                    if key in row and pd.notna(row[key]):
                        vz = float(row[key])
                        break
                    else:
                        continue
                gt_velocities[frame_idx] = np.array([vx, vy, vz], dtype=np.float64)     
        return gt_velocities                

    def write_dict_rows(self, path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        # Write a list of dict rows to a CSV file with specified fieldnames, ensuring UTF-8 encoding and no index column.
        df = pd.DataFrame(rows, columns=fieldnames)
        df.reindex(columns=fieldnames)
        df.to_csv(str(path), index=False, encoding='utf-8')

    def write_frame_index_csv(self, path: Path, frame_paths: list[Path]) -> None:
        rows: list[dict] = []
        for idx, frame_path in enumerate(frame_paths):
            try:
                timestamp_ns = str(int(frame_path.stem))
            except ValueError:
                timestamp_ns = ""
            rows.append({"frame": idx, "filename": frame_path.name, "timestamp_ns": timestamp_ns})
        self.write_dict_rows(path, ["frame", "filename", "timestamp_ns"], rows)
    
    def write_aligned_gt_csv(self, path: Path, gt_positions: dict[int, np.ndarray]) -> None:
        if not gt_positions:
            return
        rows = []
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
        self.write_dict_rows(path, ["frame", "tx", "ty", "tz"], rows)   
        
class MonocularVO():
    def __init__(self):
        pass

    def load_image(self, path: Path | str) -> np.ndarray:
        """Load an image in grayscale."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image from {path}")
        return img
    
    def _orthonormalize_rotation(r: np.ndarray) -> np.ndarray:
        """Project a near-rotation matrix onto SO(3) via SVD."""
        u, _, vt = np.linalg.svd(r)
        r_ortho = u @ vt
        if np.linalg.det(r_ortho) < 0:
            u[:, -1] *= -1.0
            r_ortho = u @ vt
        return r_ortho
    
    def detect_and_compute(self, img: np.ndarray, max_features: int, scale_factor: float, nlevels: int, edgeThreshold: int, fastThreshold: int) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """Detect ORB keypoints and compute descriptors."""
        orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=scale_factor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=fastThreshold,
        )

        kpts, descs = orb.detectAndCompute(img, None)
        if descs is None:
            raise ValueError("Descriptor is None after detectAndCompute")
        return kpts, descs
    
    def match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray, ratio_test: float, k:int = 2) -> list[cv2.DMatch]:
        """Match descriptors with BFMatcher and apply Lowe's ratio test."""
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc1, desc2, k = k)
        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance <= ratio_test * n.distance:
                good_matches.append(m)
        return good_matches

    def normalize_points(self, pts: np.ndarray, intrinsics: np.ndarray, dist_coeffs: np.ndarray | None) -> np.ndarray:
        """Undistort and normalize pixel coordinates to normalized camera coords."""
        pts_undist = cv2.undistortPoints(pts, intrinsics, dist_coeffs).reshape(-1, 1, 2)
        return pts_undist
    
    def estimate_essential(self, src_points_norm, tgt_points_norm, ratio_test: float, ransac_thresh: float, ransac_prob: float) -> tuple[np.ndarray, np.ndarray]:
        """Estimate Essential matrix with RANSAC."""
        E, mask = cv2.findEssentialMat(
            points1 = src_points_norm, 
            points2 = tgt_points_norm,
            focal = 1.,
            pp=(0., 0.),
            method = cv2.RANSAC,
            threshold = ransac_thresh,
            prob = ransac_prob
            )
        if E is None or mask is None:
            raise ValueError("Essential matrix estimation failed (E or mask is None)")
        return E, mask
    
    def recover_pose(self, E: np.ndarray, src_points_norm: np.ndarray, tgt_points_norm: np.ndarray, mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Recover relative pose (R,t) from Essential matrix."""
        pose_inliers, R, t, recover_mask = cv2.recoverPose(
            E = E,
            points1 = src_points_norm,
            points2 = tgt_points_norm,
            focal = 1.,
            pp = (0., 0.),
            mask = mask
        )
        R = self._orthonormalize_rotation(R)
        return pose_inliers, R, t, recover_mask
    
    def estimate_scale(self, curr_velo: np.ndarray | None, next_velo: np.ndarray | None, curr_gt_pose: np.ndarray | None, next_gt_pose: np.ndarray | None, fallback_scale: float, delta_t: float) -> tuple[float, str]:
        """Estimate scale using velocity or GT pose when available."""
        if curr_velo is not None and next_velo is not None:
            scale = 0.5 * (np.linalg.norm(curr_velo)+ np.linalg.norm(next_velo)) * delta_t
            return float(scale), 'velocity'
        else:
            if curr_gt_pose is not None and next_gt_pose is not None:
                scale = np.linalg.norm(next_gt_pose - curr_gt_pose)
                return float(scale), 'gt_pose'
            else:
                raise ValueError("Insufficient data for scale estimation: need either both velocities or both GT poses")
    
    def estimate_pair(
        self,
        curr_frame: int,
        next_frame: int,
        gt_poses: dict[int, np.ndarray],
        gt_velocities: dict[int, np.ndarray],
        intrinsics: np.ndarray,
        dist_coeffs: np.ndarray | None,
        curr_image_dir: Path,
        next_image_dir: Path,
        config: VO_config
    ) -> dict:
        """Run monocular VO on a pair of images with optional GT scale."""
        # Implementation of the monocular VO pipeline for a single image pair
        # using the provided configuration parameters.
        curr_gt_poses = gt_poses.get(curr_frame, None)
        next_gt_poses = gt_poses.get(next_frame, None)
        curr_velo = gt_velocities.get(curr_frame, None)
        next_velo = gt_velocities.get(next_frame, None)
        
        curr_img = cv2.imread(str(curr_image_dir), cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(str(next_image_dir), cv2.IMREAD_GRAYSCALE)
        # If images fail to load, raise an error immediately
        if curr_img is None or next_img is None:
            raise ValueError(f"Failed to read images from {curr_image_dir} or {next_image_dir}")
            
        # Step 1: Detect and compute ORB keypoints and descriptors
        kpts1,descs1 = self.detect_and_compute(curr_img, config.max_features, config.scale_factor, config.nlevels, config.edgeThreshold, config.fastThreshold)
        kpts2,descs2 = self.detect_and_compute(next_img, config.max_features, config.scale_factor, config.nlevels, config.edgeThreshold, config.fastThreshold)
        good_matches = self.match_descriptors(descs1, descs2, config.ratio_test)

        # If not enough good matches, return failure status with reason
        if len(good_matches) < 8:
            raise ValueError(f"Not enough good matches ({len(good_matches)} < 8)")
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        tgt_pts = np.float32([kpts2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # normalize points using camera intrinsics + distortion
        src_pts_norm = self.normalize_points(src_pts, intrinsics, dist_coeffs)
        tgt_pts_norm = self.normalize_points(tgt_pts, intrinsics, dist_coeffs)

        # Estimate Essential matrix with RANSAC in normalized coordinates
        E,mask = self.estimate_essential(src_pts_norm, tgt_pts_norm, config.ratio_test, ransac_thresh = 0.003, ransac_prob = 0.99)
        pose_inliers, R, t, recover_mask = self.recover_pose(E, src_pts_norm, tgt_pts_norm, mask)

        median_parallax_px_val = float(np.median(np.linalg.norm(tgt_pts[mask.ravel().astype(bool)] - src_pts[mask.ravel().astype(bool)], axis=2))) if mask.sum() > 0 else 0.0



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


def _write_frame_index_csv(self, path: Path, frame_paths: list[Path]) -> None:
    rows: list[dict] = []
    for idx, frame_path in enumerate(frame_paths):
        timestamp_ns = ""
        try:
            timestamp_ns = str(int(frame_path.stem))
        except ValueError:
            pass
        rows.append({"frame": idx, "filename": frame_path.name, "timestamp_ns": timestamp_ns})
    self._write_dict_rows(path, ["frame", "filename", "timestamp_ns"], rows)


def _write_aligned_gt_csv(self, path: Path, gt_positions: dict[int, np.ndarray]) -> None:
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
    self._write_dict_rows(path, ["frame", "tx", "ty", "tz"], rows)


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
