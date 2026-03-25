from __future__ import annotations

"""Monocular visual odometry core for Task2."""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(slots=True)
class VO_config:
    """Configuration parameters for monocular VO."""
    max_features: int = 2000
    ratio_test: float = 0.7
    min_inliers: int = 15
    min_parallax_px: float = 0.5
    fallback_scale: float = 1.0
    scale_factor: float = 1.2
    nlevels: int = 8
    edge_threshold: int = 31
    fast_threshold: int = 20
    ransac_thresh: float = 0.003
    ransac_prob: float = 0.999
    delta_t: float = 1.0

    @property
    def edgeThreshold(self) -> int:
        return self.edge_threshold

    @property
    def fastThreshold(self) -> int:
        return self.fast_threshold

    def validate(self) -> None:
        if self.max_features <= 0:
            raise ValueError(f"max_features must be > 0, got {self.max_features}")
        if not (0.0 < self.ratio_test < 1.0):
            raise ValueError(f"ratio_test must be in (0, 1), got {self.ratio_test}")
        if self.min_inliers < 1:
            raise ValueError(f"min_inliers must be >= 1, got {self.min_inliers}")
        if self.min_parallax_px < 0.0:
            raise ValueError(f"min_parallax_px must be >= 0, got {self.min_parallax_px}")
        if self.fallback_scale <= 0.0:
            raise ValueError(f"fallback_scale must be > 0, got {self.fallback_scale}")
        if self.scale_factor <= 1.0:
            raise ValueError(f"scale_factor must be > 1.0, got {self.scale_factor}")
        if self.nlevels < 1:
            raise ValueError(f"nlevels must be >= 1, got {self.nlevels}")
        if self.edge_threshold < 1:
            raise ValueError(f"edge_threshold must be >= 1, got {self.edge_threshold}")
        if self.fast_threshold < 0:
            raise ValueError(f"fast_threshold must be >= 0, got {self.fast_threshold}")
        if self.ransac_thresh <= 0.0:
            raise ValueError(f"ransac_thresh must be > 0, got {self.ransac_thresh}")
        if not (0.0 < self.ransac_prob < 1.0):
            raise ValueError(f"ransac_prob must be in (0, 1), got {self.ransac_prob}")
        if self.delta_t <= 0.0:
            raise ValueError(f"delta_t must be > 0, got {self.delta_t}")


VOConfig = VO_config

_FRAME_KEYS = ["frame", "#timestamp", "#timestamp [ns]", "timestamp", "timestamp [ns]"]
_POS_X_KEYS = ["tx", "p_RS_R_x [m]", "p_RS_R_x"]
_POS_Y_KEYS = ["ty", "p_RS_R_y [m]", "p_RS_R_y"]
_POS_Z_KEYS = ["tz", "p_RS_R_z [m]", "p_RS_R_z"]
_VEL_X_KEYS = ["vx", "v_RS_R_x [m s^-1]", "v_RS_R_x"]
_VEL_Y_KEYS = ["vy", "v_RS_R_y [m s^-1]", "v_RS_R_y"]
_VEL_Z_KEYS = ["vz", "v_RS_R_z [m s^-1]", "v_RS_R_z"]


def _safe_mean(values: list[float]) -> float:
    """Return the mean, or NaN when the list is empty."""
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _read_intrinsics(path: Path) -> np.ndarray:
    """Read camera intrinsics from YAML."""
    data = _read_yaml(path)

    K = data.get("K")
    if K is not None:
        intrinsics = np.asarray(K, dtype=np.float64)
        if intrinsics.shape == (3, 3):
            return intrinsics

    intrinsics = data.get("intrinsics")
    if intrinsics is not None and len(intrinsics) >= 4:
        return np.array(
            [
                [float(intrinsics[0]), 0.0, float(intrinsics[2])],
                [0.0, float(intrinsics[1]), float(intrinsics[3])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    keys = ("fx", "fy", "cx", "cy")
    if all(key in data for key in keys):
        return np.array(
            [
                [float(data["fx"]), 0.0, float(data["cx"])],
                [0.0, float(data["fy"]), float(data["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    raise ValueError(f"Could not parse intrinsics from {path}")


def _read_distortion_coeffs(path: Path) -> np.ndarray | None:
    """Read distortion coefficients from YAML if present."""
    data = _read_yaml(path)
    for key in ("distortion_coefficients", "dist_coeffs", "distortion", "D"):
        value = data.get(key)
        if value is None:
            continue
        coeffs = np.asarray(value, dtype=np.float64).reshape(-1, 1)
        return coeffs if coeffs.size > 0 else None
    return None


def _iter_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        for raw_row in csv.DictReader(f):
            yield {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in raw_row.items()}


def _pick_first(row: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _read_gt_poses(path: Path) -> dict[int, np.ndarray]:
    """Read GT positions from CSV."""
    gt_poses: dict[int, np.ndarray] = {}
    for row in _iter_csv_rows(path):
        frame_val = _pick_first(row, _FRAME_KEYS)
        tx_val = _pick_first(row, _POS_X_KEYS)
        ty_val = _pick_first(row, _POS_Y_KEYS)
        tz_val = _pick_first(row, _POS_Z_KEYS)
        if None in (frame_val, tx_val, ty_val, tz_val):
            continue
        gt_poses[int(frame_val)] = np.array(
            [float(tx_val), float(ty_val), float(tz_val)],
            dtype=np.float64,
        )
    return gt_poses


def _read_gt_velocities(path: Path) -> dict[int, np.ndarray]:
    """Read GT velocities from CSV when velocity columns are available."""
    gt_velocities: dict[int, np.ndarray] = {}
    for row in _iter_csv_rows(path):
        frame_val = _pick_first(row, _FRAME_KEYS)
        vx_val = _pick_first(row, _VEL_X_KEYS)
        vy_val = _pick_first(row, _VEL_Y_KEYS)
        vz_val = _pick_first(row, _VEL_Z_KEYS)
        if None in (frame_val, vx_val, vy_val, vz_val):
            continue
        gt_velocities[int(frame_val)] = np.array(
            [float(vx_val), float(vy_val), float(vz_val)],
            dtype=np.float64,
        )
    return gt_velocities


def _write_dict_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write CSV rows with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _sorted_frame_paths(frames_dir: Path) -> list[Path]:
    frame_paths = (
        list(frames_dir.glob("*.png"))
        + list(frames_dir.glob("*.jpg"))
        + list(frames_dir.glob("*.jpeg"))
    )
    if not frame_paths:
        return []

    def _sort_key(path: Path) -> tuple[int, int | str]:
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.name)

    return sorted(frame_paths, key=_sort_key)


def _candidate_frame_keys(frame_idx: int, frame_path: Path) -> list[int]:
    keys: list[int] = []
    try:
        keys.append(int(frame_path.stem))
    except ValueError:
        pass
    keys.append(frame_idx)
    return keys


def _align_series_to_sequence(
    frame_paths: list[Path],
    series_by_key: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    aligned: dict[int, np.ndarray] = {}
    for idx, frame_path in enumerate(frame_paths):
        for key in _candidate_frame_keys(idx, frame_path):
            if key in series_by_key:
                aligned[idx] = np.asarray(series_by_key[key], dtype=np.float64)
                break
    return aligned


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


def _load_image(path: Path | str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image from {path}")
    return img


def _orthonormalize_rotation(r: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3) using SVD."""
    u, _, vt = np.linalg.svd(r)
    r_ortho = u @ vt
    if np.linalg.det(r_ortho) < 0:
        u[:, -1] *= -1.0
        r_ortho = u @ vt
    return r_ortho


def _detect_and_compute(
    img: np.ndarray,
    *,
    max_features: int,
    scale_factor: float,
    nlevels: int,
    edge_threshold: int,
    fast_threshold: int,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=scale_factor,
        nlevels=nlevels,
        edgeThreshold=edge_threshold,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=fast_threshold,
    )
    keypoints, descriptors = orb.detectAndCompute(img, None)
    if descriptors is None or len(keypoints) == 0:
        raise ValueError("Descriptor extraction failed")
    return keypoints, descriptors


def _match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    *,
    ratio_test: float,
    k: int = 2,
) -> list[cv2.DMatch]:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc1, desc2, k=k)
    good_matches: list[cv2.DMatch] = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance <= ratio_test * n.distance:
            good_matches.append(m)
    return good_matches


def _normalize_points(
    pts: np.ndarray,
    intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None,
) -> np.ndarray:
    coeffs = None if dist_coeffs is None or dist_coeffs.size == 0 else dist_coeffs
    return cv2.undistortPoints(pts, intrinsics, coeffs).reshape(-1, 1, 2)


def _estimate_essential(
    src_points_norm: np.ndarray,
    tgt_points_norm: np.ndarray,
    *,
    ransac_thresh: float,
    ransac_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    E, mask = cv2.findEssentialMat(
        points1=src_points_norm,
        points2=tgt_points_norm,
        focal=1.0,
        pp=(0.0, 0.0),
        method=cv2.RANSAC,
        threshold=ransac_thresh,
        prob=ransac_prob,
    )
    if E is None or mask is None:
        raise ValueError("Essential matrix estimation failed")
    if E.shape != (3, 3):
        E = np.asarray(E[:3, :3], dtype=np.float64)
    return E, mask


def _recover_pose(
    E: np.ndarray,
    src_points_norm: np.ndarray,
    tgt_points_norm: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    pose_inliers, R, t, recover_mask = cv2.recoverPose(
        E=E,
        points1=src_points_norm,
        points2=tgt_points_norm,
        focal=1.0,
        pp=(0.0, 0.0),
        mask=mask,
    )
    return int(pose_inliers), _orthonormalize_rotation(R), t, recover_mask


def _estimate_scale(
    curr_velo: np.ndarray | None,
    next_velo: np.ndarray | None,
    curr_gt_pose: np.ndarray | None,
    next_gt_pose: np.ndarray | None,
    *,
    fallback_scale: float,
    delta_t: float,
) -> tuple[float, str]:
    if curr_velo is not None and next_velo is not None:
        avg_speed = 0.5 * (np.linalg.norm(curr_velo) + np.linalg.norm(next_velo))
        scale = float(avg_speed * delta_t)
        if np.isfinite(scale) and scale > 1e-12:
            return scale, "velocity"

    if curr_gt_pose is not None and next_gt_pose is not None:
        scale = float(np.linalg.norm(next_gt_pose - curr_gt_pose))
        if np.isfinite(scale) and scale > 1e-12:
            return scale, "gt_pose"

    if np.isfinite(fallback_scale) and fallback_scale > 1e-12:
        return float(fallback_scale), "fallback"

    return 1.0, "default"


def _failure_result(
    reason: str,
    *,
    good_matches: int = 0,
    essential_inliers: int = 0,
    pose_inliers: int = 0,
    scale: float = 0.0,
    scale_source: str = "none",
    median_parallax_px: float = 0.0,
) -> dict:
    return {
        "status": "failure",
        "reason": reason,
        "good_matches": int(good_matches),
        "essential_inliers": int(essential_inliers),
        "pose_inliers": int(pose_inliers),
        "scale": float(scale),
        "scale_source": scale_source,
        "median_parallax_px": float(median_parallax_px),
        "R": np.eye(3, dtype=np.float64),
        "t": np.zeros((3, 1), dtype=np.float64),
    }


def _estimate_delta_t(curr_frame_path: Path, next_frame_path: Path, default: float = 1.0) -> float:
    try:
        curr_stem = curr_frame_path.stem
        next_stem = next_frame_path.stem
        curr_val = int(curr_stem)
        next_val = int(next_stem)
    except ValueError:
        return default

    delta = abs(next_val - curr_val)
    if delta == 0:
        return default

    digits = max(len(curr_stem), len(next_stem))
    if digits >= 18:
        scale = 1e-9
    elif digits >= 15:
        scale = 1e-6
    elif digits >= 12:
        scale = 1e-3
    else:
        scale = 1.0

    delta_t = float(delta) * scale
    return delta_t if delta_t > 0.0 else default


def est_vo_pair(
    *,
    gt_poses: dict[int, np.ndarray] | None,
    gt_velocities: dict[int, np.ndarray] | None,
    intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None,
    curr_image_dir: Path,
    next_image_dir: Path,
    curr_frame: int | None = None,
    next_frame: int | None = None,
    max_features: int = 2000,
    ratio_test: float = 0.7,
    min_inliers: int = 15,
    min_parallax_px: float = 0.5,
    fallback_scale: float = 1.0,
    delta_t: float = 1.0,
    scale_factor: float = 1.2,
    nlevels: int = 8,
    edge_threshold: int = 31,
    fast_threshold: int = 20,
    ransac_thresh: float = 0.003,
    ransac_prob: float = 0.999,
) -> dict:
    """Estimate relative pose for one image pair."""
    gt_poses = gt_poses or {}
    gt_velocities = gt_velocities or {}

    try:
        curr_img = _load_image(curr_image_dir)
        next_img = _load_image(next_image_dir)

        kpts1, descs1 = _detect_and_compute(
            curr_img,
            max_features=max_features,
            scale_factor=scale_factor,
            nlevels=nlevels,
            edge_threshold=edge_threshold,
            fast_threshold=fast_threshold,
        )
        kpts2, descs2 = _detect_and_compute(
            next_img,
            max_features=max_features,
            scale_factor=scale_factor,
            nlevels=nlevels,
            edge_threshold=edge_threshold,
            fast_threshold=fast_threshold,
        )

        good_matches = _match_descriptors(descs1, descs2, ratio_test=ratio_test)
        if len(good_matches) < 8:
            return _failure_result(
                f"not enough good matches ({len(good_matches)} < 8)",
                good_matches=len(good_matches),
            )

        src_pts = np.float32([kpts1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        tgt_pts = np.float32([kpts2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts_norm = _normalize_points(src_pts, intrinsics, dist_coeffs)
        tgt_pts_norm = _normalize_points(tgt_pts, intrinsics, dist_coeffs)

        E, essential_mask = _estimate_essential(
            src_pts_norm,
            tgt_pts_norm,
            ransac_thresh=ransac_thresh,
            ransac_prob=ransac_prob,
        )
        essential_inliers = int(np.count_nonzero(essential_mask))
        if essential_inliers < min_inliers:
            return _failure_result(
                f"not enough inliers ({essential_inliers} < {min_inliers})",
                good_matches=len(good_matches),
                essential_inliers=essential_inliers,
            )

        pose_inliers, R, t, recover_mask = _recover_pose(E, src_pts_norm, tgt_pts_norm, essential_mask)
        if pose_inliers < min_inliers:
            return _failure_result(
                f"not enough inliers ({pose_inliers} < {min_inliers})",
                good_matches=len(good_matches),
                essential_inliers=essential_inliers,
                pose_inliers=pose_inliers,
            )

        pose_mask = recover_mask.ravel().astype(bool)
        if not np.any(pose_mask):
            return _failure_result(
                "not enough inliers (0 < required minimum)",
                good_matches=len(good_matches),
                essential_inliers=essential_inliers,
                pose_inliers=pose_inliers,
            )

        parallax = np.linalg.norm(
            tgt_pts[pose_mask].reshape(-1, 2) - src_pts[pose_mask].reshape(-1, 2),
            axis=1,
        )
        median_parallax_px = float(np.median(parallax)) if parallax.size > 0 else 0.0
        if median_parallax_px < min_parallax_px:
            return _failure_result(
                f"Parallax too low ({median_parallax_px:.3f} px < {min_parallax_px:.3f} px)",
                good_matches=len(good_matches),
                essential_inliers=essential_inliers,
                pose_inliers=pose_inliers,
                median_parallax_px=median_parallax_px,
            )

        curr_gt_pose = gt_poses.get(curr_frame) if curr_frame is not None else None
        next_gt_pose = gt_poses.get(next_frame) if next_frame is not None else None
        curr_velo = gt_velocities.get(curr_frame) if curr_frame is not None else None
        next_velo = gt_velocities.get(next_frame) if next_frame is not None else None
        scale, scale_source = _estimate_scale(
            curr_velo,
            next_velo,
            curr_gt_pose,
            next_gt_pose,
            fallback_scale=fallback_scale,
            delta_t=delta_t,
        )

        return {
            "status": "success",
            "reason": "",
            "good_matches": len(good_matches),
            "essential_inliers": essential_inliers,
            "pose_inliers": pose_inliers,
            "scale": float(scale),
            "scale_source": scale_source,
            "median_parallax_px": median_parallax_px,
            "R": np.asarray(R, dtype=np.float64).reshape(3, 3),
            "t": np.asarray(t, dtype=np.float64).reshape(3, 1) * float(scale),
        }
    except Exception as exc:
        return _failure_result(str(exc))


class VOdataIO:
    """Compatibility wrapper around the module-level IO helpers."""

    def read_intrinsics(self, path: Path) -> np.ndarray:
        return _read_intrinsics(path)

    def read_distortion_coeffs(self, path: Path) -> np.ndarray | None:
        return _read_distortion_coeffs(path)

    def read_gt_poses(self, path: Path) -> dict[int, np.ndarray]:
        return _read_gt_poses(path)

    def read_velocities(self, path: Path) -> dict[int, np.ndarray]:
        return _read_gt_velocities(path)

    def write_dict_rows(self, path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        _write_dict_rows(path, fieldnames, rows)

    def write_frame_index_csv(self, path: Path, frame_paths: list[Path]) -> None:
        _write_frame_index_csv(path, frame_paths)

    def write_aligned_gt_csv(self, path: Path, gt_positions: dict[int, np.ndarray]) -> None:
        _write_aligned_gt_csv(path, gt_positions)


class MonocularVO:
    """Compatibility wrapper around the module-level VO helpers."""

    def load_image(self, path: Path | str) -> np.ndarray:
        return _load_image(path)

    def _orthonormalize_rotation(self, r: np.ndarray) -> np.ndarray:
        return _orthonormalize_rotation(r)

    def detect_and_compute(
        self,
        img: np.ndarray,
        max_features: int,
        scale_factor: float,
        nlevels: int,
        edgeThreshold: int,
        fastThreshold: int,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        return _detect_and_compute(
            img,
            max_features=max_features,
            scale_factor=scale_factor,
            nlevels=nlevels,
            edge_threshold=edgeThreshold,
            fast_threshold=fastThreshold,
        )

    def match_descriptors(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_test: float,
        k: int = 2,
    ) -> list[cv2.DMatch]:
        return _match_descriptors(desc1, desc2, ratio_test=ratio_test, k=k)

    def normalize_points(
        self,
        pts: np.ndarray,
        intrinsics: np.ndarray,
        dist_coeffs: np.ndarray | None,
    ) -> np.ndarray:
        return _normalize_points(pts, intrinsics, dist_coeffs)

    def estimate_essential(
        self,
        src_points_norm: np.ndarray,
        tgt_points_norm: np.ndarray,
        ratio_test: float,
        ransac_thresh: float,
        ransac_prob: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del ratio_test
        return _estimate_essential(
            src_points_norm,
            tgt_points_norm,
            ransac_thresh=ransac_thresh,
            ransac_prob=ransac_prob,
        )

    def recover_pose(
        self,
        E: np.ndarray,
        src_points_norm: np.ndarray,
        tgt_points_norm: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        return _recover_pose(E, src_points_norm, tgt_points_norm, mask)

    def estimate_scale(
        self,
        curr_velo: np.ndarray | None,
        next_velo: np.ndarray | None,
        curr_gt_pose: np.ndarray | None,
        next_gt_pose: np.ndarray | None,
        fallback_scale: float,
        delta_t: float,
    ) -> tuple[float, str]:
        return _estimate_scale(
            curr_velo,
            next_velo,
            curr_gt_pose,
            next_gt_pose,
            fallback_scale=fallback_scale,
            delta_t=delta_t,
        )

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
        config: VO_config,
    ) -> dict:
        config.validate()
        return est_vo_pair(
            gt_poses=gt_poses,
            gt_velocities=gt_velocities,
            intrinsics=intrinsics,
            dist_coeffs=dist_coeffs,
            curr_image_dir=curr_image_dir,
            next_image_dir=next_image_dir,
            curr_frame=curr_frame,
            next_frame=next_frame,
            max_features=config.max_features,
            ratio_test=config.ratio_test,
            min_inliers=config.min_inliers,
            min_parallax_px=config.min_parallax_px,
            fallback_scale=config.fallback_scale,
            delta_t=config.delta_t,
            scale_factor=config.scale_factor,
            nlevels=config.nlevels,
            edge_threshold=config.edge_threshold,
            fast_threshold=config.fast_threshold,
            ransac_thresh=config.ransac_thresh,
            ransac_prob=config.ransac_prob,
        )


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
    """Run sequence VO and write Task2-compatible CSV outputs."""
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

    gt_poses_raw: dict[int, np.ndarray] = {}
    gt_velocities_raw: dict[int, np.ndarray] = {}
    if gt_csv_path is not None and gt_csv_path.exists():
        gt_poses_raw = _read_gt_poses(gt_csv_path)
        gt_velocities_raw = _read_gt_velocities(gt_csv_path)

    aligned_gt_positions = _align_series_to_sequence(frame_paths, gt_poses_raw)
    aligned_gt_velocities = _align_series_to_sequence(frame_paths, gt_velocities_raw)

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
        delta_t = _estimate_delta_t(frame_paths[idx - 1], frame_paths[idx], default=1.0)
        pair_result = est_vo_pair(
            gt_poses=aligned_gt_positions,
            gt_velocities=aligned_gt_velocities,
            intrinsics=intrinsics,
            dist_coeffs=dist_coeffs,
            curr_image_dir=frame_paths[idx - 1],
            next_image_dir=frame_paths[idx],
            curr_frame=idx - 1,
            next_frame=idx,
            max_features=max_features,
            ratio_test=ratio_test,
            min_inliers=min_inliers,
            min_parallax_px=min_parallax_px,
            fallback_scale=last_valid_scale,
            delta_t=delta_t,
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
        "gt_scale_used": int(bool(aligned_gt_positions) or bool(aligned_gt_velocities)),
        "distortion_compensated": int(dist_coeffs is not None and dist_coeffs.size > 0),
        "last_valid_scale": float(last_valid_scale),
    }

    if aligned_gt_positions:
        valid_gt_frames = sorted(set(aligned_gt_positions.keys()) & set(range(len(est_rows))))
        if valid_gt_frames:
            gt0 = aligned_gt_positions[valid_gt_frames[0]]
            est_positions = {
                int(row["frame"]): np.array(
                    [float(row["tx"]), float(row["ty"]), float(row["tz"])],
                    dtype=np.float64,
                )
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
    stats["scale_source_counts"] = (
        "; ".join(f"{k}:{v}" for k, v in sorted(scale_source_counter.items()))
        if scale_source_counter
        else ""
    )
    stats["failure_reason_counts"] = (
        "; ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items()))
        if reason_counts
        else ""
    )

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
