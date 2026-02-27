import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from orb_match import ORB_feature_matching


def GThomo_read(gt_path: Path | str) -> np.ndarray:
    gt_path = Path(gt_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    gt_homo = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            gt_homo.extend(map(float, parts))

    if len(gt_homo) != 9:
        raise ValueError(f"Expected 9 values in ground truth file {gt_path}, got {len(gt_homo)}")

    return np.array(gt_homo, dtype=np.float64).reshape(3, 3)


def _parse_ref_target_from_gt_name(gt_path: Path) -> tuple[str, str]:
    match = re.match(r"^H_(?P<ref>\d+)_(?P<target>\d+)$", gt_path.stem)
    if not match:
        raise ValueError(f"Failed to parse image indices from ground truth file name: {gt_path}")
    return f"{match.group('ref')}.ppm", f"{match.group('target')}.ppm"


def _as_nx2(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 3 and pts.shape[1:] == (1, 2):
        return pts.reshape(-1, 2)
    if pts.ndim == 2 and pts.shape[1] == 2:
        return pts
    raise ValueError(f"Expected points shape (N,2) or (N,1,2), got {pts.shape}")


def _project_points(H: np.ndarray, pts_nx2: np.ndarray) -> np.ndarray:
    pts = _as_nx2(pts_nx2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _points_inside_image(pts_nx2: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = _as_nx2(pts_nx2)
    x_ok = (pts[:, 0] >= 0) & (pts[:, 0] < width)
    y_ok = (pts[:, 1] >= 0) & (pts[:, 1] < height)
    return x_ok & y_ok


def _extract_keypoint_points(match_result: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    kp1 = match_result.get("kps1", match_result.get("kp1"))
    kp2 = match_result.get("kps2", match_result.get("kp2"))
    if kp1 is None or kp2 is None:
        return None, None
    kp1_pts = np.float32([kp.pt for kp in kp1]).reshape(-1, 2)
    kp2_pts = np.float32([kp.pt for kp in kp2]).reshape(-1, 2)
    return kp1_pts, kp2_pts


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) > 0 else float("nan")


def compute_repeatability_flann(
    gt_homo: np.ndarray,
    kp1_pts: np.ndarray,
    kp2_pts: np.ndarray,
    img2_hw: tuple[int, int],
    dist_thr_px: float,
) -> float:
    kp1_pts = _as_nx2(np.asarray(kp1_pts, dtype=np.float32))
    kp2_pts = _as_nx2(np.asarray(kp2_pts, dtype=np.float32))
    if len(kp1_pts) == 0 or len(kp2_pts) == 0:
        return 0.0

    h2, w2 = img2_hw
    # project kp1 to img2
    proj_kp1 = _project_points(gt_homo, kp1_pts)
    kp1_visible = _points_inside_image(proj_kp1, w2, h2)
    proj_kp1_vis = proj_kp1[kp1_visible]
    if len(proj_kp1_vis) == 0:
        return 0.0

    # FLANN match
    index_params = dict(algorithm=1, trees=4)
    search_params = dict(checks=32)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(proj_kp1_vis, kp2_pts, k=1)
    repeated_matches = []
    for m in matches:
        if len(m) > 0 and m[0].distance < dist_thr_px:
            repeated_matches.append(m)

    return len(repeated_matches) / len(kp1_pts)


def compute_repeatability_strict(
    gt_homo: np.ndarray,
    kp1_pts: np.ndarray,
    kp2_pts: np.ndarray,
    img1_hw: tuple[int, int],
    img2_hw: tuple[int, int],
    dist_thr_px: float,
) -> float:
    kp1_pts = _as_nx2(np.asarray(kp1_pts, dtype=np.float32))
    kp2_pts = _as_nx2(np.asarray(kp2_pts, dtype=np.float32))
    if len(kp1_pts) == 0 or len(kp2_pts) == 0:
        return 0.0

    h1, w1 = img1_hw
    h2, w2 = img2_hw

    proj_kp1 = _project_points(gt_homo, kp1_pts)
    kp1_visible = _points_inside_image(proj_kp1, w2, h2)
    proj_kp1_vis = proj_kp1[kp1_visible]

    inv_h = np.linalg.inv(gt_homo)
    backproj_kp2 = _project_points(inv_h, kp2_pts)
    kp2_visible = _points_inside_image(backproj_kp2, w1, h1)
    kp2_vis = kp2_pts[kp2_visible]

    if len(proj_kp1_vis) == 0 or len(kp2_vis) == 0:
        return 0.0

    diffs = proj_kp1_vis[:, np.newaxis, :] - kp2_vis[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)

    nn12 = np.argmin(dists, axis=1)
    nn21 = np.argmin(dists, axis=0)
    idx1 = np.arange(len(proj_kp1_vis))
    mutual = idx1 == nn21[nn12]
    close = dists[idx1, nn12] < dist_thr_px
    repeatable = int(np.sum(mutual & close))

    denom = min(len(proj_kp1_vis), len(kp2_vis))
    return float(repeatable / denom) if denom > 0 else 0.0


def GT_compute(
    gt_path: Path | str,
    match_result: dict | None = None,
    output_dir: Path | str | None = None,
    correct_thr_px: float = 3.0,
    homography_success_thr_px: float = 5.0,
    min_inliers: int = 10,
) -> dict:
    gt_path = Path(gt_path)
    gt_homo = GThomo_read(gt_path)
    ref_img, target_img = _parse_ref_target_from_gt_name(gt_path)

    if match_result is None:
        if output_dir is None:
            output_dir = Path("Match/ORB/output") / gt_path.parent.name / f"{Path(ref_img).stem}_{Path(target_img).stem}"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        match_result = ORB_feature_matching(
            gt_path.parent / ref_img,
            gt_path.parent / target_img,
            output_dir,
            return_points=True,
        )

    ref_pts = _as_nx2(match_result["good_pts1"])
    target_pts = _as_nx2(match_result['good_pts2'])
    gt_target_pts = _project_points(gt_homo, ref_pts)

    # ---------------- match accuracy of good matches ----------------
    gt_errors = np.linalg.norm(gt_target_pts - target_pts, axis=1)
    correct_mask = gt_errors < correct_thr_px
    correct_matches = int(np.sum(correct_mask))
    gm_accuracy = float(correct_matches / len(ref_pts)) if len(ref_pts) > 0 else 0.0

    # ---------------- accuracy of estimated homography on good matche points ----------------
    est_pts = _project_points(match_result["H"], ref_pts)
    est_errors = np.linalg.norm(est_pts - gt_target_pts, axis=1)
    est_correct_mask = est_errors < homography_success_thr_px
    est_correct_matches = int(np.sum(est_correct_mask))
    est_accuracy = float(est_correct_matches / len(ref_pts)) if len(ref_pts) > 0 else 0.0
    est_mean_error = _safe_mean(est_errors)
    homo_success = (
        est_correct_matches >= min_inliers
        and est_mean_error < homography_success_thr_px
    )

    # ---------------- accuracy of estimated homography on RANSAC inliers ----------------
    ransac_inliers = _as_nx2(match_result["good_pts1"][match_result["inlier_mask"]])
    ransac_pts = _project_points(match_result["H"], ransac_inliers)
    ransac_gt_pts = _project_points(gt_homo, ransac_inliers)
    ransac_errors = np.linalg.norm(ransac_pts - ransac_gt_pts, axis=1)
    ransac_correct_mask = ransac_errors < homography_success_thr_px
    ransac_correct_matches = int(np.sum(ransac_correct_mask))
    ransac_accuracy = float(ransac_correct_matches / len(ransac_inliers)) if len(ransac_inliers) > 0 else 0.0
    ransac_mean_error = _safe_mean(ransac_errors)

    # --- Repeatability ---
    repeatability = None
    repeatability_flann = None
    repeatability_strict = None
    kp1_pts, kp2_pts = _extract_keypoint_points(match_result)
    if kp1_pts is not None and kp2_pts is not None:
        img1 = cv2.imread(str(gt_path.parent / ref_img), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(gt_path.parent / target_img), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise FileNotFoundError(f"Failed to read images for repeatability: {ref_img}, {target_img}")
        repeatability_flann = compute_repeatability_flann(
            gt_homo=gt_homo,
            kp1_pts=kp1_pts,
            kp2_pts=kp2_pts,
            img2_hw=img2.shape[:2],
            dist_thr_px=correct_thr_px,
        )
        repeatability_strict = compute_repeatability_strict(
            gt_homo=gt_homo,
            kp1_pts=kp1_pts,
            kp2_pts=kp2_pts,
            img1_hw=img1.shape[:2],
            img2_hw=img2.shape[:2],
            dist_thr_px=correct_thr_px,
        )
        repeatability = repeatability_flann

    return {
        "correct_matches": correct_matches,
        "total_good_matches": len(ref_pts),
        "good_matches_accuracy": gm_accuracy,
        "homo_inliers": est_correct_matches,
        "estimated_homography_accuracy": est_accuracy,
        "estimated_homography_mean_error_px": est_mean_error,
        "estimated_homography_success": homo_success,
        "gt_inliers": ransac_correct_matches,
        "gt_inlier_accuracy": ransac_accuracy,
        "gt_mean_error_px": ransac_mean_error,
        "repeatability": repeatability,
        "repeatability_flann": repeatability_flann,
        "repeatability_strict": repeatability_strict,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute GT-based ORB metrics for one HPatches pair")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to GT homography file, e.g. .../H_1_2")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save ORB visualizations")
    parser.add_argument("--correct_thr_px", type=float, default=3.0, help="Pixel threshold for correct matches")
    parser.add_argument(
        "--homography_success_thr_px",
        type=float,
        default=5.0,
        help="Mean inlier reprojection error threshold for homography success",
    )
    parser.add_argument("--min_inliers", type=int, default=10, help="Minimum inliers for homography success")
    args = parser.parse_args()

    metrics = GT_compute(
        gt_path=args.gt_path,
        output_dir=args.output_dir,
        correct_thr_px=args.correct_thr_px,
        homography_success_thr_px=args.homography_success_thr_px,
        min_inliers=args.min_inliers,
    )

    for k, v in metrics.items():
        print(f"{k}: {v}")
