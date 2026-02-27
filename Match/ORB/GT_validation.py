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


def repeatibility_flann(gt_homo: np.ndarray, kp1_pts: np.ndarray, kp2_pts: np.ndarray, img2_hw: tuple[int, int], dist_thr_px: float) -> float:
    kp1_pts = _as_nx2(np.asarray(kp1_pts, dtype=np.float32))
    kp2_pts = _as_nx2(np.asarray(kp2_pts, dtype=np.float32))
    if len(kp1_pts) == 0 or len(kp2_pts) == 0:
        return 0.0

    h2, w2 = img2_hw
    # project kp1 to img2
    proj_kp1 = _project_points(gt_homo, kp1_pts)
    kp1_visible = _points_inside_image(proj_kp1, w2, h2)
    proj_kp1_vis = proj_kp1[kp1_visible]

    # FLANN match
    index_params = dict(algorithm = 1,trees = 4)
    search_params = dict(checks = 32)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(proj_kp1_vis, kp2_pts, k=1)
    repeated_matches = []
    for m in matches:
        if m.distance < dist_thr_px:
            repeated_matches.append(m)

    return len(repeated_matches) / len(kp1_pts)


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
    homo_success = (
    est_correct_matches >= min_inliers
    and np.mean(est_errors) < homography_success_thr_px
)

    # ---------------- accuracy of estimated homography on RANSAC inliers ----------------
    ransac_inliers = _as_nx2(match_result["good_pts1"][match_result["inlier_mask"]])
    ransac_pts = _project_points(match_result["H"], ransac_inliers)
    ransac_gt_pts = _project_points(gt_homo, ransac_inliers)
    ransac_errors = np.linalg.norm(ransac_pts - ransac_gt_pts, axis=1)
    ransac_correct_mask = ransac_errors < homography_success_thr_px
    ransac_correct_matches = int(np.sum(ransac_correct_mask))
    ransac_accuracy = float(ransac_correct_matches / len(ransac_inliers)) if len(ransac_inliers) > 0 else 0.0

    # --- Repeatability ---
    repeatability = None
    if "kps1" in match_result and "kps2" in match_result:
        kp1_pts = np.float32([kp.pt for kp in match_result["kps1"]]).reshape(-1, 2)
        kp2_pts = np.float32([kp.pt for kp in match_result["kps2"]]).reshape(-1, 2)
        img2 = cv2.imread(str(gt_path.parent / target_img), cv2.IMREAD_GRAYSCALE)
        if img2 is None:
            raise FileNotFoundError(f"Failed to read images for repeatability: {ref_img}, {target_img}")
        repeatability = repeatibility_flann(
                        gt_homo, 
                        kp1_pts, 
                        kp2_pts, 
                        img2.shape[:2], 
                        correct_thr_px
                        )

    return {
        "correct_matches": correct_matches,
        "total_good_matches": len(ref_pts),
        "good_matches_accuracy": gm_accuracy,
        "homo_inliers": est_correct_matches,
        "estimated_homography_accuracy": est_accuracy,
        "estimated_homography_mean_error_px": float(np.mean(est_errors)),
        "estimated_homography_success": homo_success,
        "gt_inliers": ransac_correct_matches,
        "gt_inlier_accuracy": ransac_accuracy,
        "gt_mean_error_px": float(np.mean(ransac_errors)),
        "repeatability": repeatability,
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
