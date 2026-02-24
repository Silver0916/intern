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
    pts = _as_nx2(pts_nx2).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


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

    #for good matches 
    gt_errors = np.linalg.norm(gt_target_pts - target_pts, axis=1)
    correct_mask = gt_errors < correct_thr_px
    correct_matches = int(np.sum(correct_mask))
    correct_match_ratio = float(correct_matches / len(ref_pts)) if len(ref_pts) > 0 else 0.0

    # evaluate H_est
    est_pts = cv2.perspectiveTransform(ref_pts,match_result['H'])
    est_errors = np.linalg.norm(est_pts - ref_pts)
    est_correct_mask = est_errors < homography_success_thr_px
    est_correct_matches = int(np.sum(est_correct_mask))
    est_correct_match_ratio = float(est_correct_matches/len(ref_pts) if len(ref_pts) > 0 else 0.0)

    return {
        
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
