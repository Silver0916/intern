"""
evaluate.py — Master evaluation script for ORB vs SIFT on HPatches dataset.

Iterates all 116 scenes × pairs 2-6 × {ORB, SIFT}, computes GT-based metrics
(correct match ratio, repeatability, homography accuracy) and timing, then
writes everything to a master CSV.

Usage:
    python evaluate.py
    python evaluate.py --dataset_dir "D:\intern_dataset\hpatches-sequences-release" \
                       --eval_output_dir "D:\projs\intern\Match\eval\output" \
                       --correct_thr_px 3.0
"""

import sys
import csv
import argparse
from pathlib import Path

# Allow importing from sibling packages without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ORB"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SIFT"))

from orb_match import ORB_feature_matching
from SIFT_match import SIFT_feature_matching
from GT_compute import GT_compute

# ----------- Default paths -----------
DATASET_DIR    = Path(r"D:\intern_dataset\hpatches-sequences-release")
ORB_OUT_ROOT   = Path(r"D:\projs\intern\Match\ORB\output\hpatches-sequences-release")
SIFT_OUT_ROOT  = Path(r"D:\projs\intern\Match\SIFT\output\hpatches-sequences-release")
EVAL_OUTPUT_DIR = Path(r"D:\projs\intern\Match\eval\output")

FIELDNAMES = [
    "algorithm", "scene", "scene_type", "pair_idx",
    "kp1", "kp2", "good_matches", "inliers", "inlier_ratio",
    "correct_matches", "total_good_matches", "correct_match_ratio",
    "homo_inliers", "homo_inlier_ratio", "homo_mean_error_px", "homo_success",
    "repeatability",
    "t_detect", "t_desc", "t_match",
    "status", "reason",
]


def evaluate_all(
    dataset_dir: Path = DATASET_DIR,
    orb_output_root: Path = ORB_OUT_ROOT,
    sift_output_root: Path = SIFT_OUT_ROOT,
    eval_output_dir: Path = EVAL_OUTPUT_DIR,
    correct_thr_px: float = 3.0,
) -> Path:
    """
    Run ORB and SIFT matching with GT evaluation on all scenes and pairs.
    Returns the path to the written master CSV.
    """
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_output_dir / "eval_results.csv"

    scenes = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if not scenes:
        raise FileNotFoundError(f"No scene subfolders found under: {dataset_dir}")

    algo_configs = [
        ("ORB",  ORB_feature_matching,  orb_output_root),
        ("SIFT", SIFT_feature_matching, sift_output_root),
    ]

    total = len(scenes) * 5 * len(algo_configs)
    done = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for scene_path in scenes:
            scene_name = scene_path.name
            scene_type = "illumination" if scene_name.startswith("i_") else "viewpoint"
            ref_img_path = scene_path / "1.ppm"

            for pair_idx in range(2, 7):
                target_img_path = scene_path / f"{pair_idx}.ppm"
                gt_path = scene_path / f"H_1_{pair_idx}"

                if not ref_img_path.exists() or not target_img_path.exists() or not gt_path.exists():
                    continue

                for algo_name, match_fn, out_root in algo_configs:
                    done += 1
                    pair_out_dir = out_root / scene_name / f"1_{pair_idx}"
                    pair_out_dir.mkdir(parents=True, exist_ok=True)

                    row = {
                        "algorithm": algo_name,
                        "scene": scene_name,
                        "scene_type": scene_type,
                        "pair_idx": pair_idx,
                    }

                    try:
                        match_result = match_fn(
                            ref_img_path, target_img_path, pair_out_dir,
                            return_points=True,
                        )

                        gt_metrics = GT_compute(
                            gt_path=gt_path,
                            match_result=match_result,
                            correct_thr_px=correct_thr_px,
                        )

                        inlier_count = int(match_result["inlier_mask"].sum())
                        good_count = len(match_result["good_pts1"])

                        row.update({
                            "kp1": len(match_result["kp1"]),
                            "kp2": len(match_result["kp2"]),
                            "good_matches": good_count,
                            "inliers": inlier_count,
                            "inlier_ratio": f"{inlier_count / good_count:.6f}" if good_count > 0 else "0",
                            "correct_matches": gt_metrics["correct_matches"],
                            "total_good_matches": gt_metrics["total_good_matches"],
                            "correct_match_ratio": f"{gt_metrics['correct_match_ratio']:.6f}",
                            "homo_inliers": gt_metrics["homo_inliers"],
                            "homo_inlier_ratio": f"{gt_metrics['homo_inlier_ratio']:.6f}",
                            "homo_mean_error_px": f"{gt_metrics['homo_mean_error_px']:.4f}",
                            "homo_success": gt_metrics["homo_success"],
                            "repeatability": f"{gt_metrics['repeatability']:.6f}" if gt_metrics["repeatability"] is not None else "",
                            "t_detect": f"{match_result['t_detect']:.6f}",
                            "t_desc": f"{match_result['t_desc']:.6f}",
                            "t_match": f"{match_result['t_match']:.6f}",
                            "status": "success",
                            "reason": "",
                        })

                    except Exception as e:
                        row.update({k: "" for k in FIELDNAMES if k not in row})
                        row["status"] = "failure"
                        row["reason"] = str(e)

                    writer.writerow(row)
                    f.flush()
                    print(f"[{done}/{total}] [{algo_name}] {scene_name} 1-{pair_idx}: {row['status']}")

    print(f"\nEvaluation complete. Results written to: {csv_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ORB and SIFT on HPatches with GT metrics.")
    parser.add_argument("--dataset_dir", type=str, default=str(DATASET_DIR),
                        help="Path to hpatches-sequences-release directory.")
    parser.add_argument("--orb_output_root", type=str, default=str(ORB_OUT_ROOT),
                        help="Root directory for ORB visualisation output.")
    parser.add_argument("--sift_output_root", type=str, default=str(SIFT_OUT_ROOT),
                        help="Root directory for SIFT visualisation output.")
    parser.add_argument("--eval_output_dir", type=str, default=str(EVAL_OUTPUT_DIR),
                        help="Directory where eval_results.csv will be saved.")
    parser.add_argument("--correct_thr_px", type=float, default=3.0,
                        help="Pixel threshold for a match to be considered correct (default: 3.0).")
    args = parser.parse_args()

    evaluate_all(
        dataset_dir=Path(args.dataset_dir),
        orb_output_root=Path(args.orb_output_root),
        sift_output_root=Path(args.sift_output_root),
        eval_output_dir=Path(args.eval_output_dir),
        correct_thr_px=args.correct_thr_px,
    )
