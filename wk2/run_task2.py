from __future__ import annotations

"""End-to-end Task2 runner.

Pipeline:
1) Optionally generate a synthetic monocular sequence and GT data.
2) Run monocular VO on consecutive frames.
3) Render trajectory and error plots.

This file is intentionally thin: it only wires modules together and exposes
CLI arguments for quick experiments.
"""

import argparse
import importlib
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
# Keep local imports working when this file is executed directly.
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from visualize_trajectory import render_trajectory_plots
from vo_mono import run_monocular_vo


def _build_arg_parser() -> argparse.ArgumentParser:
    """Define CLI arguments for the full Task2 workflow."""
    parser = argparse.ArgumentParser(description="Task2 pipeline: simulate sequence -> run monocular VO -> visualize.")
    parser.add_argument("--output", type=str, default="wk2/output", help="Output directory root.")

    parser.add_argument("--use_sim", action="store_true", help="Generate synthetic sequence before VO.")
    parser.add_argument("--frames_dir", type=str, default=None, help="Existing frames directory (default mode).")
    parser.add_argument("--gt_csv", type=str, default=None, help="Existing GT pose CSV path (default mode).")
    parser.add_argument("--intrinsics_yaml", type=str, default=None, help="Existing intrinsics YAML path (default mode).")

    parser.add_argument("--num_frames", type=int, default=180, help="Number of synthetic frames.")
    parser.add_argument("--image_w", type=int, default=640, help="Synthetic frame width.")
    parser.add_argument("--image_h", type=int, default=480, help="Synthetic frame height.")
    parser.add_argument("--n_points", type=int, default=2000, help="Number of synthetic world points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--max_features", type=int, default=2000, help="ORB max features.")
    parser.add_argument("--ratio_test", type=float, default=0.7, help="Lowe ratio threshold.")
    parser.add_argument("--min_inliers", type=int, default=15, help="Minimum accepted pose inliers.")
    parser.add_argument(
        "--min_parallax_px",
        type=float,
        default=0.5,
        help="Minimum median pixel parallax on pose inliers to accept a frame-to-frame update.",
    )
    parser.add_argument("--frame_step", type=int, default=1, help="Use every Nth frame from --frames_dir.")
    return parser


def main() -> None:
    """Run the configured workflow and print generated artifacts and metrics."""
    args = _build_arg_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: prepare sequence inputs.
    # Default mode: reuse existing files under --output (or explicit paths).
    # Optional mode (--use_sim): generate synthetic frames + GT + intrinsics.
    if not args.use_sim:
        frames_dir = Path(args.frames_dir) if args.frames_dir else output_dir / "frames"
        gt_csv = Path(args.gt_csv) if args.gt_csv else output_dir / "gt_poses.csv"
        intrinsics_yaml = Path(args.intrinsics_yaml) if args.intrinsics_yaml else output_dir / "camera_intrinsics.yaml"
    else:
        try:
            sim_module = importlib.import_module("simulation_drone_sequence")
            generate_synthetic_sequence = getattr(sim_module, "generate_synthetic_sequence")
        except (ModuleNotFoundError, AttributeError) as exc:
            raise SystemExit("simulation_drone_sequence.py is required when --use_sim is set.") from exc
        sim_result = generate_synthetic_sequence(
            output_dir=output_dir,
            num_frames=args.num_frames,
            image_w=args.image_w,
            image_h=args.image_h,
            n_points=args.n_points,
            seed=args.seed,
        )
        frames_dir = Path(sim_result["frames_dir"])
        gt_csv = Path(sim_result["gt_csv"])
        intrinsics_yaml = Path(sim_result["intrinsics_yaml"])

    # Stage 2: estimate trajectory with monocular VO.
    vo_result = run_monocular_vo(
        frames_dir=frames_dir,
        intrinsics_yaml=intrinsics_yaml,
        gt_csv=gt_csv,
        out_dir=output_dir,
        max_features=args.max_features,
        ratio_test=args.ratio_test,
        min_inliers=args.min_inliers,
        min_parallax_px=args.min_parallax_px,
        frame_step=args.frame_step,
    )

    # Stage 3: visualize estimated trajectory against GT.
    gt_for_vis = Path(vo_result["aligned_gt_csv"]) if vo_result.get("aligned_gt_csv") else gt_csv
    vis_result = render_trajectory_plots(
        gt_csv=gt_for_vis,
        est_csv=Path(vo_result["est_csv"]),
        out_dir=output_dir,
    )

    print("Task2 pipeline completed.")
    print(f"  output_dir: {output_dir}")
    print(f"  frames_dir: {frames_dir}")
    print(f"  gt_csv: {gt_csv}")
    if vo_result.get("aligned_gt_csv"):
        print(f"  aligned_gt_csv: {vo_result['aligned_gt_csv']}")
    print(f"  intrinsics_yaml: {intrinsics_yaml}")
    print(f"  est_csv: {vo_result['est_csv']}")
    print(f"  metrics_csv: {vo_result['metrics_csv']}")
    for key, value in vis_result.items():
        print(f"  {key}: {value}")
    print("  stats:")
    for key, value in vo_result["stats"].items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
