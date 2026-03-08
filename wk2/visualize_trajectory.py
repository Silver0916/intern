from __future__ import annotations

"""Trajectory and error plotting utilities for Task2 outputs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_positions(csv_path: Path) -> dict[int, np.ndarray]:
    """Load per-frame 3D positions from a pose CSV file."""
    positions: dict[int, np.ndarray] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for raw_row in csv.DictReader(f):
            row = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in raw_row.items()}

            frame_val = None
            for key in ["frame", "#timestamp", "#timestamp [ns]", "timestamp", "timestamp [ns]"]:
                if key in row and row[key] != "":
                    frame_val = row[key]
                    break

            tx_val = None
            ty_val = None
            tz_val = None
            for key in ["tx", "p_RS_R_x [m]", "p_RS_R_x"]:
                if key in row and row[key] != "":
                    tx_val = row[key]
                    break
            for key in ["ty", "p_RS_R_y [m]", "p_RS_R_y"]:
                if key in row and row[key] != "":
                    ty_val = row[key]
                    break
            for key in ["tz", "p_RS_R_z [m]", "p_RS_R_z"]:
                if key in row and row[key] != "":
                    tz_val = row[key]
                    break

            if frame_val is None or tx_val is None or ty_val is None or tz_val is None:
                continue

            frame = int(frame_val)
            positions[frame] = np.array([float(tx_val), float(ty_val), float(tz_val)], dtype=np.float64)
    return positions


def _align_sim3(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> np.ndarray:
    """Align src trajectory to dst trajectory with Umeyama similarity transform."""
    if src_xyz.shape != dst_xyz.shape or src_xyz.ndim != 2 or src_xyz.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src_xyz.shape} and {dst_xyz.shape}")

    n = src_xyz.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for Sim(3) alignment.")

    src_mean = np.mean(src_xyz, axis=0)
    dst_mean = np.mean(dst_xyz, axis = 0)
    src_centered = src_xyz - src_mean
    dst_centered = dst_xyz - dst_mean

    cov = dst_centered.T @ src_centered / n
    U, S, Vt = np.linalg.svd(cov)
    var_src = np.sum(src_centered ** 2) /n
    R = U @ Vt
    if np.linalg.det(R) <0:
        D = np.diag([1, 1, -1])
        Vt[-1, :] *= -1
        R = U @ Vt
        scale = np.trace(np.diag(S) @ D) / var_src
    else:
        scale = np.trace(np.diag(S)) / var_src
    t = dst_mean - scale * R @ src_mean.T
    return scale * R @ src_xyz.T + t


def render_trajectory_plots(gt_csv: Path, est_csv: Path, out_dir: Path) -> dict:
    """Render top-view, 3D trajectory, and position-error plots."""
    gt_csv = Path(gt_csv)
    est_csv = Path(est_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_pos = _load_positions(gt_csv)
    est_pos = _load_positions(est_csv)
    common_frames = sorted(set(gt_pos.keys()) & set(est_pos.keys()))
    if not common_frames:
        raise ValueError("No overlapping frame IDs between GT and estimated poses.")

    # Shift GT so that frame-0 is origin, matching VO initialization.
    gt0 = gt_pos[common_frames[0]]
    gt_xyz = np.asarray([gt_pos[idx] - gt0 for idx in common_frames], dtype=np.float64)
    est_xyz_raw = np.asarray([est_pos[idx] for idx in common_frames], dtype=np.float64)
    est_xyz = _align_sim3(est_xyz_raw, gt_xyz)
    errors = np.linalg.norm(est_xyz - gt_xyz, axis=1)

    plot_2d = out_dir / "trajectory_2d.png"
    plot_3d = out_dir / "trajectory_3d.png"
    plot_err = out_dir / "position_error.png"

    # Plot 1: top-view trajectory, common for navigation debugging.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="GT", linewidth=2.0)
    ax.plot(est_xyz[:, 0], est_xyz[:, 2], label="Estimated", linewidth=1.8)
    ax.set_title("Top-View Trajectory (X-Z, Sim(3) Aligned)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_2d, dpi=150)
    plt.close(fig)

    # Plot 2: full 3D trajectory.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="GT", linewidth=2.0)
    ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], label="Estimated", linewidth=1.8)
    ax.set_title("3D Trajectory (Sim(3) Aligned)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_3d, dpi=150)
    plt.close(fig)

    # Plot 3: frame-wise absolute position error.
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(common_frames, errors, color="tab:red", linewidth=1.8)
    ax.set_title("Position Error vs Frame (After Sim(3) Alignment)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Euclidean Error (m)")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(plot_err, dpi=150)
    plt.close(fig)

    return {"plot_2d": plot_2d, "plot_3d": plot_3d, "plot_err": plot_err}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render trajectory plots from GT and estimated poses.")
    parser.add_argument("--gt_csv", type=str, required=True, help="GT poses CSV file.")
    parser.add_argument("--est_csv", type=str, required=True, help="Estimated poses CSV file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for PNG outputs.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    outputs = render_trajectory_plots(gt_csv=Path(args.gt_csv), est_csv=Path(args.est_csv), out_dir=Path(args.out_dir))
    print("Plots written:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
