"""
Generate comparison charts from eval_results.csv.

Produces charts comparing ORB vs SIFT on HPatches:
1. Good match accuracy by scene type
2. Accuracy degradation vs pair distance
3. Repeatability (FLANN vs strict) by scene type
4. Speed comparison (detect / describe / match)
5. Scatter: repeatability vs good match accuracy (FLANN and strict)

Usage:
    python plot_results.py
    python plot_results.py --csv_path "..." --out_dir "..."
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DEFAULT_CSV = Path(r"D:\projs\intern\Match\eval\output\eval_results.csv")
DEFAULT_OUT = Path(r"D:\projs\intern\Match\eval\output\charts")


def _apply_legacy_aliases(df: pd.DataFrame) -> pd.DataFrame:
    # New metric aliases from older CSV schemas.
    if "good_matches_accuracy" not in df.columns and "correct_match_ratio" in df.columns:
        df["good_matches_accuracy"] = df["correct_match_ratio"]

    if "estimated_homography_mean_error_px" not in df.columns and "homo_mean_error_px" in df.columns:
        df["estimated_homography_mean_error_px"] = df["homo_mean_error_px"]

    if "estimated_homography_accuracy" not in df.columns and "homo_inlier_ratio" in df.columns:
        df["estimated_homography_accuracy"] = df["homo_inlier_ratio"]

    if "repeatability_flann" not in df.columns:
        if "repeatability" in df.columns:
            df["repeatability_flann"] = df["repeatability"]
        else:
            df["repeatability_flann"] = np.nan

    if "repeatability_strict" not in df.columns:
        df["repeatability_strict"] = np.nan

    if "repeatability" not in df.columns:
        df["repeatability"] = df["repeatability_flann"]

    return df


def load_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "success"].copy()
    df = _apply_legacy_aliases(df)

    numeric_cols = [
        "good_matches_accuracy",
        "repeatability",
        "repeatability_flann",
        "repeatability_strict",
        "t_detect",
        "t_desc",
        "t_match",
        "estimated_homography_accuracy",
        "estimated_homography_mean_error_px",
        "inlier_ratio",
        "gt_inlier_accuracy",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["t_total"] = df["t_detect"] + df["t_desc"] + df["t_match"]
    return df


def plot_good_match_accuracy_by_scene_type(df: pd.DataFrame, out_dir: Path):
    grouped = (
        df.groupby(["algorithm", "scene_type"])["good_matches_accuracy"]
        .mean()
        .unstack(level="scene_type")
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(grouped.index))
    width = 0.35
    colors = {"illumination": "#4C72B0", "viewpoint": "#DD8452"}

    for i, stype in enumerate(grouped.columns):
        bars = ax.bar(
            x + (i - 0.5) * width,
            grouped[stype],
            width,
            label=stype.capitalize(),
            color=colors.get(stype, f"C{i}"),
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Good Match Accuracy by Scene Type")
    ax.set_ylabel("Good Match Accuracy")
    ax.set_xlabel("Algorithm")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(title="Scene Type")
    plt.tight_layout()
    plt.savefig(out_dir / "chart1_good_matches_accuracy_by_scene_type.png", dpi=150)
    plt.close()
    print("Saved chart1")


def plot_accuracy_vs_pair_distance(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    styles = {"ORB": ("o-", "#4C72B0"), "SIFT": ("s--", "#DD8452")}

    for ax, stype in zip(axes, ["illumination", "viewpoint"]):
        sub = df[df["scene_type"] == stype]
        for algo, (style, color) in styles.items():
            vals = sub[sub["algorithm"] == algo].groupby("pair_idx")["good_matches_accuracy"].mean()
            ax.plot(vals.index, vals.values, style, color=color, label=algo, linewidth=1.8, markersize=6)
        ax.set_title(f"{stype.capitalize()} Sequences")
        ax.set_xlabel("Target Image Index")
        ax.set_ylabel("Good Match Accuracy")
        ax.set_xticks(range(2, 7))
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Accuracy Degradation vs Pair Distance", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "chart2_accuracy_vs_pair_distance.png", dpi=150)
    plt.close()
    print("Saved chart2")


def plot_repeatability(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    width = 0.35

    for ax, stype in zip(axes, ["illumination", "viewpoint"]):
        grouped = (
            df[df["scene_type"] == stype]
            .groupby("algorithm")[["repeatability_flann", "repeatability_strict"]]
            .mean()
        )
        x = np.arange(len(grouped.index))

        bars_flann = ax.bar(x - width / 2, grouped["repeatability_flann"], width, label="FLANN", color="#55A868")
        bars_strict = ax.bar(x + width / 2, grouped["repeatability_strict"], width, label="Strict 1-to-1", color="#C44E52")

        for bar in list(bars_flann) + list(bars_strict):
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{stype.capitalize()}")
        ax.set_xlabel("Algorithm")
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Repeatability")
    axes[1].legend(title="Version")
    fig.suptitle("Repeatability by Scene Type (FLANN vs Strict)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "chart3_repeatability_versions.png", dpi=150)
    plt.close()
    print("Saved chart3")


def plot_speed_comparison(df: pd.DataFrame, out_dir: Path):
    timing = df.groupby("algorithm")[["t_detect", "t_desc", "t_match"]].mean() * 1000
    timing.columns = ["Detection", "Description", "Matching"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(len(timing))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for col, color in zip(timing.columns, colors):
        bars = ax.bar(timing.index, timing[col], bottom=bottom, label=col, color=color)
        for bar, b in zip(bars, bottom):
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, b + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=8, color="white")
        bottom += timing[col].values

    for i, total in enumerate(bottom):
        ax.text(i, total + 0.5, f"{total:.1f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Average Processing Time per Image Pair")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Algorithm")
    ax.legend(title="Stage", loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "chart4_speed_comparison.png", dpi=150)
    plt.close()
    print("Saved chart4")


def plot_scatter_repeatability_vs_accuracy(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"ORB": "#4C72B0", "SIFT": "#DD8452"}
    markers = {"ORB": "o", "SIFT": "s"}
    metric_defs = [
        ("repeatability_flann", "FLANN Repeatability vs Good Match Accuracy"),
        ("repeatability_strict", "Strict Repeatability vs Good Match Accuracy"),
    ]

    for ax, (rep_col, title) in zip(axes, metric_defs):
        for algo in ["ORB", "SIFT"]:
            sub = df[df["algorithm"] == algo].dropna(subset=[rep_col, "good_matches_accuracy"])
            ax.scatter(
                sub[rep_col],
                sub["good_matches_accuracy"],
                alpha=0.35,
                s=12,
                c=colors[algo],
                marker=markers[algo],
                label=algo,
            )
        ax.set_xlabel(rep_col)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Good Match Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "chart5_scatter_repeatability_vs_accuracy.png", dpi=150)
    plt.close()
    print("Saved chart5")


def print_summary_table(df: pd.DataFrame):
    cols = ["good_matches_accuracy", "repeatability", "repeatability_strict", "t_total", "inlier_ratio"]
    summary = df.groupby(["algorithm", "scene_type"])[cols].mean()
    summary["t_total_ms"] = summary["t_total"] * 1000
    summary = summary.drop(columns="t_total")
    print("\n=== Summary (mean across all pairs) ===")
    print(summary.round(4).to_string())
    print()


def main(csv_path: str | Path, out_dir: str | Path):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)
    print(f"Loaded {len(df)} successful rows from {csv_path}")

    print_summary_table(df)

    plot_good_match_accuracy_by_scene_type(df, out_dir)
    plot_accuracy_vs_pair_distance(df, out_dir)
    plot_repeatability(df, out_dir)
    plot_speed_comparison(df, out_dir)
    plot_scatter_repeatability_vs_accuracy(df, out_dir)

    print(f"\nAll charts saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ORB vs SIFT comparison charts.")
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV), help="Path to eval_results.csv from evaluate.py")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT), help="Directory to save chart PNG files")
    args = parser.parse_args()
    main(args.csv_path, args.out_dir)
