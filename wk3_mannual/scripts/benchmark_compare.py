from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    report_dir = project_root / "report"
    report_dir.mkdir(exist_ok=True)

    pt = pd.read_csv(results_dir / "pytorch_results.csv")
    trt = pd.read_csv(results_dir / "trt_results.csv")

    pt_summary = {
        "model":           "PyTorch (FP32)",
        "avg_fps":         round(pt["fps"].mean(), 1),
        "avg_inference_ms": round(pt["inference_ms"].mean(), 2),
        "avg_detections":  round(pt["num_detections"].mean(), 1),
        "avg_cpu_percent": round(pt["cpu_percent"].mean(), 1),
        "avg_gpu_mem_mb":  round(pt["gpu_mem_mb"].mean(), 1),
        "map50":           round(pt["map50"].iloc[0], 4),
    }

    trt_summary = {
        "model":           "TensorRT (INT8)",
        "avg_fps":         round(trt["fps"].mean(), 1),
        "avg_inference_ms": round(trt["inference_ms"].mean(), 2),
        "avg_detections":  round(trt["num_detections"].mean(), 1),
        "avg_cpu_percent": round(trt["cpu_percent"].mean(), 1),
        "avg_gpu_mem_mb":  round(trt["gpu_mem_mb"].mean(), 1),
        "map50":           round(trt["map50"].iloc[0], 4),
    }

    summary_df = pd.DataFrame([pt_summary, trt_summary]).set_index("model")
    print("\n=== Benchmark Comparison ===")
    print(summary_df.to_string())

    out_csv = report_dir / "comparison.csv"
    summary_df.to_csv(out_csv)
    print(f"\nSaved -> {out_csv}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ["PyTorch (FP32)", "TensorRT (INT8)"]
    colors = ["#4C72B0", "#DD8452"]

    axes[0].bar(labels, [pt_summary["avg_fps"], trt_summary["avg_fps"]], color=colors)
    axes[0].set_title("Average FPS")
    axes[0].set_ylabel("FPS")

    axes[1].bar(labels, [pt_summary["avg_cpu_percent"], trt_summary["avg_cpu_percent"]], color=colors)
    axes[1].set_title("Average CPU Usage (%)")
    axes[1].set_ylabel("CPU %")

    axes[2].bar(labels, [pt_summary["avg_detections"], trt_summary["avg_detections"]], color=colors)
    axes[2].set_title("Avg Detections per Image")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    out_png = report_dir / "comparison.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
