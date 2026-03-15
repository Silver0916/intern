from ultralytics import YOLO
from pathlib import Path
import torch
import pandas as pd
import psutil
import time


def collect_image(test_dir: Path) -> list[Path]:
    return sorted(p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def load_model(model_path: Path) -> tuple[YOLO, str]:
    model = YOLO(str(model_path))
    if torch.cuda.is_available():
        model.to("cuda")
        return model, "cuda"
    else:
        return model, "cpu"


def infer_one_image(model: YOLO, img_path: Path, device: str) -> tuple[float, int, float, float]:
    psutil.cpu_percent(interval=None) 
    start_time = time.time()
    result = model.predict(source=str(img_path), device=device, verbose=False)
    end_time = time.time()

    inference_time = end_time - start_time
    cpu_usage = psutil.cpu_percent(interval=None)
    gpu_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    num_detections = len(result[0].boxes)

    return inference_time, num_detections, cpu_usage, gpu_mem_mb


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_model("yolov5s.pt")

    metrics = model.val(data="coco8.yaml", device=device, verbose=False)
    map50 = metrics.box.map50

    from ultralytics.utils import DATASETS_DIR
    img_dir = Path(DATASETS_DIR) / "coco8" / "images" / "val"
    rows = []
    for img_path in collect_image(img_dir):
        inference_time, num_detections, cpu_usage, gpu_mem_mb = infer_one_image(model, img_path, device)
        rows.append({
            "image":          img_path.name,
            "inference_ms":   round(inference_time * 1000, 2),
            "fps":            round(1 / inference_time, 2),
            "num_detections": num_detections,
            "cpu_percent":    cpu_usage,
            "gpu_mem_mb":     round(gpu_mem_mb, 2),
            "map50":          round(map50, 4),
        })

    df = pd.DataFrame(rows)
    out_path = results_dir / "pytorch_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(rows)} rows -> {out_path}")
    print(f"mAP@0.5: {map50:.4f}  |  avg FPS: {df['fps'].mean():.1f}  |  avg CPU: {df['cpu_percent'].mean():.1f}%")


if __name__ == "__main__":
    main()
