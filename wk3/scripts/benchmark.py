#!/usr/bin/env python3
"""Benchmark PyTorch baseline vs TensorRT INT8 engine on coco8 validation set."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import tensorrt as trt
from cuda.bindings import runtime as cudart
from ultralytics import YOLO

from yolo_utils import decode_yolov8_output, iou_one_to_many, load_yolo_labels_for_image, preprocess_bgr

try:
    import pynvml
except Exception:  # pragma: no cover - optional dependency
    pynvml = None


@dataclass
class Metrics:
    model: str
    map50: float
    latency_ms: float
    fps: float
    cpu_util_avg: float
    gpu_util_avg: float
    num_images: int
    repeats: int
    notes: str


class UtilSampler:
    """Periodic CPU/GPU usage sampler."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.cpu_samples: list[float] = []
        self.gpu_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_handle = None

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            if self._nvml_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    self.gpu_samples.append(float(util.gpu))
                except Exception:
                    pass
            time.sleep(self.interval)

    def stop(self) -> tuple[float, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        cpu_avg = float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
        gpu_avg = float(np.mean(self.gpu_samples)) if self.gpu_samples else 0.0
        return cpu_avg, gpu_avg


class TrtRunner:
    """TensorRT engine inference runner."""

    def __init__(self, engine_path: Path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.in_shape = tuple(self.engine.get_tensor_shape("images"))
        self.out_shape = tuple(self.engine.get_tensor_shape("output0"))
        self.out_dtype = trt.nptype(self.engine.get_tensor_dtype("output0"))

        self.host_out = np.empty(self.out_shape, dtype=self.out_dtype)
        self.in_nbytes = int(np.prod(self.in_shape) * np.dtype(np.float32).itemsize)
        self.out_nbytes = int(self.host_out.nbytes)

        err, self.d_input = cudart.cudaMalloc(self.in_nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc input failed: {err}")
        err, self.d_output = cudart.cudaMalloc(self.out_nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc output failed: {err}")
        self.bindings = [int(self.d_input), int(self.d_output)]

    def __del__(self):
        if hasattr(self, "d_input"):
            cudart.cudaFree(self.d_input)
        if hasattr(self, "d_output"):
            cudart.cudaFree(self.d_output)

    def infer(self, inp: np.ndarray) -> np.ndarray:
        err = cudart.cudaMemcpy(
            self.d_input,
            inp.ctypes.data,
            inp.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )[0]
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpy H2D failed: {err}")

        ok = self.context.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 failed")

        err = cudart.cudaMemcpy(
            self.host_out.ctypes.data,
            self.d_output,
            self.out_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )[0]
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpy D2H failed: {err}")
        return self.host_out.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("/mnt/d/projs/datasets/coco8/images/val"))
    parser.add_argument("--labels-dir", type=Path, default=Path("/mnt/d/projs/datasets/coco8/labels/val"))
    parser.add_argument("--pt-model", type=Path, default=Path("wk3/models/original/yolov8n.pt"))
    parser.add_argument("--trt-engine", type=Path, default=Path("wk3/models/optimized/yolov8n_int8.engine"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-csv", type=Path, default=Path("wk3/data/perf_compare.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("wk3/data/perf_details.json"))
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    return sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))


def build_ground_truth(images: list[Path], labels_dir: Path) -> dict[str, list[tuple[int, np.ndarray]]]:
    gt: dict[str, list[tuple[int, np.ndarray]]] = {}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt[img_path.name] = load_yolo_labels_for_image(label_path, w, h)
    return gt


def compute_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return float(np.trapz(mpre, mrec))


def evaluate_map50(
    predictions: dict[str, list[dict[str, Any]]],
    gt: dict[str, list[tuple[int, np.ndarray]]],
) -> float:
    gt_classes = sorted({cls for items in gt.values() for cls, _ in items})
    if not gt_classes:
        return 0.0

    aps: list[float] = []
    for cls in gt_classes:
        gt_by_img: dict[str, list[np.ndarray]] = {}
        total_gt = 0
        for img_name, anns in gt.items():
            boxes = [b for c, b in anns if c == cls]
            if boxes:
                gt_by_img[img_name] = boxes
                total_gt += len(boxes)
        if total_gt == 0:
            continue

        dets: list[tuple[str, float, np.ndarray]] = []
        for img_name, preds in predictions.items():
            for p in preds:
                if int(p["class_id"]) == cls:
                    dets.append((img_name, float(p["score"]), np.asarray(p["bbox"], dtype=np.float32)))
        if not dets:
            aps.append(0.0)
            continue

        dets.sort(key=lambda x: x[1], reverse=True)
        matched = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_img.items()}
        tp = np.zeros(len(dets), dtype=np.float32)
        fp = np.zeros(len(dets), dtype=np.float32)

        for i, (img_name, _, box) in enumerate(dets):
            gts = gt_by_img.get(img_name)
            if not gts:
                fp[i] = 1.0
                continue
            gt_boxes = np.stack(gts, axis=0)
            ious = iou_one_to_many(box, gt_boxes)
            best = int(np.argmax(ious))
            if ious[best] >= 0.5 and not matched[img_name][best]:
                tp[i] = 1.0
                matched[img_name][best] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / (total_gt + 1e-9)
        prec = tp_cum / (tp_cum + fp_cum + 1e-9)
        aps.append(compute_ap(rec, prec))

    return float(np.mean(aps)) if aps else 0.0


def benchmark_pytorch(
    model_path: Path,
    images: list[Path],
    gt: dict[str, list[tuple[int, np.ndarray]]],
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    warmup: int,
    repeats: int,
) -> Metrics:
    model = YOLO(str(model_path))
    warmup_image = str(images[0])
    for _ in range(warmup):
        model.predict(source=warmup_image, imgsz=imgsz, conf=conf_thres, iou=iou_thres, device="cpu", verbose=False)

    predictions: dict[str, list[dict[str, Any]]] = {}
    latencies: list[float] = []
    sampler = UtilSampler()
    sampler.start()
    for r in range(repeats):
        for img_path in images:
            t0 = time.perf_counter()
            result = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=conf_thres,
                iou=iou_thres,
                device="cpu",
                verbose=False,
            )[0]
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if r == 0:
                dets: list[dict[str, Any]] = []
                if result.boxes is not None:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    conf = result.boxes.conf.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy().astype(np.int32)
                    for b, s, c in zip(xyxy, conf, cls):
                        dets.append({"class_id": int(c), "score": float(s), "bbox": b.astype(np.float32)})
                predictions[img_path.name] = dets
    cpu_avg, gpu_avg = sampler.stop()

    latency = float(np.mean(latencies))
    fps = 1000.0 / latency if latency > 0 else 0.0
    map50 = evaluate_map50(predictions, gt)
    return Metrics(
        model="PyTorch-CPU(yolov8n.pt)",
        map50=map50,
        latency_ms=latency,
        fps=fps,
        cpu_util_avg=cpu_avg,
        gpu_util_avg=gpu_avg,
        num_images=len(images),
        repeats=repeats,
        notes="end-to-end latency (preprocess+inference+postprocess)",
    )


def benchmark_trt(
    engine_path: Path,
    images: list[Path],
    gt: dict[str, list[tuple[int, np.ndarray]]],
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    warmup: int,
    repeats: int,
) -> Metrics:
    runner = TrtRunner(engine_path)
    warmup_img = cv2.imread(str(images[0]))
    if warmup_img is None:
        raise FileNotFoundError(images[0])
    warmup_tensor, _ = preprocess_bgr(warmup_img, imgsz=imgsz)
    for _ in range(warmup):
        runner.infer(warmup_tensor)

    predictions: dict[str, list[dict[str, Any]]] = {}
    latencies: list[float] = []
    sampler = UtilSampler()
    sampler.start()
    for r in range(repeats):
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            t0 = time.perf_counter()
            x, meta = preprocess_bgr(img, imgsz=imgsz)
            out = runner.infer(x)
            dets = decode_yolov8_output(out, meta, conf_thres=conf_thres, iou_thres=iou_thres)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if r == 0:
                predictions[img_path.name] = dets
    cpu_avg, gpu_avg = sampler.stop()

    latency = float(np.mean(latencies))
    fps = 1000.0 / latency if latency > 0 else 0.0
    map50 = evaluate_map50(predictions, gt)
    return Metrics(
        model="TensorRT-INT8(yolov8n_int8.engine)",
        map50=map50,
        latency_ms=latency,
        fps=fps,
        cpu_util_avg=cpu_avg,
        gpu_util_avg=gpu_avg,
        num_images=len(images),
        repeats=repeats,
        notes="end-to-end latency (preprocess+inference+postprocess)",
    )


def write_outputs(csv_path: Path, json_path: Path, rows: list[Metrics]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "mAP50",
                "latency_ms",
                "fps",
                "cpu_util_avg",
                "gpu_util_avg",
                "num_images",
                "repeats",
                "notes",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.model,
                    f"{r.map50:.4f}",
                    f"{r.latency_ms:.3f}",
                    f"{r.fps:.3f}",
                    f"{r.cpu_util_avg:.2f}",
                    f"{r.gpu_util_avg:.2f}",
                    r.num_images,
                    r.repeats,
                    r.notes,
                ]
            )

        speedup = rows[0].latency_ms / rows[1].latency_ms if rows[1].latency_ms > 0 else 0.0
        writer.writerow(["speedup(pt_over_trt)", "", f"{speedup:.3f}", "", "", "", "", "", ""])

    payload = {
        "results": [r.__dict__ for r in rows],
        "speedup_pt_over_trt": rows[0].latency_ms / rows[1].latency_ms if rows[1].latency_ms > 0 else 0.0,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    images = list_images(args.images_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {args.images_dir}")
    gt = build_ground_truth(images, args.labels_dir)

    trt_metrics = benchmark_trt(
        engine_path=args.trt_engine,
        images=images,
        gt=gt,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    pt_metrics = benchmark_pytorch(
        model_path=args.pt_model,
        images=images,
        gt=gt,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    rows = [pt_metrics, trt_metrics]
    write_outputs(args.output_csv, args.output_json, rows)

    for row in rows:
        print(
            f"{row.model}: mAP50={row.map50:.4f}, latency={row.latency_ms:.3f}ms, "
            f"FPS={row.fps:.3f}, CPU={row.cpu_util_avg:.2f}%, GPU={row.gpu_util_avg:.2f}%"
        )
    speedup = pt_metrics.latency_ms / trt_metrics.latency_ms if trt_metrics.latency_ms > 0 else 0.0
    print(f"speedup(pt_over_trt): {speedup:.3f}x")
    print(f"saved: {args.output_csv}")
    print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
