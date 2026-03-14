#!/usr/bin/env python3
"""Build TensorRT INT8 engine from YOLOv8 ONNX using entropy calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from yolo_utils import preprocess_bgr


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """TensorRT INT8 calibrator for fixed-shape images."""

    def __init__(self, calibration_images: list[Path], imgsz: int, cache_file: Path):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.index = 0
        self.batches = self._load_batches(calibration_images, imgsz)
        if not self.batches:
            raise ValueError("No calibration images found")
        self.batch_nbytes = int(self.batches[0].nbytes)
        err, ptr = cudart.cudaMalloc(self.batch_nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc failed: {err}")
        self.device_input = int(ptr)

    def __del__(self):
        if getattr(self, "device_input", None):
            cudart.cudaFree(self.device_input)
            self.device_input = None

    @staticmethod
    def _load_batches(image_paths: list[Path], imgsz: int) -> list[np.ndarray]:
        batches: list[np.ndarray] = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            x, _ = preprocess_bgr(img, imgsz=imgsz)
            batches.append(np.ascontiguousarray(x))
        return batches

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):  # noqa: ANN001 - TensorRT interface
        if self.index >= len(self.batches):
            return None
        batch = self.batches[self.index]
        self.index += 1
        err = cudart.cudaMemcpy(
            self.device_input,
            batch.ctypes.data,
            self.batch_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )[0]
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpy H2D failed: {err}")
        return [self.device_input]

    def read_calibration_cache(self):  # noqa: ANN001 - TensorRT interface
        if self.cache_file.exists():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):  # noqa: ANN001 - TensorRT interface
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("wk3/models/original/yolov8n.onnx"))
    parser.add_argument("--engine", type=Path, default=Path("wk3/models/optimized/yolov8n_int8.engine"))
    parser.add_argument("--cache", type=Path, default=Path("wk3/models/optimized/int8_calib.cache"))
    parser.add_argument("--calib-dir", type=Path, default=Path("/mnt/d/projs/datasets/coco8/images/val"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace-gb", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib_images = sorted(args.calib_dir.rglob("*.jpg")) + sorted(args.calib_dir.rglob("*.png"))
    if not calib_images:
        raise FileNotFoundError(f"No images in {args.calib_dir}")

    logger_level = trt.Logger.INFO if args.verbose else trt.Logger.WARNING
    logger = trt.Logger(logger_level)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = EntropyCalibrator(calib_images, args.imgsz, args.cache)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    print(f"engine saved: {args.engine} ({args.engine.stat().st_size} bytes)")
    print(f"calib cache: {args.cache}")


if __name__ == "__main__":
    main()
