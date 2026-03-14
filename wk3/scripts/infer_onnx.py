#!/usr/bin/env python3
"""Run ONNX inference for YOLOv8 and save visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import onnxruntime as ort

from yolo_utils import decode_yolov8_output, draw_detections, preprocess_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("wk3/models/original/yolov8n.onnx"))
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/mnt/d/projs/datasets/coco8/images/val/000000000036.jpg"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=Path("wk3/data/predict_onnx.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)

    x, meta = preprocess_bgr(image, imgsz=args.imgsz)
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    output = sess.run(None, {"images": x})[0]
    detections = decode_yolov8_output(output, meta, conf_thres=args.conf, iou_thres=args.iou)

    vis = draw_detections(image, detections)
    cv2.imwrite(str(args.output), vis)
    print(f"saved: {args.output}")
    print(f"detections: {len(detections)}")


if __name__ == "__main__":
    main()
