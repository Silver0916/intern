#!/usr/bin/env python3
"""Run inference using PyTorch YOLO model and save visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("wk3/models/original/yolov8n.pt"))
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/mnt/d/projs/datasets/coco8/images/val/000000000036.jpg"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=Path("wk3/data/predict_pt.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    result = model.predict(
        source=str(args.image),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]

    plotted = result.plot()
    cv2.imwrite(str(args.output), plotted)
    print(f"saved: {args.output}")
    print(f"detections: {len(result.boxes)}")


if __name__ == "__main__":
    main()
