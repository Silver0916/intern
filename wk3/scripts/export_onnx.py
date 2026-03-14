#!/usr/bin/env python3
"""Export YOLO .pt model to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-model", type=Path, default=Path("wk3/models/original/yolov8n.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-onnx",
        type=Path,
        default=Path("wk3/models/original/yolov8n.onnx"),
        help="Final output path. Export is done by ultralytics then moved to this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.pt_model))

    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        device=args.device,
    )
    exported_path = Path(exported)
    if exported_path.resolve() != args.output_onnx.resolve():
        args.output_onnx.write_bytes(exported_path.read_bytes())
    print(f"ONNX exported: {args.output_onnx}")


if __name__ == "__main__":
    main()
