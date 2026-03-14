#!/usr/bin/env python3
"""Utility helpers for YOLOv8 ONNX/TensorRT postprocess and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class LetterboxMeta:
    ratio: float
    pad_w: float
    pad_h: float
    original_h: int
    original_w: int


def letterbox(image: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, LetterboxMeta]:
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad_w, new_unpad_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    dw = new_shape[1] - new_unpad_w
    dh = new_shape[0] - new_unpad_h
    dw /= 2
    dh /= 2
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))

    out = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    meta = LetterboxMeta(ratio=r, pad_w=dw, pad_h=dh, original_h=h, original_w=w)
    return out, meta


def preprocess_bgr(image: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    lb, meta = letterbox(image, (imgsz, imgsz))
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))[None]
    return np.ascontiguousarray(chw), meta


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    return boxes


def scale_boxes_to_original(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= meta.pad_w
    boxes[:, [1, 3]] -= meta.pad_h
    boxes[:, :4] /= meta.ratio
    return clip_boxes(boxes, meta.original_w, meta.original_h)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = np.clip(area1 + area2 - inter, 1e-9, None)
    return inter / union


def class_wise_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thres: float = 0.45,
) -> np.ndarray:
    keep: list[int] = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        order = idx[np.argsort(scores[idx])[::-1]]
        while len(order) > 0:
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break
            ious = iou_one_to_many(boxes[i], boxes[order[1:]])
            order = order[1:][ious <= iou_thres]
    return np.array(keep, dtype=np.int64)


def decode_yolov8_output(
    output0: np.ndarray,
    meta: LetterboxMeta,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> list[dict[str, float | int | np.ndarray]]:
    pred = output0[0].T  # (8400, 84)
    if pred.size == 0:
        return []

    class_scores = pred[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep_mask = scores >= conf_thres
    if not keep_mask.any():
        return []

    boxes = xywh2xyxy(pred[keep_mask, :4])
    class_ids = class_ids[keep_mask]
    scores = scores[keep_mask]
    boxes = scale_boxes_to_original(boxes, meta)

    keep = class_wise_nms(boxes, scores, class_ids, iou_thres)
    dets: list[dict[str, float | int | np.ndarray]] = []
    for i in keep:
        dets.append({"class_id": int(class_ids[i]), "score": float(scores[i]), "bbox": boxes[i]})
    return dets


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, float | int | np.ndarray]],
    color: tuple[int, int, int] = (50, 220, 50),
) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"].astype(int)
        score = det["score"]
        cls = det["class_id"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{cls}:{score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def load_yolo_labels_for_image(label_path: Path, image_w: int, image_h: int) -> list[tuple[int, np.ndarray]]:
    if not label_path.exists():
        return []
    anns: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        cls_id = int(cls)
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        x1 = (cx - w / 2) * image_w
        y1 = (cy - h / 2) * image_h
        x2 = (cx + w / 2) * image_w
        y2 = (cy + h / 2) * image_h
        anns.append((cls_id, np.array([x1, y1, x2, y2], dtype=np.float32)))
    return anns

