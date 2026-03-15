import cv2
from pathlib import Path
import numpy as np

def preprocess(img:Path | str) -> np.ndarray:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1]
    img= np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(preds: np.ndarray, conf_threshold: float = 0.25) -> int:
    # yolov5su output shape: (1, 84, 8400) — 4 bbox + 80 class scores, no objectness
    output = preds.reshape(84, -1)      # (84, 8400)
    class_scores = output[4:, :]        # (80, 8400)
    max_conf = class_scores.max(axis=0) # (8400,)
    mask = max_conf > conf_threshold
    return int(mask.sum())