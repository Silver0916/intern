#!/usr/bin/env python3
"""Quick environment check for task3 pipeline."""

from __future__ import annotations

import importlib
import platform
from pathlib import Path


def _safe_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, str(exc)


def main() -> None:
    print("=== System ===")
    print(f"platform: {platform.platform()}")
    print(f"python: {platform.python_version()}")
    print(f"cwd: {Path.cwd()}")

    print("\n=== Packages ===")
    torch, torch_err = _safe_import("torch")
    onnxruntime, ort_err = _safe_import("onnxruntime")
    tensorrt, trt_err = _safe_import("tensorrt")
    pynvml, nvml_err = _safe_import("pynvml")
    ultralytics, yolo_err = _safe_import("ultralytics")

    print(f"torch: {getattr(torch, '__version__', 'N/A')} ({'OK' if torch else torch_err})")
    print(
        "onnxruntime: "
        f"{getattr(onnxruntime, '__version__', 'N/A')} ({'OK' if onnxruntime else ort_err})"
    )
    print(
        "tensorrt: "
        f"{getattr(tensorrt, '__version__', 'N/A')} ({'OK' if tensorrt else trt_err})"
    )
    print(
        "ultralytics: "
        f"{getattr(ultralytics, '__version__', 'N/A')} ({'OK' if ultralytics else yolo_err})"
    )
    print(f"pynvml: {'OK' if pynvml else nvml_err}")

    if onnxruntime:
        print(f"onnxruntime providers: {onnxruntime.get_available_providers()}")

    print("\n=== CUDA ===")
    if torch:
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch device count: {torch.cuda.device_count()}")
            print(f"torch device[0]: {torch.cuda.get_device_name(0)}")
            print(f"torch capability[0]: {torch.cuda.get_device_capability(0)}")

    if pynvml:
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            print(f"nvml device count: {count}")
            if count:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                print(f"nvml device[0]: {name}")
            pynvml.nvmlShutdown()
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"nvml error: {exc}")


if __name__ == "__main__":
    main()
