from pathlib import Path
import time
import tensorrt as trt
import torch
import pandas as pd
import psutil
from ultralytics import YOLO
from ultralytics.utils import DATASETS_DIR
from utils import preprocess, postprocess


def load_engine(engine_path: Path) -> trt.ICudaEngine:

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    data = engine_path.read_bytes()
    return runtime.deserialize_cuda_engine(data)

def allocate_buffers(engine):
    inputs, outputs,bindings = [],[],[]
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        tensor = torch.zeros(tuple(shape), dtype=torch.float32).cuda()
        bindings.append(tensor.data_ptr())
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(tensor)
        else:
            outputs.append(tensor)
    return inputs, outputs, bindings

def collect_images(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def infer_one_image(context, inputs, outputs, bindings, img_path) -> tuple[float, int, float, float]:
    img = preprocess(img_path)
    inputs[0].copy_(torch.from_numpy(img))

    engine = context.engine
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        context.set_tensor_address(name, bindings[i])

    psutil.cpu_percent(interval=None)
    start_time = time.time()
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    end_time = time.time()

    inference_time = end_time - start_time
    cpu_percent = psutil.cpu_percent(interval=None)
    gpu_mem_mb = torch.cuda.memory_allocated() / 1024 ** 2

    output_np = outputs[0].cpu().numpy()
    num_det = postprocess(output_np)

    return inference_time, num_det, cpu_percent, gpu_mem_mb


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    engine_path = project_root / "engines" / "yolov5su_int8.engine"
    img_dir = Path(DATASETS_DIR) / "coco8" / "images" / "val"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings = allocate_buffers(engine)

    rows = []
    for img_path in collect_images(img_dir):
        inference_time, num_det, cpu_pct, gpu_mb = infer_one_image(
            context, inputs, outputs, bindings, img_path
        )
        rows.append({
            "image":          img_path.name,
            "inference_ms":   round(inference_time * 1000, 2),
            "fps":            round(1 / inference_time, 2),
            "num_detections": num_det,
            "cpu_percent":    cpu_pct,
            "gpu_mem_mb":     round(gpu_mb, 2),
        })

    trt_model = YOLO(str(engine_path), task="detect")
    trt_metrics = trt_model.val(data="coco8.yaml", verbose=False)
    map50 = trt_metrics.box.map50

    for row in rows:
        row["map50"] = round(map50, 4)

    df = pd.DataFrame(rows)
    out_path = results_dir / "trt_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(rows)} rows -> {out_path}")
    print(f"mAP@0.5: {map50:.4f}  |  avg FPS: {df['fps'].mean():.1f}  |  avg CPU: {df['cpu_percent'].mean():.1f}%  |  avg GPU: {df['gpu_mem_mb'].mean():.1f} MB")


if __name__ == "__main__":
    main()
