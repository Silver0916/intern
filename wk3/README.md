# Week3 - AI模型嵌入式部署与优化

本目录提供任务三完整复现链路：
- 原始模型导出（PyTorch -> ONNX）
- TensorRT INT8 量化与引擎生成
- 原始模型与优化模型性能对比（mAP50/FPS/CPU/GPU）

## 1. 环境检查

```bash
source wk3/.venv/bin/activate
python wk3/scripts/check_env.py
```

## 2. 导出 ONNX

```bash
python wk3/scripts/export_onnx.py \
  --pt-model wk3/models/original/yolov8n.pt \
  --output-onnx wk3/models/original/yolov8n.onnx
```

## 3. 构建 TensorRT INT8 引擎

```bash
python wk3/scripts/build_trt_int8_engine.py \
  --onnx wk3/models/original/yolov8n.onnx \
  --engine wk3/models/optimized/yolov8n_int8.engine \
  --cache wk3/models/optimized/int8_calib.cache \
  --calib-dir /mnt/d/projs/datasets/coco8/images
```

## 4. 推理示例

```bash
python wk3/scripts/infer_pytorch.py --output wk3/data/predict_pt.jpg
python wk3/scripts/infer_onnx.py --output wk3/data/predict_onnx.jpg
```

## 5. 基准对比

```bash
python wk3/scripts/benchmark.py --repeats 8 --warmup 2
```

输出：
- `wk3/data/perf_compare.csv`
- `wk3/data/perf_details.json`

## 6. 报告与提交

- 报告：`wk3/reports/task3_report.md`
- 提交压缩包：`wk3/submission/task3_submission.zip`
