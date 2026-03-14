# AI模型嵌入式部署与优化报告

## 1. 任务目标

本任务围绕资源受限平台的模型部署，完成以下目标：
- 选择轻量模型并研究 INT8 量化对性能与精度影响
- 使用 TensorRT 进行解析、量化和引擎构建
- 在同一测试集上对比原始模型与优化模型的速度、资源占用和精度
- 给出 Jetson 端部署验证方案

本次实现模型为 `YOLOv8n`（轻量检测模型，与 YOLOv5s 同类轻量级目标检测路线）。

## 2. 实验环境

- 日期：2026-03-13
- 系统：WSL2 Ubuntu (Linux 6.6)
- Python：3.12.3
- GPU：NVIDIA GeForce RTX 5080 Laptop GPU
- 关键依赖：
  - torch 2.5.1+cu121
  - ultralytics 8.3.0
  - tensorrt 10.15.1.29
  - onnxruntime 1.24.3
- 数据集：`coco8`（验证集 4 张图）

## 3. 技术路线与执行步骤

### 步骤一：模型与量化方案

- 原始模型：`yolov8n.pt`
- 量化方案：TensorRT PTQ (Post-Training Quantization) + INT8 熵校准
- 校准数据：`coco8/images`（train+val）

### 步骤二：模型导出与 TensorRT 优化

1. PyTorch 导出 ONNX：
```bash
python wk3/scripts/export_onnx.py
```

2. TensorRT INT8 引擎生成：
```bash
python wk3/scripts/build_trt_int8_engine.py \
  --onnx wk3/models/original/yolov8n.onnx \
  --engine wk3/models/optimized/yolov8n_int8.engine \
  --cache wk3/models/optimized/int8_calib.cache \
  --calib-dir /mnt/d/projs/datasets/coco8/images
```

### 步骤三：原始模型 vs TensorRT 优化模型对比

执行：
```bash
python wk3/scripts/benchmark.py --repeats 8 --warmup 2
```

输出文件：
- `wk3/data/perf_compare.csv`
- `wk3/data/perf_details.json`

本次结果（端到端时延：预处理+推理+后处理）：

| 模型 | mAP50 | 延迟(ms) | FPS | CPU占用(%) | GPU占用(%) |
|---|---:|---:|---:|---:|---:|
| PyTorch-CPU (yolov8n.pt) | 0.8264 | 33.511 | 29.841 | 27.45 | 6.83 |
| TensorRT-INT8 (yolov8n_int8.engine) | 0.1667 | 7.071 | 141.415 | 2.13 | 13.67 |

综合加速比（延迟维度）：`4.739x`

### 步骤四：Jetson 实机部署验证方案

本机完成了完整 PC 端优化与验证。Jetson 板端建议按以下步骤执行：

1. 拷贝 `yolov8n_int8.engine` 与推理脚本到 Jetson
2. 使用 Jetson 对应 TensorRT 版本重新校准/重建引擎（推荐）
3. 在同一测试集运行 `benchmark.py` 的 TensorRT 部分
4. 记录并对比：
   - FPS
   - 平均延迟
   - GPU 占用与功耗（`tegrastats`）
   - mAP50 变化

## 4. 结果分析

1. INT8 TensorRT 在速度和 CPU 占用上明显优于原始 PyTorch CPU 推理。
2. 当前 INT8 模型在 `coco8` 上精度下降明显，主要原因：
   - 校准数据规模极小（仅 8 张图）
   - TensorRT 10.x 对传统 calibrator 路径已标注 deprecated，建议转显式量化链路
   - 本环境下 PyTorch 与 `sm_120` 存在 CUDA kernel 兼容问题，原始模型未在 GPU 路径做公平对照
3. 若在正式项目中落地，建议：
   - 扩大校准集（至少几百张代表性图像）
   - 使用更稳定的显式量化/量化感知训练（QAT）
   - 在目标 Jetson 设备上重新生成引擎，避免跨平台偏差

## 5. 提交产物清单

- 报告：
  - `wk3/reports/task3_report.md`
- 优化前后模型：
  - `wk3/models/original/yolov8n.pt`
  - `wk3/models/original/yolov8n.onnx`
  - `wk3/models/optimized/yolov8n_int8.engine`
  - `wk3/models/optimized/int8_calib.cache`
- 量化与推理脚本：
  - `wk3/scripts/build_trt_int8_engine.py`
  - `wk3/scripts/benchmark.py`
  - `wk3/scripts/export_onnx.py`
  - `wk3/scripts/infer_pytorch.py`
  - `wk3/scripts/infer_onnx.py`
- 性能数据：
  - `wk3/data/perf_compare.csv`
  - `wk3/data/perf_details.json`
