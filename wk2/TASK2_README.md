# Task2: 单目视觉里程计与无人机轨迹仿真

## 1. 目标
- 生成一段可复现的仿真无人机飞行图像序列与真值位姿。
- 使用 ORB + 本质矩阵 + RANSAC 的单目 VO 估计轨迹。
- 输出 2D/3D 轨迹图和位置误差曲线。

## 2. 代码结构
- `simulation_drone_sequence.py`: 生成仿真图像序列、GT 位姿 CSV、相机内参 YAML。
- `vo_monocular.py`: 执行单目视觉里程计并输出估计位姿和指标。
- `visualize_trajectory.py`: 绘制轨迹与误差图。
- `run_task2.py`: 一键执行全流程。

## 3. 依赖环境
- Python 3.10+
- `opencv-python`
- `numpy`
- `matplotlib`
- `pyyaml`

示例安装:

```bash
pip install opencv-python numpy matplotlib pyyaml
```

## 4. 一键运行

在仓库根目录执行:

```bash
python wk2/run_task2.py --output wk2/output --num_frames 180 --seed 42
```

## 5. 主要输出
- `wk2/output/frames/*.png`: 仿真图像序列
- `wk2/output/gt_poses.csv`: 仿真真值位姿
- `wk2/output/camera_intrinsics.yaml`: 相机内参
- `wk2/output/estimated_poses.csv`: VO 估计位姿
- `wk2/output/vo_pair_metrics.csv`: 每帧对 VO 过程指标
- `wk2/output/metrics_summary.csv`: 汇总指标
- `wk2/output/trajectory_2d.png`: 2D 轨迹图
- `wk2/output/trajectory_3d.png`: 3D 轨迹图
- `wk2/output/position_error.png`: 位置误差曲线

## 6. 算法流程
1. ORB 提取前后帧关键点与描述子。
2. BFMatcher(Hamming) + ratio test 筛选匹配点。
3. `findEssentialMat(..., RANSAC)` 估计本质矩阵。
4. `recoverPose` 恢复相邻帧相对位姿。
5. 使用 GT 相邻帧平移模长恢复单目尺度并累计位姿。
6. 统计成功/失败帧、匹配数量、轨迹误差并可视化。

## 7. 复跑 VO（跳过仿真）

```bash
python wk2/run_task2.py \
  --skip_sim \
  --output wk2/output \
  --frames_dir wk2/output/frames \
  --gt_csv wk2/output/gt_poses.csv \
  --intrinsics_yaml wk2/output/camera_intrinsics.yaml
```
