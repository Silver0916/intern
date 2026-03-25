# 视觉-惯性里程计（VIO）技术报告

## 1. 任务目标

本周任务目标是基于公开无人机数据集实现一个松耦合（Loosely-coupled）视觉-惯性里程计系统，将第二周单目视觉里程计（VO）的位置信息与 IMU 测量数据进行融合，并与 Ground Truth 做定量对比分析。

本次实验使用 EuRoC MAV Dataset 中 `MH_01_easy` 序列，重点完成以下内容：

- 学习 IMU 的工作原理、测量输出与误差来源
- 理解基于扩展卡尔曼滤波（EKF）的误差状态建模方法
- 实现 VO + IMU 的松耦合融合程序
- 对比纯 VO、融合后 VIO 与 Ground Truth 的轨迹差异
- 分析 VIO 在快速运动、弱纹理场景中的优势与当前实现的局限

## 2. 数据集与实验环境

### 2.1 数据集

- 数据集：EuRoC MAV Dataset
- 序列：`MH_01_easy`
- 数据根目录（WSL 路径）：

```text
/mnt/d/intern_dataset/machine_hall/machine_hall/MH_01_easy/MH_01_easy
```

本实验用到的数据文件包括：

- IMU 数据：`mav0/imu0/data.csv`
- 相机帧索引：`mav0/cam0/frame_index.csv`
- 第二周 VO 输出：`mav0/cam0/estimated_poses.csv`
- 对齐 GT：`mav0/cam0/gt_poses_aligned.csv`
- Ground Truth 全状态：`mav0/state_groundtruth_estimate0/data.csv`

### 2.2 实验环境

- 系统：WSL2 Ubuntu
- Python：3.12
- 主要依赖：`numpy`
- 运行解释器：`wk3/.venv/bin/python`

## 3. 理论基础

### 3.1 IMU 测量模型

IMU 主要输出角速度和线加速度：

- 陀螺仪输出：`omega_m = omega + b_g + n_g`
- 加速度计输出：`a_m = a + b_a + n_a`

其中：

- `b_g`、`b_a` 分别为陀螺仪和加速度计偏置
- `n_g`、`n_a` 分别为测量噪声

在状态传播中，需要先去除偏置，再结合姿态将机体系加速度转到世界系，并叠加重力项完成位置与速度积分。

### 3.2 松耦合 VIO 思路

本实验采用松耦合方案，核心思想是：

1. IMU 以高频率执行预测步骤，更新位置、速度、姿态
2. VO 以低频率提供位置观测，作为 EKF 的测量更新
3. 通过滤波器在预测与观测之间平衡两类传感器信息

与紧耦合方案相比，松耦合实现更简单，适合作业阶段快速完成从 VO 到 VIO 的系统搭建。

### 3.3 误差状态 EKF

本实现使用 15 维误差状态：

```text
delta_x = [delta_p, delta_v, delta_theta, delta_ba, delta_bg]
```

其中：

- `delta_p`：位置误差（3 维）
- `delta_v`：速度误差（3 维）
- `delta_theta`：姿态小角度误差（3 维）
- `delta_ba`：加速度计偏置误差（3 维）
- `delta_bg`：陀螺仪偏置误差（3 维）

测量模型采用最小可行版本：

```text
z_vo = p + noise
```

即仅使用 VO 的位置观测对滤波器进行更新。

## 4. 关键实现

本周新增核心脚本：

- `wk3_mannual/wk4/vio_efk.py`

主要完成了以下模块：

### 4.1 数据读取与格式兼容

脚本兼容 EuRoC 原始列名和第二周 VO 输出列名，能够直接读取：

- IMU CSV
- GT CSV
- `frame_index.csv`
- `estimated_poses.csv`

同时跳过 VO 中 `failure` 状态的帧，避免把明显退化的 VO 结果当成有效观测。

### 4.2 姿态表示与传播

采用四元数表示姿态，实现了：

- 四元数归一化
- 四元数乘法
- 小角度向四元数的映射
- 四元数到旋转矩阵的转换

在传播阶段，先利用陀螺仪更新姿态，再利用姿态将加速度投影到世界系，完成位置和速度积分。

### 4.3 协方差传播与数值稳定

本作业版本采用保守化的最小可行过程模型：

- 位置与速度通过离散运动学传播
- 姿态与偏置主要依靠过程噪声维持不确定性
- 观测更新前后对协方差做对称化和正定投影
- 对创新协方差 `S` 做数值正则，避免矩阵奇异

这样做的目的不是追求最强精度，而是优先保证作业版本能在完整序列上稳定跑通。

### 4.4 初始化策略

系统初始化时：

- 位置、速度初始化为零
- 姿态通过前若干 IMU 样本估计重力方向
- 若提供 GT 全状态，则读取首帧偏置作为初始化参考

## 5. 运行方式

执行命令如下：

```bash
wk3/.venv/bin/python wk3_mannual/wk4/vio_efk.py \
  --imu_csv /mnt/d/intern_dataset/machine_hall/machine_hall/MH_01_easy/MH_01_easy/mav0/imu0/data.csv \
  --frame_index_csv /mnt/d/intern_dataset/machine_hall/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/frame_index.csv \
  --vo_est_csv /mnt/d/intern_dataset/machine_hall/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/estimated_poses.csv \
  --gt_csv /mnt/d/intern_dataset/machine_hall/machine_hall/MH_01_easy/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
  --out_csv wk3_mannual/wk4/output/mh01_vio_estimated_poses.csv
```

生成结果：

- VIO 轨迹：`wk3_mannual/wk4/output/mh01_vio_estimated_poses.csv`
- VIO 轨迹图：`wk3_mannual/wk4/output/plots/`
- VO 基线轨迹图：`wk3_mannual/wk4/output/vo_plots/`
- 对比指标：`wk3_mannual/wk4/output/compare_metrics.csv`

## 6. 实验结果

### 6.1 定量结果

对 VO 和 VIO 分别与 GT 做 Sim(3) 对齐，并在 2910 个共同帧上统计位置误差，结果如下：

| 方法 | 共同帧数 | 平均误差/m | 中位误差/m | RMSE/m | 终点误差/m | 最大误差/m |
|---|---:|---:|---:|---:|---:|---:|
| VO | 2910 | 2.0580 | 1.4468 | 2.4560 | 6.8382 | 6.8382 |
| VIO | 2910 | 3.7144 | 3.4498 | 4.0723 | 5.6179 | 7.1097 |

从本次结果看：

- VIO 的终点误差从 `6.8382 m` 降低到 `5.6179 m`
- 但 VIO 的平均误差和 RMSE 均高于纯 VO

### 6.2 轨迹结果

可直接插入以下图片到最终 PDF：

- 纯 VO 俯视图：`output/vo_plots/trajectory_2d.png`
- 纯 VO 三维轨迹：`output/vo_plots/trajectory_3d.png`
- 纯 VO 误差曲线：`output/vo_plots/position_error.png`
- VIO 俯视图：`output/plots/trajectory_2d.png`
- VIO 三维轨迹：`output/plots/trajectory_3d.png`
- VIO 误差曲线：`output/plots/position_error.png`

示意图如下：

![VO top view](output/vo_plots/trajectory_2d.png)

![VIO top view](output/plots/trajectory_2d.png)

## 7. 结果分析

### 7.1 当前 VIO 的改进点

本实现虽然没有在全程平均误差上超过 VO，但仍体现出以下特点：

- 在终点误差上略优于纯 VO，说明 IMU 传播对长时间漂移具备一定抑制作用
- 当 VO 某些帧失败时，系统仍可依赖 IMU 完成短时状态传播，避免轨迹完全中断
- 融合框架已经建立，后续可以继续加入更强的观测模型和更准确的过程模型

### 7.2 为什么平均误差没有提升

本次 VIO 没有全面优于 VO，主要原因包括：

1. 当前方案只使用了 VO 位置观测，没有融合 VO 姿态、速度或图像特征级信息，观测维度偏弱。
2. 在当前版本中，已经在 `wk4` 内部补充了基于相邻图像对的视觉相对旋转估计，并将其作为姿态观测加入 EKF；但由于该姿态观测是作业级增量实现，仍缺少更严格的协方差建模和更稳定的相对姿态链路。
3. 过程模型为了保证数值稳定做了保守化处理，状态间耦合被弱化，滤波增益受限。
4. 偏置估计与外参误差没有进一步优化，导致 IMU 传播误差仍会累积。
5. 第二周 VO 的尺度恢复已经借助了 GT 速度/位姿信息，因此该 VO 基线本身较强，VIO 要在此基础上继续明显超越会更困难。
6. 参数尚未针对 `MH_01_easy` 进行系统调参，测量噪声与过程噪声设定偏经验值。

### 7.3 特定场景下的优势与局限

优势：

- 快速转弯或运动突变时，IMU 能提供高频动态信息，理论上比纯 VO 更连续
- 图像短时模糊、遮挡或特征点不足时，IMU 可提供短期补偿

局限：

- 当 IMU 偏置建模不准确时，积分漂移会迅速累积
- 当前松耦合实现只能做较弱修正，无法充分利用图像几何约束
- 若 VO 自身尺度或局部位置已有较大误差，融合后也可能把偏差带入系统

## 8. 后续改进方向

若继续完善本 VIO 系统，可优先从以下方面改进：

1. 将 VO 姿态也纳入观测更新，而不仅仅使用位置。
2. 建立更完整的误差状态转移矩阵，恢复位置、速度、姿态、偏置之间的耦合关系。
3. 利用相机-IMU 外参和预积分方法，提高预测与更新的一致性。
4. 对多组 EuRoC 序列进行调参与对比，避免单序列结论偶然性。
5. 进一步尝试紧耦合 VIO，以图像特征重投影误差为更新量。

## 9. 提交产物清单

- 核心代码：`wk3_mannual/wk4/vio_efk.py`
- VIO 输出：`wk3_mannual/wk4/output/mh01_vio_estimated_poses.csv`
- 对比指标：`wk3_mannual/wk4/output/compare_metrics.csv`
- VO 图像：`wk3_mannual/wk4/output/vo_plots/`
- VIO 图像：`wk3_mannual/wk4/output/plots/`

## 10. 结论

本周完成了基于 EuRoC `MH_01_easy` 的松耦合 EKF-VIO 实现，系统能够直接读取第二周 VO 输出与 EuRoC IMU 数据，完成传播、更新、保存与可视化全流程。

从实验结果看，当前作业版本已经成功实现“视觉 + IMU”融合，但精度收益仍有限：终点误差略有改善，而平均误差和 RMSE 尚未优于纯 VO。这说明本系统已经具备完整技术闭环，但若追求更高精度，还需要继续改进观测模型、过程模型以及参数设置。
