# 训练自己数据的方案（GaussianPretrain）

## 1. 目标与输入输出
- **目标**：用带有多相机图像与激光点云的自有数据在 GaussianPretrain 框架中执行多模态预训练，使模型学习到统一 3D 表达并可复用于下游检测/重建任务。
- **输入**：同步的 LiDAR + 多相机数据、对应的时间戳、传感器标定、三维标注框、可选的历史 sweeps。
- **输出**：符合 `NuScenesSweepDataset` 期望的 unified `*.pkl` 标注文件、定制配置文件、预训练权重与日志。

## 2. 数据采集与标定
### 2.1 采集要求
- LiDAR 至少 10Hz，建议保留历史 sweeps（`lidar_sweep_num` 默认 10）；相机需涵盖 360° 或任务相关扇区，同时提供相邻帧（`cam_sweep_num` 默认 1，可增大以引入视频上下文）。
- 每条样本应包含：激光点云（`.bin` 或 `.pcd`）、对应 6~8 个相机的 RGB 图像、标注框与实例属性（类别、速度、可见性等）。
- 确保统一的时间系（GPS/PTS）。若存在触发延迟，记录 `timestamp` 方便放入 `info['timestamp']`。

### 2.2 标定矩阵
- 需要离线求解传感器到车体（ego）的旋转和平移：
  $$
  \mathbf{T}_{\text{lidar}\rightarrow\text{ego}} =
  \begin{bmatrix}
  \mathbf{R}_{LE} & \mathbf{t}_{LE} \\
  0 & 1
  \end{bmatrix},\quad
  \mathbf{T}_{\text{cam}\rightarrow\text{ego}} =
  \begin{bmatrix}
  \mathbf{R}_{CE} & \mathbf{t}_{CE} \\
  0 & 1
  \end{bmatrix}
  $$
- 推导 `lidar2cam`：
  $$
  \mathbf{T}_{\text{lidar}\rightarrow\text{cam}} =
  (\mathbf{T}_{\text{cam}\rightarrow\text{ego}})^{-1}\,\mathbf{T}_{\text{lidar}\rightarrow\text{ego}}
  $$
  并存入 `info['cams'][cam]['sensor2lidar_rotation']` / `sensor2lidar_translation`。
- 像素投影需提供相机内参 $\mathbf{K}$：
  $$
  \mathbf{p}_{\text{img}} \sim \mathbf{K}\,\mathbf{T}_{\text{lidar}\rightarrow\text{cam}}\,\mathbf{p}_{\text{lidar}}
  $$
  其中 `cam_intrinsic` 填写 4×4 齐次矩阵（见 `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py`）。

### 2.3 目录组织
推荐模仿 nuScenes 结构，便于沿用转换脚本：
```text
data/custom_dataset/
  samples/
    CAM_FRONT/xxxx.jpg
    CAM_FRONT_RIGHT/...
    LIDAR_TOP/xxxx.bin
  sweeps/
    LIDAR_TOP/...
    CAM_FRONT/...
  v1.0-custom/
    scene.json            # 元数据
    sample.json
    sample_data.json
    ego_pose.json
    calibrated_sensor.json
    instance.json / sample_annotation.json
```
若无 JSON，可用任意自定义格式，只需在生成 `info` 时提供同样信息（token、路径、标定）。

## 3. 生成 unified 标注文件
### 3.1 统一坐标与时间轴
1. 选择全局坐标（如汽车底盘中心）。记录 ego-to-global 变换：
   $$
   \mathbf{T}_{\text{ego}\rightarrow\text{global}} =
   \begin{bmatrix}
   \mathbf{R}_{EG} & \mathbf{t}_{EG} \\
   0 & 1
   \end{bmatrix}
   $$
   保存到 `info['ego2global_rotation']`、`info['ego2global_translation']`。
2. 对每个标注框，将中心、尺寸、朝向统一为 LiDAR 坐标，速度 $(v_x,v_y)$ 可选：
   $$
   \text{gt\_boxes} = [x,y,z,w,l,h,\sin\theta,\cos\theta,v_x,v_y]
   $$

### 3.2 `info` 字段对照
每条样本需最少包含：
```python
info = dict(
    token='unique_id',
    timestamp=ts_microseconds,
    lidar_path='samples/LIDAR_TOP/xxx.bin',
    sweeps=[{ 'data_path': ..., 'time_lag': ... }, ...],
    cams={
        'CAM_FRONT': {
            'data_path': 'samples/CAM_FRONT/xxx.jpg',
            'sensor2lidar_rotation': ...,   # 3x3
            'sensor2lidar_translation': ...,# 3
            'cam_intrinsic': ...,           # 4x4
        },
        ...
    },
    cam_sweeps_info={
        'CAM_FRONT': [
            {
                'data_path': 'sweeps/CAM_FRONT/xxx.jpg',
                'timestamp': ts_prev,
                'sensor2lidar_rotation': ...,
                'sensor2lidar_translation': ...,
                'cam_intrinsic': ...,
            },
            ...
        ],
        ...
    },
    gt_boxes=np.array([...], dtype=np.float32),
    gt_names=np.array([...], dtype='<U32'),
    num_lidar_pts=np.array([...], dtype=np.int64),
    valid_flag=np.ones(len(gt_boxes), dtype=bool),
)
```
- `cam_sweeps_info` 是 GaussianPretrain 额外需要的字段，对应 `LoadMultiViewMultiSweepImageFromFiles`。
- `sweeps` 为 LiDAR 历史帧；若无，可复制当前帧并将 `time_lag=0.0`。

### 3.3 转换脚本与工具链
1. **改造 UVTR 转换器**（推荐）：
   - 克隆 UVTR 仓库到 `../UVTR`，设置 `PYTHONPATH`。
   - 参照 `create_data.py nuscenes` 流程，将数据读取逻辑替换为自有数据（封装成 `CustomUnifiedBuilder`）。
   - 产出 `custom_unified_infos_{train,val}.pkl`。
2. **直接编写脚本**：
   ```python
   from pathlib import Path
   import mmcv

   infos = []
   for sample in load_meta():
       info = build_info(sample)  # 生成上述字段
       infos.append(info)
   mmcv.dump(infos, 'data/custom/custom_unified_infos_train.pkl')
   ```
3. **关键细节**：
   - `time_lag` 使用秒，GaussianPretrain 在构造 `cam_sweeps_time` 时会减去首帧并归零。
   - `sensor2lidar_rotation` 需要 `w, x, y, z` 四元数或 3×3 矩阵？在本项目中使用 3×3 矩阵（参见 `nuscenes_dataset.py` 290 行的 `np.linalg.inv`）。
   - 图像路径应为相对 `data_root` 的字符串，否则 `LoadMultiViewMultiSweepImageFromFiles` 无法找到文件。

### 3.4 自检
- **字段完整性**：使用 `python - <<'PY'` 读取 `infos[0]`，确保 `cam_sweeps_info`、`lidar_path`、`gt_boxes` 均存在。
- **可视化检查**：
  ```bash
  python tools/misc/browse_dataset.py \
      projects/configs/gaussianpretrain/gp_custom.py \
      --task multi_modality-det \
      --output-dir tmp/browse_custom \
      --cfg-options \
      data.train.ann_file=data/custom/custom_unified_infos_train.pkl \
      data.train.data_root=data/custom/
  ```
  若能正确投影 3D 框到图像，即 `lidar2img` 与内参一致。

## 4. 调整 GaussianPretrain 配置
1. 复制基础配置：
   ```bash
   cp projects/configs/gaussianpretrain/gp_0.075_convnext.py \
      projects/configs/gaussianpretrain/gp_custom_convnext.py
   ```
2. 修改关键信息：
   ```python
   data_root = 'data/custom/'
   class_names = ['car', 'truck', 'bus', ...]  # 与自有标签一致
   cam_sweep_num = 2  # 例如需要历史 1 帧
   lidar_sweep_num = 5

   point_cloud_range = [-x_max, -y_max, z_min, x_max, y_max, z_max]
   unified_voxel_size = [vx, vy, vz]
   unified_voxel_shape = [
       int((point_cloud_range[3]-point_cloud_range[0]) / vx),
       int((point_cloud_range[4]-point_cloud_range[1]) / vy),
       int((point_cloud_range[5]-point_cloud_range[2]) / vz),
   ]
   ```
   - 例如城区场景可设 `point_cloud_range=[-60,-60,-5,60,60,3]`，`unified_voxel_size=[0.6,0.6,1.6]`。
   - 若仅采集前向视角，可减小 `y` 范围以节省显存。
3. 更新数据路径：
   ```python
   data = dict(
       samples_per_gpu=1,
       workers_per_gpu=4,
       train=dict(
           ann_file=data_root + 'custom_unified_infos_train.pkl',
           data_root=data_root,
       ),
       val=dict(ann_file=data_root + 'custom_unified_infos_val.pkl'),
       test=dict(ann_file=data_root + 'custom_unified_infos_val.pkl'),
   )
   ```
4. 若传感器数量 ≠ 6，需要将 `pts_bbox_head.cam_nums` 以及 `LoadCameraParam(cam_nums=...)` 改为实际数目，并确保 `info['cams']` 字典中顺序与 `img_order` 一致。
5. 调整射线采样：`ray_sampler_cfg.close_radius/far_radius` 与数据尺度相关；可用 $r=\sqrt{x^2+y^2}$ 的 95% 分位作为 `far_radius`。

## 5. 训练流程
1. **准备权重与依赖**：确认 `convnextS_1kpretrained_official_style.pth` 在 `checkpoints/`，并按 `docs/环境安装` 完成依赖。
2. **单机单卡（调试）**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 \
   python tools/train.py \
     projects/configs/gaussianpretrain/gp_custom_convnext.py \
     --cfg-options log_config.interval=10
   ```
3. **单机多卡**：
   ```bash
   bash tools/dist_train.sh \
     projects/configs/gaussianpretrain/gp_custom_convnext.py 8 \
     --cfg-options data.samples_per_gpu=1 data.workers_per_gpu=2
   ```
4. **恢复训练**：
   ```bash
   python tools/train.py ... --resume-from work_dirs/gp_custom/latest.pth
   ```
5. **日志与评估**：默认 `evaluation.interval=total_epochs`；如需中途验证可设置 `evaluation=dict(interval=3)`。

## 6. 调优建议
- **体素/视锥分辨率**：
  $$
  N_x = \frac{x_{\max}-x_{\min}}{v_x},\quad
  N_y = \frac{y_{\max}-y_{\min}}{v_y},\quad
  N_z = \frac{z_{\max}-z_{\min}}{v_z}
  $$
  当显存不足时增大 $v_x,v_y$（降低分辨率）；当追求细节时减小 $v_z$ 并同步调低 `frustum_size[2]`。
- **相机/激光 sweeps**：更多 sweeps 有利于时序一致性，但 I/O 与显存线性增加。经验：`cam_sweep_num=2`、`lidar_sweep_num=5` 时 batch size=1 需要 ≈24GB。
- **损失项权重**：自有数据若没有高质量 RGB，建议降低 `rgb_loss` 权重、提高 `depth_loss` 或 `occ_loss`。
- **类别不平衡**：使用 `data.train.filter_empty_gt=False` 以保留稀有类，同时可在 `loss_cfg.weights` 中提升次要任务的比重。
- **学习率与 warmup**：总样本 < nuScenes（28k）时，可按比例缩短 `total_epochs`，并令 $\text{lr}_{\text{new}} = \text{lr}_{\text{base}} \times \frac{N_{\text{custom}}}{28130}$。

## 7. 验证与常见问题
- **`cam_sweeps_info` 缺失**：加载时报 `KeyError`，需确保转写脚本将历史帧及标定完整写入。
- **路径错误**：若 log 报 `FileNotFoundError: data/...jpg`，请确认 `data_root` 与 `info['cams'][cam]['data_path']` 的拼接真实存在，可使用 `os.path.exists` 抽检。
- **标定错位**：使用 `browse_dataset.py` 查看投影，若框偏移统一方向，多半是 `[x,y,z]` 单位或顺序不一致；逐项打印 `lidar2img` 的第 4 列确认平移量。
- **点云范围不匹配**：当点云被 `PointsRangeFilter` 完全裁掉时（`num_lidar_pts` 全零），请扩大 `point_cloud_range`。
- **显存不足**：降低 `samples_per_gpu`，或将 `unified_voxel_size=[0.8,0.8,2.0]`，同时调小 `ray_sampler_cfg.point_nsample`。

完成上述步骤后，即可在自有多模态数据上稳定运行 GaussianPretrain 的预训练流程。
