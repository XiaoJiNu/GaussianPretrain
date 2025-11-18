# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GaussianPretrain is a research codebase that introduces 3D Gaussian Splatting technology into vision pre-training for autonomous driving. The project achieves improvements across 3D object detection, HD map reconstruction, and occupancy prediction tasks.

**Key Technology**: Uses 3D Gaussian Splatting to generate learnable 3D Gaussian anchors from multi-view images with LiDAR depth guidance, then reconstructs RGB, Depth, and Occupancy signals.

## Architecture

### Core Framework
- Built on **MMDetection3D v0.17.3** and **PyTorch v1.9.1**
- Plugin-based architecture centered at `projects/mmdet3d_plugin/`
- Custom CUDA operations for performance-critical components

### Main Components

**Detector Models** (`projects/mmdet3d_plugin/models/detectors/`)
- `uvtr_gp.py`: UVTR with GaussianPretrain integration
- `uvtr_dn.py`: UVTR with Denoising training
- `bevformer_pretrain.py`: BEVFormer with pretraining
- `panoocc_pretrain.py`: PanoOCC with pretraining

**Dense Heads** (`projects/mmdet3d_plugin/models/dense_heads/`)
- `gaussian_head.py`: Core GaussianHead implementing 3D Gaussian Splatting
  - Creates voxel grids in 3D space
  - Implements ray sampling for multi-view reconstruction
  - Manages Gaussian parameter regression
- `render_head.py`: Rendering utilities for reconstruction
- `uvtr_head.py` / `uvtr_dn_head.py`: Detection heads for UVTR variants

**Gaussian Splatting Pipeline** (`projects/mmdet3d_plugin/models/modules/`)
- `gaussian_renderer.py`: Core rendering using diff-gaussian-rasterization
- `gs_param_network.py`: Network for predicting Gaussian parameters (position, color, rotation, scale, opacity)
- `gs_param_network_sample.py`: Sampling-based variant

**Backbones** (`projects/mmdet3d_plugin/models/backbones/`)
- `mask_convnext.py`: ConvNeXt with masking for pre-training
- `mask_resnet.py`: ResNet with masking
- Support for MAE-style masked image modeling

**View Transformation** (`projects/mmdet3d_plugin/models/utils/`)
- `uni3d_voxelpooldepth.py`: Voxel pooling with depth estimation
- `uni3d_viewtrans.py`: View transformation utilities
- Converts 2D image features to unified 3D voxel space

**Custom CUDA Operations** (`projects/mmdet3d_plugin/ops/`)
- `diff-gaussian-rasterization/`: Differentiable Gaussian rasterization (from INRIA GRAPHDECO)
- `smooth_sampler/`: Smooth sampling operations
- `voxel_pool/`: Voxel pooling operations
- `point_ops/`: Point cloud operations

### Data Flow
1. Multi-view images → Backbone (MaskConvNeXt/MaskResNet) → Feature extraction
2. Features → View Transformation → Unified 3D voxel space
3. Voxel features → GaussianHead → 3D Gaussian parameters (position, color, rotation, scale, opacity)
4. Gaussian parameters → Renderer → Reconstructed RGB/Depth/Occupancy
5. Reconstruction loss → Backpropagation for pre-training

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n gaussianpretrain python=3.8
conda activate gaussianpretrain

# Install PyTorch 1.9.1 with CUDA 11.1
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge

# Install mmcv-full and dependencies
pip install mmcv-full==1.3.11 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1 tifffile-2021.11.2 numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 pycocotools==2.0.0 nuscenes-devkit==1.0.5 spconv-cu111 gpustat numba scipy pandas matplotlib Cython shapely loguru tqdm future fire yacs jupyterlab scikit-image pybind11 tensorboardX tensorboard easydict pyyaml open3d addict pyquaternion awscli timm typing-extensions==4.7.1

# Build main package
cd GaussianPretrain
python setup.py develop

# Build diff-gaussian-rasterization CUDA extension
cd projects/mmdet3d_plugin/ops/diff-gaussian-rasterization
python setup.py develop
```

### Training

**Pre-training with GaussianPretrain:**
```bash
# Train on 8 GPUs (recommended)
bash tools/dist_train.sh projects/configs/gaussianpretrain/gp_0.075_convnext.py 8

# Alternative voxel size
bash tools/dist_train.sh projects/configs/gaussianpretrain/gp_0.1_convnext.py 8
```

**Downstream Task Fine-tuning:**
```bash
# 3D Object Detection (UVTR)
bash tools/dist_train.sh projects/configs/gaussianpretrain/uvtr_dn_ft.py 8
bash tools/dist_train.sh projects/configs/gaussianpretrain/uvtr_dn_cs_ft.py 8  # Camera+LiDAR

# Occupancy Prediction
bash tools/dist_train.sh projects/configs/occ_pretrain_ft/bevformer_occ_ft.py 8
bash tools/dist_train.sh projects/configs/occ_pretrain_ft/pano_occ_ft.py 8
```

**Single-GPU Training:**
```bash
# Replace dist_train.sh with train.py for single GPU
python tools/train.py <config_file> --gpus 1
```

### Testing

**Distributed Testing:**
```bash
bash tools/dist_test.sh <config> <checkpoint> <num_gpus>

# Example: 3D detection evaluation
bash tools/dist_test.sh projects/configs/gaussianpretrain/uvtr_dn_ft.py checkpoints/uvtr_dn_ft.pth 8
```

**Single-GPU Testing:**
```bash
python tools/test.py <config> <checkpoint> --eval bbox

# Example
python tools/test.py projects/configs/gaussianpretrain/uvtr_dn_ft.py checkpoints/uvtr_dn_ft.pth --eval bbox
```

**Evaluation Metrics:**
- 3D Detection: `--eval bbox` (outputs NDS, mAP)
- HD-Map: `--eval map` (outputs mAP)
- Occupancy: `--eval iou` (outputs mIoU)

### Data Preparation

Follow dataset preparation instructions from:
- **UVTR**: https://github.com/dvlab-research/UVTR
- **PanoOCC**: https://github.com/Robertwyq/PanoOcc

Typical nuScenes structure:
```
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
└── v1.0-test/
```

## Configuration System

Configs use MMDetection3D's modular system with inheritance:

**Base Configs** (`projects/configs/_base_/`)
- `datasets/nus-3d.py`: nuScenes dataset config
- `default_runtime.py`: Training runtime settings

**Task-Specific Configs** (`projects/configs/`)
- `gaussianpretrain/`: Pre-training configs
- `occ_pretrain_ft/`: Occupancy fine-tuning configs
- `unipad*/`: UVTR variants

**Key Config Parameters:**
- `point_cloud_range`: 3D detection range (default: [-54, -54, -5, 54, 54, 3])
- `unified_voxel_size`: Voxel resolution (0.075 or 0.1 typically)
- `fp16_enabled`: Enable mixed precision training
- `cam_sweep_num`: Number of camera temporal sweeps
- `lidar_sweep_num`: Number of LiDAR temporal sweeps

## Important Implementation Details

### Plugin System
The codebase extends MMDetection3D via plugin system:
```python
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
```
All custom modules are registered through MMDetection3D's registry system using decorators like `@DETECTORS.register_module()`, `@HEADS.register_module()`.

### CUDA Extensions
Multiple custom CUDA operations require compilation:
1. Main package CUDA ops: Built via `setup.py` at root
2. diff-gaussian-rasterization: Built separately in `projects/mmdet3d_plugin/ops/diff-gaussian-rasterization/`

Both must be built with `python setup.py develop` before training.

### Gaussian Splatting Integration
- Uses differentiable Gaussian rasterization from INRIA GRAPHDECO
- Gaussians parameterized by: position (xyz), color (rgb), rotation (quaternion), scale (3D), opacity (alpha)
- Rendering happens in `GaussianRenderer` via CUDA kernels
- Loss computed on rendered RGB, depth, and occupancy vs ground truth

### LiDAR Depth Guidance
The method uses LiDAR points to:
- Guide mask generation (identify valid regions for reconstruction)
- Provide depth supervision
- Initialize 3D Gaussian positions via ray-based guidance

### Memory Management
- FP16 training enabled by default (`fp16_enabled=True`)
- Voxel pooling and sparse convolutions for efficiency
- Configurable ray sampling (`ray_sampler_cfg`) to control memory usage

## Common Workflows

### Adding New Backbone
1. Implement in `projects/mmdet3d_plugin/models/backbones/`
2. Register with `@BACKBONES.register_module()`
3. Import in `projects/mmdet3d_plugin/__init__.py`
4. Update config to use new backbone type

### Modifying Gaussian Parameters
- Edit `gs_param_network.py` or `gs_param_network_sample.py`
- Adjust output dimensions if changing parameter representation
- Update `gaussian_renderer.py` if changing rendering logic

### Changing Voxel Resolution
- Modify `unified_voxel_size` in config
- `unified_voxel_shape` auto-computed from `point_cloud_range / unified_voxel_size`
- Affects memory usage and reconstruction detail

### Adding Custom Loss
1. Implement in `projects/mmdet3d_plugin/models/losses/`
2. Register with `@LOSSES.register_module()`
3. Add to `loss_cfg` in head config
4. Compute in head's `loss()` method

## Troubleshooting

**CUDA Extension Build Failures:**
- Ensure CUDA 11.1 toolkit installed and on PATH
- Check PyTorch CUDA version matches: `torch.version.cuda`
- Build with verbose: `python setup.py develop --verbose`

**Out of Memory:**
- Reduce batch size in config
- Lower `unified_voxel_size` (e.g., 0.075 → 0.1)
- Adjust `ray_sampler_cfg['merged_nsample']` (reduce samples per ray)
- Enable gradient checkpointing if available

**Dataset Loading Errors:**
- Verify nuScenes data structure matches expected format
- Check `data_root` path in config
- Ensure data conversion scripts from UVTR/PanoOCC were run

**Import Errors:**
- Run `python setup.py develop` from root directory
- Build diff-gaussian-rasterization extension
- Check PYTHONPATH includes repository root


## 注意事项
- 用中文和我交流
- 产生的文档都放在docs目录中，基于情况你可以在docs下创建不同的目录来存储不同的文档
- 你生成的一些中间文件，不要到处保存。一般情况下要删除。如果要保留，一定要对文档进行归类，最好放在docs中对应的目录中，或者你创建新的目录来保存