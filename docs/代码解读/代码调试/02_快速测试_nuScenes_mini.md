# 快速测试 - nuScenes mini 数据集（容器环境）

本文档针对**容器环境**，指导您完成数据准备、预处理和验证。

---

## 📋 前提条件

- ✅ 容器环境已安装 GaussianPretrain
- ✅ Conda 环境：`/opt/conda/envs/gaussianpretrain`
- ✅ 当前工作目录：`/workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain`
- ✅ 数据路径：`/workspace/data/nuscenes/v1.0-mini`

---

## 1. 数据准备

### 1.1 检查容器内数据

在**容器内**确认数据完整性：

```bash
# 激活 conda 环境
conda activate gaussianpretrain

# 检查数据路径
ls -la /workspace/data/nuscenes/v1.0-mini/

# 应该看到：
# maps/
# samples/
# sweeps/
# v1.0-mini/
```

### 1.2 创建软链接到项目目录

在**容器内**创建软链接：

```bash
# 容器内执行
cd /workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain

# 创建 data 目录
mkdir -p data

# 创建软链接
ln -s /workspace/data/nuscenes data/nuscenes

# 验证链接
ls -lh data/nuscenes
```

### 1.3 验证数据链接

```bash
# 检查链接是否正确
ls -lh data/nuscenes

# 应该看到 v1.0-mini 目录
ls data/nuscenes/

# 检查样本数据
ls data/nuscenes/samples/CAM_FRONT/ | head -5
ls data/nuscenes/samples/LIDAR_TOP/ | head -5
```

**nuScenes mini 数据集结构**（323个样本）：
```
data/nuscenes/
├── maps/                    # 地图数据（4个城市）
├── samples/                 # 关键帧数据
│   ├── CAM_FRONT/          # 前视相机图像
│   ├── CAM_FRONT_LEFT/     # 前左相机
│   ├── CAM_FRONT_RIGHT/    # 前右相机
│   ├── CAM_BACK/           # 后视相机
│   ├── CAM_BACK_LEFT/      # 后左相机
│   ├── CAM_BACK_RIGHT/     # 后右相机
│   └── LIDAR_TOP/          # 激光雷达点云
├── sweeps/                  # 中间帧数据
│   ├── CAM_FRONT/
│   └── LIDAR_TOP/
└── v1.0-mini/              # 标注文件
    ├── attribute.json
    ├── calibrated_sensor.json
    ├── category.json
    ├── ego_pose.json
    ├── instance.json
    ├── log.json
    ├── map.json
    ├── sample.json
    ├── sample_annotation.json
    ├── sample_data.json
    ├── scene.json
    ├── sensor.json
    └── visibility.json
```

---

## 2. 数据预处理

### 2.1 生成 info 文件

nuScenes 需要预处理生成 `.pkl` 文件，包含样本索引和元数据。

```bash
cd /workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain

# 运行数据预处理
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-mini

# 这个过程大约需要 2-5 分钟
```

**预期输出**：
```
[INFO] Create nuScenes infos.
load train sample: 323/323
load val sample: 0/0
sample: 323, point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
[INFO] nuScenes info train file is saved to ./data/nuscenes/nuscenes_infos_train.pkl
[INFO] nuScenes info val file is saved to ./data/nuscenes/nuscenes_infos_val.pkl
```

### 2.2 验证生成的文件

```bash
# 检查生成的 pkl 文件
ls -lh data/nuscenes/*.pkl

# 应该看到：
# nuscenes_infos_train.pkl  (~30MB)
# nuscenes_infos_val.pkl    (可能为空，mini 数据集没有 val)
```

### 2.3 测试数据加载

创建测试脚本验证数据能否正常加载：

```bash
cat > test_data_loading.py << 'EOF'
"""测试 nuScenes 数据加载"""
import pickle
from mmcv import Config

# 加载配置
config_file = 'projects/configs/gaussianpretrain/gp_0.075_convnext.py'
cfg = Config.fromfile(config_file)

# 修改数据路径为 mini
cfg.data_root = 'data/nuscenes/'
cfg.ann_file = 'data/nuscenes/nuscenes_infos_train.pkl'

# 尝试加载 info 文件
print("=" * 60)
print("测试数据加载")
print("=" * 60)

try:
    with open('data/nuscenes/nuscenes_infos_train.pkl', 'rb') as f:
        data_infos = pickle.load(f)

    print(f"✅ 成功加载 info 文件")
    print(f"📊 训练样本数: {len(data_infos['infos'])}")

    # 显示第一个样本信息
    sample = data_infos['infos'][0]
    print(f"\n第一个样本信息:")
    print(f"  - token: {sample.get('token', 'N/A')}")
    print(f"  - timestamp: {sample.get('timestamp', 'N/A')}")
    print(f"  - cams: {list(sample.get('cams', {}).keys())}")
    print(f"  - lidar_path: {sample.get('lidar_path', 'N/A')}")

    # 检查相机数量
    if 'cams' in sample:
        print(f"  - 相机数量: {len(sample['cams'])}")
        for cam_name, cam_info in list(sample['cams'].items())[:2]:
            print(f"    {cam_name}: {cam_info.get('data_path', 'N/A')}")

    print("\n✅ 数据加载测试通过！")

except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
EOF

# 运行测试
python test_data_loading.py
```

**期望输出**：
```
============================================================
测试数据加载
============================================================
✅ 成功加载 info 文件
📊 训练样本数: 323

第一个样本信息:
  - token: xxx...
  - timestamp: 1532402927647951
  - cams: ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', ...]
  - lidar_path: samples/LIDAR_TOP/xxx.pcd.bin
  - 相机数量: 6
    CAM_FRONT: samples/CAM_FRONT/xxx.jpg
    CAM_FRONT_LEFT: samples/CAM_FRONT_LEFT/xxx.jpg

✅ 数据加载测试通过！
============================================================
```

---

## 3. 修改配置文件（使用 mini 数据集）

为了在 mini 数据集上测试，需要创建一个专用配置：

```bash
cd /workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain

# 创建 mini 数据集配置
cat > projects/configs/gaussianpretrain/gp_0.075_convnext_mini.py << 'EOF'
# 继承基础配置
_base_ = ['./gp_0.075_convnext.py']

# 覆盖数据配置
data_root = 'data/nuscenes/'

# 修改数据集为 mini
data = dict(
    samples_per_gpu=1,  # 单卡批次大小
    workers_per_gpu=2,  # 数据加载线程
    train=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=_base_.train_pipeline,
        classes=_base_.class_names,
        modality=_base_.input_modality,
        test_mode=False,
    ),
    val=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',  # mini 没有单独的 val
        pipeline=_base_.test_pipeline,
        classes=_base_.class_names,
        modality=_base_.input_modality,
    ),
    test=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=_base_.test_pipeline,
        classes=_base_.class_names,
        modality=_base_.input_modality,
    )
)

# 减少训练轮数（mini 数据集）
total_epochs = 6
evaluation = dict(interval=6)

# 日志配置
log_config = dict(
    interval=10,  # 每10个iter打印一次
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

# 工作目录
work_dir = './work_dirs/gp_0.075_convnext_mini'
EOF

echo "✅ 创建了 mini 数据集配置: projects/configs/gaussianpretrain/gp_0.075_convnext_mini.py"
```

---

## 4. 快速测试数据管道

测试数据是否能正常通过训练管道：

```bash
cat > test_data_pipeline.py << 'EOF'
"""测试完整数据管道"""
from mmcv import Config
from mmdet3d.datasets import build_dataset
import torch

print("=" * 60)
print("测试数据管道")
print("=" * 60)

# 加载配置
cfg = Config.fromfile('projects/configs/gaussianpretrain/gp_0.075_convnext_mini.py')

try:
    # 构建数据集
    print("\n1. 构建数据集...")
    dataset = build_dataset(cfg.data.train)
    print(f"✅ 数据集大小: {len(dataset)} 个样本")

    # 加载一个样本
    print("\n2. 加载第一个样本...")
    sample = dataset[0]

    print(f"✅ 成功加载样本")
    print(f"   Keys: {list(sample.keys())}")

    # 检查关键数据
    if 'img' in sample:
        print(f"   - img shape: {sample['img'].data.shape if hasattr(sample['img'], 'data') else 'N/A'}")
    if 'points' in sample:
        print(f"   - points shape: {sample['points'].data.shape if hasattr(sample['points'], 'data') else 'N/A'}")
    if 'img_metas' in sample:
        print(f"   - img_metas: {type(sample['img_metas'])}")

    print("\n✅ 数据管道测试通过！可以开始训练/测试。")

except Exception as e:
    print(f"\n❌ 数据管道测试失败: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
EOF

python test_data_pipeline.py
```

---

## 5. 数据准备完成检查清单

完成以下检查，确认数据准备完毕：

- [ ] **数据目录存在**: `ls data/nuscenes/v1.0-mini/`
- [ ] **样本数据存在**: `ls data/nuscenes/samples/CAM_FRONT/ | wc -l` (应有 ~323 个文件)
- [ ] **LiDAR 数据存在**: `ls data/nuscenes/samples/LIDAR_TOP/ | wc -l`
- [ ] **info 文件已生成**: `ls data/nuscenes/nuscenes_infos_train.pkl`
- [ ] **配置文件已创建**: `ls projects/configs/gaussianpretrain/gp_0.075_convnext_mini.py`
- [ ] **数据加载测试通过**: 运行 `python test_data_loading.py`
- [ ] **数据管道测试通过**: 运行 `python test_data_pipeline.py`

---

## 6. 故障排查

### 问题1: 数据路径不存在

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/nuscenes/...'
```

**解决**:
```bash
# 检查软链接
ls -lh data/nuscenes

# 重新创建链接
rm -f data/nuscenes
ln -s /workspace/data/nuscenes data/nuscenes
```

### 问题2: info 文件生成失败

**症状**:
```
KeyError: 'sample'
```

**解决**:
```bash
# 检查 v1.0-mini 目录是否完整
ls data/nuscenes/v1.0-mini/*.json

# 重新下载或解压 v1.0-mini
```

### 问题3: 内存不足

**症状**: 数据加载时 OOM

**解决**:
```python
# 在配置中减少 workers
data = dict(
    workers_per_gpu=1,  # 从 2 或 4 减少到 1
)
```

---

## 7. 下一步

数据准备完成后，您可以：

1. **测试预训练模型** → 查看 `03_模型下载与测试.md`
2. **开始训练** → 查看 `05_训练流程_单卡与多卡.md`
3. **准备完整数据集** → 查看 `04_完整数据准备_可选.md`

---

## 8. 实际执行总结（2025-11-17）

### 8.1 重要发现：需要使用 Unified 数据格式

**核心问题**: GaussianPretrain 使用的 `NuScenesSweepDataset` 需要特殊的 **unified 数据格式**，包含 `cam_sweeps_info` 字段，而标准的 MMDetection3D 数据转换器不生成此字段。

**解决方案**: 使用 UVTR 项目的数据转换器生成 unified 格式。

### 8.2 依赖修复

在执行过程中遇到多个依赖问题，以下是修复记录：

#### 8.2.1 pycocotools 兼容性问题

```bash
# 错误信息
ValueError: numpy.ndarray size changed, may indicate binary incompatibility.
Expected 96 from C header, got 80 from PyObject

# 修复
pip uninstall -y pycocotools && pip install pycocotools==2.0.0 --no-cache-dir
```

#### 8.2.2 timm 版本问题

```bash
# 错误信息
AttributeError: module 'torch' has no attribute 'fx'

# 修复（降级到兼容 PyTorch 1.9.1 的版本）
pip install timm==0.4.12 --force-reinstall --no-deps
```

#### 8.2.3 缺失模块

```bash
pip install lyft-dataset-sdk
pip install einops
pip install 'trimesh<3.10' --no-deps  # 重要：使用 --no-deps 避免升级 numpy
```

#### 8.2.4 numba.errors 导入错误

修改文件 `mmdet3d/datasets/pipelines/data_augment_utils.py`:

```python
# 修改前
from numba.errors import NumbaPerformanceWarning

# 修改后
from numba import errors as numba_errors
warnings.filterwarnings('ignore', category=numba_errors.NumbaPerformanceWarning)
```

#### 8.2.5 PetrelBackend 问题

修改文件 `tools/data_converter/file_client.py`:

```python
# 修改 BACKEND 配置（从云存储改为本地存储）
BACKEND = EasyDict({
    'NAME': 'HardDiskBackend',  # 原为 'PetrelBackend'
    'KWARGS': {}
})

# 注释掉未使用的 tensorflow 导入
# import tensorflow as tf
```

### 8.3 使用 UVTR 数据转换器生成 Unified 格式

#### 8.3.1 获取 UVTR 代码

```bash
# UVTR 代码位置（需要提前下载）
/workspace/code/cv/AutoLabel/SSL/UVTR
```

#### 8.3.2 生成 Unified 格式数据

```bash
cd /workspace/code/cv/AutoLabel/SSL/UVTR/extra_tools

# 设置环境
source /opt/conda/etc/profile.d/conda.sh
conda activate gaussianpretrain
export PYTHONPATH="/workspace/code/cv/AutoLabel/SSL/UVTR:${PYTHONPATH}"

# 生成 unified 格式的 pkl 文件
python create_data.py nuscenes \
    --root-path /workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain/data/nuscenes \
    --out-dir /workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain/data/nuscenes \
    --extra-tag nuscenes_unified \
    --version v1.0-mini \
    --max-sweeps 10
```

#### 8.3.3 生成的文件

```bash
ls -lh data/nuscenes/nuscenes_unified*.pkl

# 输出：
# nuscenes_unified_infos_train.pkl  (24MB) - 323 训练样本
# nuscenes_unified_infos_val.pkl    (6.1MB) - 验证样本
# nuscenes_unified_dbinfos_train.pkl (4.1MB) - GT 数据库
```

#### 8.3.4 Unified 格式关键字段

```python
# 标准格式 vs Unified 格式的区别
info = {
    'cams': {...},           # 两者都有
    'sweeps': [...],         # 两者都有（LiDAR sweeps）
    'cam_sweeps': {...},     # Unified 特有：相机历史帧路径
    'cam_sweeps_info': {...}, # Unified 特有：相机历史帧标定信息（关键！）
    ...
}
```

### 8.4 更新后的配置文件

**文件路径**: `projects/configs/gaussianpretrain/gp_0.075_convnext_mini.py`

```python
# 继承基础配置
_base_ = ['./gp_0.075_convnext.py']

# 覆盖数据配置 - 使用unified格式
data_root = 'data/nuscenes/'

# 修改数据集为 mini，使用unified格式的pkl文件
data = dict(
    samples_per_gpu=1,  # 单卡批次大小
    workers_per_gpu=2,  # 数据加载线程
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_unified_infos_train.pkl',  # 使用unified格式
    ),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_unified_infos_val.pkl',  # 使用unified格式
    ),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_unified_infos_val.pkl',  # 使用unified格式
    )
)

# 减少训练轮数（mini 数据集）
total_epochs = 6
evaluation = dict(interval=6)

# 日志配置
log_config = dict(
    interval=10,  # 每10个iter打印一次
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

# 工作目录
work_dir = './work_dirs/gp_0.075_convnext_mini'
```

### 8.5 数据管道测试结果

```
============================================================
测试完整数据管道 - NuScenesSweepDataset
============================================================

1. 构建数据集...
✅ 数据集大小: 323 个样本
   数据集类型: NuScenesSweepDataset

2. 加载第一个样本...
✅ 成功加载样本
   数据字段: img_metas, img, gt_bboxes_3d, gt_labels_3d, cam_ids

3. 检查图像数据...
   - img shape: torch.Size([6, 3, 928, 1600])  # 6个相机视角
   - 相机数量: 6
   - 图像尺寸: 928 x 1600

4. 检查标注数据...
   - gt_bboxes_3d 数量: 55 个3D框
   - gt_labels_3d 数量: 55 个标签

5. 检查点云数据（如果有）...
   - 点云 shape: (247997, 5)  # ~24.8万个点

✅ 数据管道测试通过！可以开始训练。
============================================================
```

### 8.6 完整执行流程总结

1. ✅ **数据完整性检查** - 确认 `/workspace/data/nuscenes/v1.0-mini` 存在
2. ✅ **软链接创建** - `ln -s /workspace/data/nuscenes data/nuscenes`
3. ✅ **依赖修复** - 修复 pycocotools、timm、numba 等兼容性问题
4. ✅ **文件修改** - 修复 file_client.py 和 data_augment_utils.py
5. ✅ **使用 UVTR 转换器** - 生成 unified 格式的 pkl 文件（包含 cam_sweeps_info）
6. ✅ **配置文件更新** - 使用 `nuscenes_unified_infos_train.pkl`
7. ✅ **数据管道测试** - 成功加载 323 个样本，6 个相机视角，~25 万个点云点

### 8.7 注意事项

1. **不要使用标准格式**：`nuscenes_infos_train.pkl`（缺少 cam_sweeps_info）
2. **必须使用 Unified 格式**：`nuscenes_unified_infos_train.pkl`
3. **UVTR 项目是必需的**：用于生成正确格式的数据文件
4. **numpy 版本敏感**：保持 numpy==1.19.5，避免安装会升级 numpy 的包

---

## 附录: nuScenes mini 数据集信息

- **大小**: ~4.2 GB（已压缩）
- **场景数**: 10 个场景
- **样本数**: 323 个关键帧
- **覆盖城市**: Boston, Singapore
- **适用场景**: 快速调试、算法验证、环境测试

**与完整数据集的区别**:
- 完整 trainval: 850 个场景，28,130 个样本
- mini: 10 个场景，323 个样本
- mini 足够用于测试模型是否能正常运行
- 正式训练需要完整数据集
