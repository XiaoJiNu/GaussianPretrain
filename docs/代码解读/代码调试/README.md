# GaussianPretrain 代码调试指南

本目录提供 GaussianPretrain 项目的**实战调试指南**，从环境验证到模型训练的完整流程。

**容器环境信息**：
- **Conda 环境**: `/opt/conda/envs/gaussianpretrain`
- **项目路径**: `/workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain`
- **数据路径**: `/workspace/data/nuscenes/v1.0-mini`

---

## 📚 文档概览

| 文档 | 说明 | 适用场景 |
|------|------|----------|
| [01_环境验证清单](./01_环境验证清单.md) | 验证所有依赖和编译是否正确 | 首次安装后必读 |
| [02_快速测试_nuScenes_mini](./02_快速测试_nuScenes_mini.md) | 使用 mini 数据集快速测试 | 快速验证、调试 |
| [03_模型下载与测试](./03_模型下载与测试.md) | 下载预训练模型并测试 | 模型推理验证 |
| [04_完整数据准备_可选](./04_完整数据准备_可选.md) | 准备完整 nuScenes 数据集 | 正式训练前 |
| [05_训练流程_单卡与多卡](./05_训练流程_单卡与多卡.md) | 单卡/多卡训练详细指南 | 开始训练 |

---

## 🚀 快速开始

### 新手推荐路径

```
01. 环境验证
    ↓
02. mini 数据集测试
    ↓
03. 下载并测试预训练模型
    ↓
05. 单卡训练 mini 数据集（可选）
    ↓
04. 准备完整数据集（可选）
    ↓
05. 多卡训练完整数据集
```

### 快速验证流程（30 分钟）

如果您只想**快速验证项目能否运行**，按以下步骤：

```bash
# 激活 conda 环境
conda activate gaussianpretrain

# 1. 验证环境（5分钟）
cd /workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain
python verify_env.py  # 见 01_环境验证清单.md

# 2. 准备 mini 数据（10分钟）
# 按 02_快速测试_nuScenes_mini.md 创建软链接和生成 info 文件
ln -s /workspace/data/nuscenes data/nuscenes
python tools/create_data.py nuscenes --root-path ./data/nuscenes --version v1.0-mini

# 3. 下载并测试模型（15分钟）
# 按 03_模型下载与测试.md 下载模型并测试
gdown https://drive.google.com/uc?id=1f3brSVve4i3UaFBRXODu0KOy9NCkWzjY \
  -O checkpoints/gp_pretrain_convnext.pth
python test_forward.py

# ✅ 如果都通过，说明环境正常，可以开始训练！
```

---

## 📖 各文档详细说明

### 01. 环境验证清单

**目的**：确保所有依赖库和 CUDA 扩展正确安装

**核心内容**：
- Python 3.8 验证
- PyTorch 1.9.1 + CUDA 11.1 验证
- mmcv-full、mmdet3d 版本检查
- **diff-gaussian-rasterization** 编译验证（最关键）
- GPU 显存建议

**何时使用**：
- ✅ 首次安装环境后
- ✅ 更新依赖库后
- ✅ 遇到 ImportError 时

**运行时间**：~2 分钟

---

### 02. 快速测试 - nuScenes mini

**目的**：使用 323 个样本的 mini 数据集快速验证流程

**核心内容**：
- 数据软链接配置（容器环境）
- 数据预处理（生成 .pkl 文件）
- 创建 mini 专用配置文件
- 数据加载测试脚本

**何时使用**：
- ✅ 快速验证数据管道
- ✅ 调试模型修改
- ✅ 测试新配置
- ❌ 正式训练（数据太少）

**数据规模**：
- 样本数：323
- 大小：~4.2 GB
- 处理时间：~5 分钟

---

### 03. 模型下载与测试

**目的**：下载官方预训练模型并验证推理功能

**核心内容**：
- 预训练模型下载（Google Drive）
- 单卡推理测试
- 可视化渲染结果（RGB、Depth）
- 快速前向传播验证

**何时使用**：
- ✅ 验证模型是否能正常运行
- ✅ 查看预训练效果
- ✅ 开始微调前
- ✅ 调试推理流程

**可用模型**：
- `gp_pretrain_convnext.pth`: 预训练权重（~500MB）
- `uvtr_gp_det.pth`: 检测模型
- `uvtr_gp_cs_det.pth`: Camera+LiDAR 检测
- `bevformer_occ_ft.pth`: 占用预测
- `pano_occ_ft.pth`: 全景占用

**测试时间**：~15 分钟（mini 数据集）

---

### 04. 完整数据准备（可选）

**目的**：准备完整的 nuScenes trainval 数据集（28,130 样本）

**核心内容**：
- 下载 350GB 完整数据集
- 解压和目录结构说明
- 完整数据集预处理
- 存储和性能优化建议

**何时使用**：
- ✅ 正式训练发表级模型
- ✅ 复现论文结果
- ❌ 快速调试（用 mini 即可）

**数据规模**：
- 样本数：28,130（训练）+ 6,019（验证）
- 大小：~350 GB
- 下载时间：数小时到数天（取决于网络）
- 处理时间：~1 小时

**与 mini 对比**：

| 指标 | mini | full |
|------|------|------|
| 样本数 | 323 | 28,130 |
| 训练时间 | ~30分钟（6 epochs） | ~6-9天（24 epochs，8卡） |
| 检测 NDS | ~0.30 | ~0.50+ |

---

### 05. 训练流程 - 单卡与多卡

**目的**：详细的训练指南，涵盖单卡调试和多卡分布式训练

**核心内容**：
- **单卡训练**：mini 数据集调试（~30分钟完成）
- **多卡训练**：完整数据集训练（8卡，~6-9天）
- 训练监控（日志、TensorBoard、GPU）
- 检查点管理
- 常见问题（OOM、训练慢、损失 NaN 等）
- 性能优化技巧

**何时使用**：
- ✅ 开始预训练
- ✅ 微调下游任务
- ✅ 调试训练问题
- ✅ 优化训练性能

**训练时间估算**：

**mini 数据集（调试用）**：
- 1×RTX 3090：~5分钟/epoch → 6 epochs = ~30分钟

**完整数据集（正式训练）**：
- 8×RTX 3090：~9小时/epoch → 24 epochs = ~9天
- 8×V100 32GB：~6小时/epoch → 24 epochs = ~6天

**关键命令**：
```bash
# 单卡训练（mini）
python tools/train.py projects/configs/gaussianpretrain/gp_0.075_convnext_mini.py

# 8卡训练（full）
bash tools/dist_train.sh projects/configs/gaussianpretrain/gp_0.075_convnext.py 8
```

---

## 🎯 不同场景的使用指南

### 场景1：首次使用，快速验证

**目标**：确认项目能否正常运行

**步骤**：
1. 阅读并执行 `01_环境验证清单.md` 中的验证脚本
2. 按 `02_快速测试_nuScenes_mini.md` 准备 mini 数据
3. 按 `03_模型下载与测试.md` 测试预训练模型
4. ✅ 如果都通过，环境正常

**时间**：~30 分钟

---

### 场景2：调试新功能/代码修改

**目标**：快速验证代码修改是否正确

**步骤**：
1. 使用 mini 数据集（`02_快速测试_nuScenes_mini.md`）
2. 单卡训练 1-2 个 epoch（`05_训练流程_单卡与多卡.md`）
3. 检查日志和输出

**时间**：~10-20 分钟

---

### 场景3：正式训练发表级模型

**目标**：在完整数据集上训练以获得最佳性能

**步骤**：
1. 确认环境正常（`01_环境验证清单.md`）
2. 下载完整数据集（`04_完整数据准备_可选.md`）
3. 多卡训练（`05_训练流程_单卡与多卡.md`）
4. 监控训练过程（TensorBoard）
5. 评估模型性能

**时间**：~1周（包括数据准备）

---

### 场景4：遇到错误需要调试

**常见错误索引**：

| 错误类型 | 查看文档 | 章节 |
|---------|---------|------|
| `ImportError: cannot import GaussianRasterizer` | 01_环境验证清单 | Q1: 编译失败 |
| `CUDA out of memory` | 05_训练流程 | 问题1: OOM |
| `FileNotFoundError: data/nuscenes/...` | 02_快速测试 | 6. 故障排查 |
| `RuntimeError: NCCL error` | 05_训练流程 | 问题3: 分布式训练挂起 |
| `loss: nan` | 05_训练流程 | 问题4: 损失 NaN |

---

## 🛠️ 容器环境说明

本指南针对**容器环境**编写，关键路径：

**Conda 环境**：
```bash
# 激活环境
conda activate gaussianpretrain
# 或使用完整路径
source /opt/conda/bin/activate gaussianpretrain
```

**容器内项目路径**：
```
/workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain
```

**容器内数据路径**：
```
/workspace/data/nuscenes/v1.0-mini  # mini 数据集
/workspace/data/nuscenes_full       # 完整数据集（可选）
```

**软链接配置**：
```bash
# 在项目目录创建软链接指向数据
cd /workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain
ln -s /workspace/data/nuscenes data/nuscenes
```

---

## 📊 完整流程图

```
┌─────────────────────────────────────────────────────┐
│             GaussianPretrain 使用流程                │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  01. 环境验证                   │
        │  • Python 3.8                  │
        │  • PyTorch 1.9.1 + CUDA 11.1   │
        │  • diff-gaussian-rasterization │
        └────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  02. 数据准备（mini）           │
        │  • 软链接数据                   │
        │  • 生成 .pkl                   │
        │  • 测试数据加载                 │
        └────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  03. 模型测试                   │
        │  • 下载预训练模型               │
        │  • 前向传播测试                 │
        │  • 可视化结果                   │
        └────────────────────────────────┘
                         │
                         ▼
                 ┌───────┴────────┐
                 │                │
                 ▼                ▼
        ┌─────────────┐   ┌─────────────┐
        │ 快速调试     │   │ 正式训练     │
        │ (mini)      │   │ (full)      │
        │             │   │             │
        │ 05. 单卡训练 │   │ 04. 完整数据 │
        │ ~30分钟     │   │ ~350GB      │
        │             │   │             │
        │             │   │ 05. 8卡训练  │
        │             │   │ ~6-9天      │
        └─────────────┘   └─────────────┘
                 │                │
                 └────────┬────────┘
                          ▼
        ┌────────────────────────────────┐
        │  模型评估与应用                  │
        │  • 验证集评估                   │
        │  • 可视化分析                   │
        │  • 下游任务微调                 │
        └────────────────────────────────┘
```

---

## 💡 常用命令速查表

### 环境验证
```bash
# 激活环境
conda activate gaussianpretrain

# 运行验证脚本
python verify_env.py

# 检查关键组件
python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 数据准备
```bash
# 生成 mini 数据 info
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --version v1.0-mini

# 生成完整数据 info
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --version v1.0-trainval
```

### 模型测试
```bash
# 下载模型
gdown https://drive.google.com/uc?id=1f3brSVve4i3UaFBRXODu0KOy9NCkWzjY \
  -O checkpoints/gp_pretrain_convnext.pth

# 测试推理
python tools/test.py \
    projects/configs/gaussianpretrain/gp_0.075_convnext_mini.py \
    checkpoints/gp_pretrain_convnext.pth \
    --eval bbox
```

### 训练
```bash
# 单卡训练（mini）
python tools/train.py \
    projects/configs/gaussianpretrain/gp_0.075_convnext_mini.py

# 8卡训练（full）
bash tools/dist_train.sh \
    projects/configs/gaussianpretrain/gp_0.075_convnext.py \
    8
```

### 监控
```bash
# 查看日志
tail -f work_dirs/gp_pretrain/*.log | grep loss

# GPU监控
gpustat -i 1

# TensorBoard
tensorboard --logdir work_dirs/gp_pretrain --port 6006
```

---

## ❓ 常见问题快速索引

| 问题 | 文档位置 |
|------|---------|
| 如何验证环境是否正确？ | 01_环境验证清单 |
| mini 数据集在哪下载？ | 02_快速测试_nuScenes_mini（第1节） |
| 如何测试预训练模型？ | 03_模型下载与测试（第2节） |
| 完整数据集多大？ | 04_完整数据准备_可选（第1.1节） |
| 单卡训练需要多久？ | 05_训练流程_单卡与多卡（第1.2节） |
| CUDA OOM 怎么办？ | 05_训练流程_单卡与多卡（问题1） |
| 训练速度慢怎么办？ | 05_训练流程_单卡与多卡（问题2） |
| 损失变 NaN 怎么办？ | 05_训练流程_单卡与多卡（问题4） |

---

## 📞 获取帮助

如果遇到文档未覆盖的问题：

1. **检查项目 README**：`/GaussianPretrain/README.md`
2. **查看 CLAUDE.md**：`/GaussianPretrain/CLAUDE.md`（面向开发者）
3. **查看代码解读**：`/GaussianPretrain/docs/代码解读/`
4. **提交 Issue**：GitHub Issues（如果是开源项目）

---

## 📝 文档维护

**最后更新**：2024-01

**适用版本**：
- GaussianPretrain: v1.0
- MMDetection3D: v0.17.3
- PyTorch: 1.9.1
- CUDA: 11.1

**贡献者**：Claude Code

---

## 🎉 开始使用

**推荐新手第一步**：

```bash
# 激活 conda 环境
conda activate gaussianpretrain

cd /workspace/code/cv/AutoLabel/SSL/GaussianPretrain-all/GaussianPretrain

# 1. 验证环境（必做）
cat docs/代码解读/代码调试/01_环境验证清单.md

# 2. 快速测试（推荐）
cat docs/代码解读/代码调试/02_快速测试_nuScenes_mini.md
```

祝您使用愉快！🚀
