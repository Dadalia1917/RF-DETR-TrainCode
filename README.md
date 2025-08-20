# RF-DETR 通用训练代码

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

本项目是基于RF-DETR官方文档和源码实现的通用训练代码，提供了完整的模型训练、评估和模型转换功能。RF-DETR是一个高性能的实时目标检测模型，在COCO数据集上达到了SOTA性能。

## ✨ 功能特性

- 🚀 **完整的训练流程**：支持从数据准备到模型训练的完整流程
- 📊 **多种模型规格**：支持RF-DETR Nano、Small、Medium、Base、Large等多种模型
- 🎯 **灵活的配置**：支持自定义训练参数、数据集配置等
- 💾 **模型转换**：提供PTH转PT格式的转换工具
- 📈 **训练监控**：集成TensorBoard和Wandb支持
- 🔧 **优化训练**：支持混合精度训练、EMA、早停等优化技术

## 📋 环境要求

- Python 3.11
- CUDA 12.8 (推荐)
- 16GB+ GPU显存 (训练大模型时)

## 🛠️ 安装说明

### 1. 配置镜像源
```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

### 2. 创建并激活环境
```bash
conda create -n DETR python=3.11
conda activate DETR
```

### 3. 安装基础依赖
```bash
# 安装ultralytics
pip install -U ultralytics

# 卸载可能存在的旧版本torch
pip uninstall torch
pip uninstall torchvision

# 安装CUDA版本的PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 4. 安装Flash Attention (可选，需要下载对应版本)
```bash
# 请根据您的系统下载对应的flash_attn wheel文件
pip install flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl
```

### 5. 安装其他依赖
```bash
# 安装指定版本的torch和xformers
pip install torch==2.8.0 xformers

# 安装数据增强库
pip install -U albumentations

# 安装机器学习相关库
pip install huggingface_hub datasets

# 安装UI界面库
pip install pyqt5

# 安装Web服务相关库
pip install flask flask-socketio openai

# 安装数据库相关库
pip install sqlalchemy flask_bcrypt flask_login

# 安装RF-DETR和可视化库
pip install -q rfdetr supervision
```

## 📁 项目结构

```
RF-DETR-Traincode/
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── train.py                 # 主训练脚本
├── convert_model.py         # 模型格式转换工具
├── config/                  # 配置文件目录
│   ├── training_config.yaml # 训练配置文件
│   └── model_config.yaml    # 模型配置文件
├── data/                    # 数据集目录
│   ├── train/              # 训练数据
│   ├── val/                # 验证数据
│   └── test/               # 测试数据
├── output/                  # 训练输出目录
│   ├── checkpoints/        # 模型检查点
│   ├── logs/              # 训练日志
│   └── results/           # 评估结果
└── utils/                   # 工具函数
    ├── dataset.py          # 数据集处理
    ├── visualization.py    # 可视化工具
    └── metrics.py          # 评估指标
```

## 🚀 快速开始

### 1. 准备数据集

将您的数据集按照COCO格式组织，或使用Roboflow格式：

```
data/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── val/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

### 2. 开始训练

#### 基础训练 (SGD + AMP优化)
```bash
python train.py
```

#### 高级训练策略
```bash
python train_advanced.py
```

选项包括：
- **SGD + AMP**: 追求最高精度，收敛较慢但泛化更好
- **AdamW + AMP**: 平衡精度和训练速度
- **对比实验**: 同时测试多种优化器策略
- **Medium模型**: 更大模型获得更高精度

### 3. 模型转换

训练完成后，可以将PTH格式转换为PT格式：

```bash
python convert_model.py \
    --input_path ./output/checkpoints/best_model.pth \
    --output_path ./output/best_model.pt \
    --model_size medium
```

## ⚙️ 配置参数

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_dir` | str | - | 数据集路径 |
| `model_size` | str | "medium" | 模型规格 (nano/small/medium/base/large) |
| `epochs` | int | 100 | 训练轮数 |
| `batch_size` | int | 8 | 批次大小 |
| `lr` | float | 1e-4 | 学习率 |
| `resolution` | int | 640 | 输入图像分辨率 |
| `use_ema` | bool | True | 是否使用EMA |
| `early_stopping` | bool | True | 是否使用早停 |

### 模型规格对比

| 模型 | 参数量 | 分辨率 | mAP@50 | mAP@50:95 | 推理速度 |
|------|--------|--------|--------|-----------|----------|
| RF-DETR-N | 30.5M | 384 | 67.6 | 48.4 | 2.32ms |
| RF-DETR-S | 32.1M | 512 | 72.1 | 53.0 | 3.52ms |
| RF-DETR-M | 33.7M | 576 | 73.6 | 54.7 | 4.52ms |

### 优化器和训练技术对比

| 技术组合 | 收敛速度 | 最终精度 | 显存使用 | 适用场景 |
|----------|----------|----------|----------|----------|
| **SGD + AMP** | 较慢 | ⭐⭐⭐⭐⭐ | 低 | 追求最高精度 |
| **AdamW + AMP** | 快 | ⭐⭐⭐⭐ | 低 | 平衡训练 |
| **SGD** | 较慢 | ⭐⭐⭐⭐ | 高 | 传统训练 |
| **AdamW** | 快 | ⭐⭐⭐ | 高 | 快速验证 |

#### AMP (自动混合精度) 优势
- 🚀 **训练加速**: 30-50%速度提升
- 💾 **显存节省**: 减少约50%显存使用
- 🎯 **精度保持**: 现代GPU上无损精度
- ⚡ **更大批次**: 可以使用更大的batch_size

## 📊 训练监控

### TensorBoard
```bash
tensorboard --logdir ./output/logs
```

### Wandb (可选)
在训练脚本中设置：
```python
wandb_config = {
    "project": "rf-detr-training",
    "name": "experiment-1"
}
```

## 🔧 高级用法

### 自定义配置文件

```yaml
# config/training_config.yaml
model:
  size: "medium"
  pretrained: True
  
training:
  epochs: 100
  batch_size: 8
  learning_rate: 1e-4
  optimizer: "AdamW"
  scheduler: "cosine"
  
data:
  resolution: 640
  augmentation: True
  multi_scale: True
```

### 断点续训

```bash
python train.py \
    --resume ./output/checkpoints/checkpoint_epoch_50.pth \
    --dataset_dir ./data
```

## 📝 注意事项

1. **GPU显存需求**：建议使用16GB+显存的GPU进行训练
2. **数据集格式**：确保数据集按照COCO格式正确组织
3. **预训练权重**：首次运行会自动下载预训练权重
4. **模型保存**：训练过程中会自动保存最佳模型和检查点

## 🤝 贡献

欢迎提交Issue和Pull Request来改进本项目！

## 📄 许可证

本项目基于Apache 2.0许可证开源。

## 🔗 相关链接

- [RF-DETR官方仓库](https://github.com/roboflow/rf-detr)
- [RF-DETR官方文档](https://rfdetr.roboflow.com)
- [论文地址](https://arxiv.org/abs/2501.03595)

## 📞 联系方式

如有问题，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至项目维护者

---

**Star ⭐ 本项目如果对您有帮助！**