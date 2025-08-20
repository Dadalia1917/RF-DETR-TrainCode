# coding:utf-8
import torch
from rfdetr import RFDETRSmall

# 数据集路径
dataset_dir = 'dataset'

if __name__ == '__main__':
    # 加载RF-DETR模型 (Small版本比Base精度更高)
    model = RFDETRSmall()
    
    # 训练模型 - 基于官方代码优化
    model.train(
        dataset_dir=dataset_dir,
        epochs=100,                 # 100轮训练
        batch_size=4,               # 适合8G显存
        grad_accum_steps=4,         # 有效batch_size=16
        lr=2e-4,                    # 基于官方1e-4适度提升
        output_dir='runs/train_rf-detr-s'
    )