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
        epochs=300,                 # 300轮训练
        imgsz=640,
        batch_size=16,               # 适合8G显存
        grad_accum_steps=4,         # 有效batch_size=64 (16×4)
        lr=1e-4,                    # 基于官方1e-4
        optimizer='SGD',            # SGD优化器，更好的泛化性能
        amp=True,                   # 启用混合精度，提升训练效率
        output_dir='runs/train_rf-detr-s'
    )