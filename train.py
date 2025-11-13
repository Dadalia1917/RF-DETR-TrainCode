# coding:utf-8
import torch
import warnings
from rfdetr import RFDETRSmall

# 数据集路径
dataset_dir = 'dataset'

if __name__ == '__main__':
    # 指定使用第0块GPU
    device_id = 'cuda:0'  # 修改为完整的设备字符串

    # 加载RF-DETR模型 (Small版本比Base精度更高)
    model = RFDETRSmall()

    # 训练模型 - 通过device参数指定GPU
    model.train(
        dataset_dir=dataset_dir,
        epochs=300,  # 300轮训练
        imgsz=640,
        batch_size=16,  # 适合16G显存
        grad_accum_steps=2,  # 有效batch_size=32 (16×2)
        lr=1e-4,  # 基于官方1e-4
        optimizer='SGD',  # SGD优化器，更好的泛化性能
        amp=True,  # 启用混合精度，提升训练效率
        output_dir='runs/train_rf-detr-s',
        device=device_id,  # 关键：指定设备ID（完整的设备字符串）
        distributed=False  # Windows上建议关闭分布式
    )