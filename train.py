# coding:utf-8
import torch
from rfdetr import RFDETRSmall

# 数据集路径
dataset_dir = 'dataset'

if __name__ == '__main__':
    # 加载RF-DETR模型 (Small版本比Base精度更高)
    model = RFDETRSmall()
    
    # 训练模型 - SGD优化器 + AMP混合精度训练优化
    model.train(
        dataset_dir=dataset_dir,
        epochs=120,                 # SGD通常需要更多轮数
        batch_size=6,               # AMP节省显存，可适度增加
        grad_accum_steps=3,         # 有效batch_size=18
        lr=1e-3,                    # SGD需要更高学习率
        weight_decay=5e-4,          # SGD建议更高权重衰减
        lr_scheduler='cosine',      # 余弦学习率调度
        warmup_epochs=5,            # SGD需要预热
        amp=True,                   # 开启混合精度训练
        use_ema=True,               # 指数移动平均提升精度
        early_stopping=True,        # 早停防止过拟合
        early_stopping_patience=15, # SGD收敛较慢，增加耐心
        output_dir='runs/train_rf-detr-s_sgd_amp'
    )