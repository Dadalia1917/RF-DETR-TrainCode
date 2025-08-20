# coding:utf-8
"""
RF-DETR推理测试脚本
验证训练好的模型能够正常进行检测任务
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRNano
import supervision as sv
import time


def load_trained_model(checkpoint_path='runs/train_rf-detr-s/checkpoint_best_regular.pth'):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 训练checkpoint路径
    
    Returns:
        loaded model
    """
    print(f"正在加载训练好的模型: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        
        # 检查其他可能的模型文件 - 优先检查新的runs目录结构
        search_dirs = [
            Path("runs/train_rf-detr-n"),  # 新的目录结构
        ]
        
        checkpoint_path = None
        for search_dir in search_dirs:
            if search_dir.exists():
                available_models = list(search_dir.glob("checkpoint_best_*.pth"))
                if available_models:
                    print(f"✓ 在{search_dir}中发现以下可用模型:")
                    for model in available_models:
                        print(f"  - {model.name}")
                    checkpoint_path = available_models[0]
                    print(f"使用: {checkpoint_path}")
                    break
        
        if checkpoint_path is None:
            print("❌ 没有找到训练好的模型文件")
            print("请确保已完成训练，或检查以下目录:")
            for search_dir in search_dirs:
                print(f"  - {search_dir}")
            return None
    
    try:
        # 首先加载checkpoint获取正确的配置
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 从checkpoint中获取正确的类别数量
        # RF-DETR会自动将num_classes+1，所以我们需要从权重形状反推
        if 'model' in checkpoint and 'class_embed.weight' in checkpoint['model']:
            actual_num_classes = checkpoint['model']['class_embed.weight'].shape[0]
            # 由于RF-DETR内部会+1，所以我们设置为actual_num_classes-1
            num_classes = actual_num_classes - 1 if actual_num_classes > 1 else actual_num_classes
            print(f"✓ 从权重形状推断: 实际权重{actual_num_classes}类，设置num_classes={num_classes}")
        else:
            num_classes = 3  # 默认3，让RF-DETR自动+1变成4
            print(f"✓ 使用默认类别数: {num_classes}")
        
        # 创建模型实例 - 使用计算后的类别数量
        model = RFDETRNano(num_classes=num_classes, pretrain_weights=None)
        
        # 加载模型权重 - 正确的结构是 model.model.model
        if 'model' in checkpoint:
            # 过滤掉不匹配的class_embed权重
            state_dict = checkpoint['model']
            model_state_dict = model.model.model.state_dict()
            
            # 检查并过滤不匹配的权重
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"⚠ 跳过形状不匹配的权重: {k} {v.shape} vs {model_state_dict[k].shape}")
                else:
                    print(f"⚠ 跳过不存在的权重: {k}")
            
            model.model.model.load_state_dict(filtered_state_dict, strict=False)
            print("✓ 成功加载兼容的模型权重")
        else:
            model.model.model.load_state_dict(checkpoint, strict=False)
            print("✓ 成功加载模型权重 (直接格式)")
        
        # 获取训练配置信息
        if 'args' in checkpoint:
            args = checkpoint['args']
            if hasattr(args, 'class_names'):
                print(f"✓ 类别名称: {args.class_names}")
                # 保存类别名称供后续使用
                model._inference_class_names = args.class_names
            else:
                # 根据类别数量生成通用类别名称
                model._inference_class_names = [f'class_{i}' for i in range(num_classes)]
                print(f"✓ 使用通用类别名称: {model._inference_class_names}")
        else:
            # 根据类别数量生成通用类别名称
            model._inference_class_names = [f'class_{i}' for i in range(num_classes)]
            print(f"✓ 使用通用类别名称: {model._inference_class_names}")
        
        # 设置为评估模式
        model.model.model.eval()
        
        # 检查设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model.model = model.model.model.to(device)
        print(f"✓ 模型已加载到: {device}")
        
        return model
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None


def detect_image(model, image_path, confidence_threshold=0.3):
    """
    对图像进行检测
    
    Args:
        model: 训练好的模型
        image_path: 图像路径
        confidence_threshold: 置信度阈值
    
    Returns:
        detections: 检测结果
        image: 原始图像
    """
    print(f"\n正在检测图像: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ 图像文件不存在: {image_path}")
        return None, None
    
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        print(f"✓ 图像尺寸: {image.size}")
        
        # 开始推理
        print(f"正在进行检测... (置信度阈值: {confidence_threshold})")
        start_time = time.time()
        
        # 使用模型进行预测
        detections = model.predict(image_path, threshold=confidence_threshold)
        
        inference_time = time.time() - start_time
        print(f"✓ 推理完成，耗时: {inference_time:.3f}秒")
        
        # 检查检测结果
        if len(detections.xyxy) > 0:
            print(f"✓ 检测到 {len(detections.xyxy)} 个目标")
            
            # 显示检测详情
            for i, (bbox, confidence, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
                class_name = model.class_names[class_id] if hasattr(model, 'class_names') and class_id < len(model.class_names) else f"class_{class_id}"
                print(f"  目标 {i+1}: {class_name} (置信度: {confidence:.3f})")
                print(f"          边界框: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            print("⚠ 未检测到任何目标")
        
        return detections, image
        
    except Exception as e:
        print(f"❌ 检测失败: {e}")
        return None, None


def visualize_results(image, detections, model, output_path='detection_result.jpg'):
    """
    可视化检测结果
    
    Args:
        image: 原始图像
        detections: 检测结果
        output_path: 保存路径
    """
    print(f"\n正在生成可视化结果...")
    
    try:
        # 创建图像副本用于绘制
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 颜色列表
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
        ]
        
        # 绘制检测框
        for i, (bbox, confidence, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
            x1, y1, x2, y2 = bbox
            color = colors[class_id % len(colors)]
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 准备标签文本
            if hasattr(model, '_inference_class_names') and model._inference_class_names:
                class_names_list = model._inference_class_names
            else:
                # 动态生成类别名称
                class_names_list = [f'class_{i}' for i in range(len(set(detections.class_id)))]
            
            class_name = class_names_list[class_id] if class_id < len(class_names_list) else f"class_{class_id}"
            label = f"{class_name}: {confidence:.3f}"
            
            # 绘制标签背景
            bbox_text = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(bbox_text, fill=color)
            
            # 绘制标签文本
            draw.text((x1, y1-25), label, fill=(255, 255, 255), font=font)
        
        # 保存结果
        vis_image.save(output_path)
        print(f"✓ 可视化结果已保存: {output_path}")
        
        # 显示结果信息
        print(f"✓ 检测摘要:")
        print(f"  - 总目标数: {len(detections.xyxy)}")
        print(f"  - 平均置信度: {detections.confidence.mean():.3f}")
        print(f"  - 最高置信度: {detections.confidence.max():.3f}")
        
        return vis_image
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        return None


def main():
    """主函数"""
    print("=" * 60)
    print("RF-DETR模型推理测试")
    print("验证训练好的模型检测能力")
    print("=" * 60)
    
    # 1. 加载训练好的模型
    model = load_trained_model()
    if model is None:
        print("❌ 无法加载模型，请检查训练是否完成")
        return
    
    # 2. 检查测试图像
    test_image_path = "test1.jpg"
    
    # 如果test1.jpg不存在，尝试使用示例图像
    if not Path(test_image_path).exists():
        print(f"⚠ 测试图像 {test_image_path} 不存在")
        
        # 尝试使用RF-DETR自带的测试图像
        example_image = "rfdetr/assets/test.jpg"
        if Path(example_image).exists():
            test_image_path = example_image
            print(f"✓ 使用示例图像: {test_image_path}")
        else:
            print("❌ 请将测试图像命名为 test1.jpg 并放在RF-DETR目录下")
            return
    
    # 3. 进行检测
    detections, original_image = detect_image(model, test_image_path, confidence_threshold=0.3)
    
    if detections is None:
        print("❌ 检测失败")
        return
    
    # 4. 可视化结果
    if len(detections.xyxy) > 0:
        result_image = visualize_results(original_image, detections, model, 'detection_result.jpg')
        if result_image:
            print("\n🎉 检测测试完成！")
            print("您可以查看以下文件:")
            print("  - detection_result.jpg: 检测结果可视化")
    else:
        print("\n⚠ 未检测到目标，可能原因:")
        print("  1. 置信度阈值设置过高")
        print("  2. 图像中没有训练过的类别目标")
        print("  3. 模型需要更多训练")
        
        # 尝试降低阈值重新检测
        print("\n尝试降低置信度阈值到0.1...")
        detections_low, _ = detect_image(model, test_image_path, confidence_threshold=0.1)
        if detections_low and len(detections_low.xyxy) > 0:
            visualize_results(original_image, detections_low, model, 'detection_result_low_threshold.jpg')
            print("✓ 低阈值检测结果已保存为 detection_result_low_threshold.jpg")


if __name__ == '__main__':
    main()