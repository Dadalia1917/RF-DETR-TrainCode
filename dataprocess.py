# coding:utf-8
"""
数据格式转换脚本
将YOLO格式转换为RF-DETR需要的COCO格式
"""

import json
import os
from pathlib import Path
from PIL import Image
import shutil
import yaml


def load_class_names_from_yaml(yolo_dataset_path):
    """
    从YOLO数据集的data.yaml文件中加载类别名称
    """
    yaml_path = Path(yolo_dataset_path) / 'data.yaml'
    
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if 'names' in data:
                class_names = data['names']
                print(f"✓ 从 {yaml_path} 加载类别名称: {class_names}")
                return class_names
            else:
                print(f"⚠ {yaml_path} 中未找到'names'字段")
        except Exception as e:
            print(f"⚠ 读取 {yaml_path} 失败: {e}")
    else:
        print(f"⚠ 未找到 {yaml_path} 文件")
    
    # 如果无法从yaml读取，则使用默认类别名称
    print("使用默认类别名称: class_0, class_1, ...")
    # 尝试通过扫描标签文件来推断类别数量
    train_labels_dir = Path(yolo_dataset_path) / 'train' / 'labels'
    if train_labels_dir.exists():
        max_class_id = -1
        for label_file in train_labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            max_class_id = max(max_class_id, class_id)
            except:
                continue
        
        if max_class_id >= 0:
            class_names = [f'class_{i}' for i in range(max_class_id + 1)]
            print(f"✓ 从标签文件推断类别数量: {len(class_names)} 个类别")
            return class_names
    
    # 最后的fallback
    print("❌ 无法确定类别数量，请手动指定")
    return ['class_0', 'class_1', 'class_2', 'class_3']  # 默认4个类别


def yolo_to_coco(yolo_dataset_path='datasets/Data', output_path='dataset'):
    """
    将YOLO格式数据集转换为COCO格式
    
    Args:
        yolo_dataset_path: YOLO数据集路径，默认'dataset'
        output_path: 输出COCO数据集路径，默认'dataset_coco'
    """
    
    print(f"开始转换数据格式...")
    print(f"输入路径: {yolo_dataset_path}")
    print(f"输出路径: {output_path}")
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 自动从data.yaml读取类别名称
    class_names = load_class_names_from_yaml(yolo_dataset_path)
    
    # 检查YOLO数据集是否有test集
    yolo_test_path = Path(yolo_dataset_path) / 'test'
    has_test_set = yolo_test_path.exists() and (yolo_test_path / 'images').exists() and (yolo_test_path / 'labels').exists()
    
    # 处理训练集、验证集和测试集 (按照截图COCO格式)
    split_mapping = {'train': 'train', 'val': 'valid'}  # val -> valid
    
    # 如果有test集，添加到处理列表
    if has_test_set:
        split_mapping['test'] = 'test'
        print("✓ 检测到test集，将一同转换")
    else:
        print("⚠ 未检测到test集，稍后将复制valid集作为test集")
    
    # 处理所有数据集分割
    processed_splits = {}
    
    for yolo_split, coco_split in split_mapping.items():
        print(f"\n处理 {yolo_split} -> {coco_split} 集...")
        
        yolo_split_path = Path(yolo_dataset_path) / yolo_split
        if not yolo_split_path.exists():
            print(f"跳过 {yolo_split} 集，目录不存在")
            continue
            
        # 获取图像和标签路径
        images_path = yolo_split_path / 'images'
        labels_path = yolo_split_path / 'labels'
        
        if not images_path.exists() or not labels_path.exists():
            print(f"跳过 {yolo_split} 集，images或labels目录不存在")
            continue
        
        # 创建COCO格式的输出目录 (按照截图格式)
        coco_split_dir = output_path / coco_split
        coco_split_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建COCO注释结构
        coco_data = {
            "info": {
                "description": f"RF-DETR {coco_split} dataset",
                "version": "1.0"
            },
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # 添加类别信息 (COCO格式类别ID从1开始，但模型需要从0开始)
        for i, class_name in enumerate(class_names):
            coco_data["categories"].append({
                "id": i,  # 保持从0开始，与YOLO一致
                "name": class_name,
                "supercategory": "object"
            })
        
        # 处理图像和标注
        image_id = 0
        annotation_id = 0
        
        # 获取所有图像文件
        image_files = list(images_path.glob('*.jpg'))
        print(f"找到 {len(image_files)} 张图像")
        
        for img_file in image_files:
            # 获取对应的标签文件
            label_file = labels_path / (img_file.stem + '.txt')
            
            if not label_file.exists():
                continue
            
            # 读取图像尺寸
            try:
                with Image.open(img_file) as img:
                    width, height = img.size
            except:
                continue
            
            # 添加图像信息
            image_info = {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": img_file.name
            }
            coco_data["images"].append(image_info)
            
            # 复制图像文件到COCO格式目录 (直接放在split目录下)
            output_img_path = coco_split_dir / img_file.name
            if not output_img_path.exists():
                shutil.copy2(img_file, output_img_path)
            
            # 读取YOLO格式的标注
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_width = float(parts[3])
                bbox_height = float(parts[4])
                
                # 验证类别ID是否在有效范围内
                if class_id < 0 or class_id >= len(class_names):
                    print(f"⚠ 警告：无效的类别ID {class_id}，跳过该标注")
                    continue
                
                # 转换为COCO格式的边界框
                x = (x_center - bbox_width / 2) * width
                y = (y_center - bbox_height / 2) * height
                w = bbox_width * width
                h = bbox_height * height
                
                # 确保边界框在图像范围内
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                area = w * h
                
                # 添加注释信息
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1
            
            image_id += 1
        
        # 保存COCO格式的注释文件 (按照截图格式，每个split目录下有自己的注释文件)
        annotation_file = coco_split_dir / f"_annotations.coco.json"
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        print(f"{yolo_split} -> {coco_split} 集转换完成: {len(coco_data['images'])} 张图像, {len(coco_data['annotations'])} 个标注")
        
        # 保存处理后的数据供test集使用
        processed_splits[coco_split] = coco_data
    
    # 如果没有原始test集，则创建test集 (复制valid集的数据)
    if not has_test_set and 'valid' in processed_splits:
        print(f"\n创建test集 (复制valid集数据)...")
        
        test_dir = output_path / 'test'
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制valid集的COCO数据作为test集
        test_coco_data = processed_splits['valid'].copy()
        test_coco_data['info']['description'] = "RF-DETR test dataset (copied from valid)"
        
        # 复制valid集的图像到test目录
        valid_dir = output_path / 'valid'
        for img_file in valid_dir.glob('*.jpg'):
            test_img_path = test_dir / img_file.name
            if not test_img_path.exists():
                shutil.copy2(img_file, test_img_path)
        
        # 保存test集注释文件
        test_annotation_file = test_dir / "_annotations.coco.json"
        with open(test_annotation_file, 'w', encoding='utf-8') as f:
            json.dump(test_coco_data, f, indent=2, ensure_ascii=False)
        
        print(f"test集创建完成: {len(test_coco_data['images'])} 张图像, {len(test_coco_data['annotations'])} 个标注")
    elif has_test_set:
        print(f"\n✓ test集已从原始数据转换完成")
    
    print(f"\n数据集转换完成！输出目录: {output_path}")


def check_yolo_dataset(dataset_path='datasets/Data'):
    """检查YOLO数据集状态"""
    print("检查YOLO数据集...")
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ YOLO数据集目录 {dataset_path} 不存在")
        return False
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            images_count = len(list((split_path / 'images').glob('*.jpg')))
            labels_count = len(list((split_path / 'labels').glob('*.txt')))
            print(f"✓ {split} 集: {images_count} 张图像, {labels_count} 个标签")
        else:
            if split == 'test':
                print(f"⚠ {split} 集目录不存在 (将复制valid集数据)")
            else:
                print(f"❌ {split} 集目录不存在")
    
    return True

def check_coco_dataset(dataset_path='dataset'):
    """检查COCO数据集状态"""
    print("检查COCO数据集...")
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ COCO数据集目录 {dataset_path} 不存在")
        return False
    
    for split in ['train', 'valid', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            annotation_file = split_path / "_annotations.coco.json"
            images_count = len(list(split_path.glob('*.jpg')))
            print(f"✓ {split} 集: {images_count} 张图像, 注释文件: {'存在' if annotation_file.exists() else '不存在'}")
        else:
            print(f"❌ {split} 集目录不存在")
    
    return True


if __name__ == '__main__':
    print("=" * 50)
    print("RF-DETR 数据格式转换工具")
    print("YOLO格式 (datasets/Data) -> COCO格式 (dataset)")
    print("=" * 50)
    
    # 检查原始YOLO数据集
    if not check_yolo_dataset('datasets/Data'):
        print("请先准备YOLO格式的数据集在 datasets/Data 目录")
        exit(1)
    
    # 转换数据格式
    yolo_to_coco('datasets/Data', 'dataset')
    
    # 检查转换后的数据集
    print("\n" + "=" * 30)
    check_coco_dataset('dataset')
    
    print("\n转换完成！现在可以开始训练了")
    print("运行命令: python train.py")
