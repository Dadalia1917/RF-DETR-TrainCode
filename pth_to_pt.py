# coding:utf-8
"""
RF-DETR模型格式转换工具
将训练好的.pth格式转换为.pt格式
"""

import torch
import argparse
import os
from pathlib import Path
from rfdetr import RFDETRMedium, RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall


def get_model_class(model_name):
    """根据模型名称获取对应的模型类"""
    model_mapping = {
        'medium': RFDETRMedium,
        'base': RFDETRBase,
        'large': RFDETRLarge,
        'nano': RFDETRNano,
        'small': RFDETRSmall
    }
    
    # 从文件名推断模型类型
    model_name_lower = model_name.lower()
    for key, model_class in model_mapping.items():
        if key in model_name_lower:
            return model_class
    
    # 默认返回Medium
    print(f"⚠ 无法从文件名推断模型类型，使用默认的RFDETRMedium")
    return RFDETRMedium


def convert_pth_to_pt(pth_path, pt_path=None, model_type='auto'):
    """
    将.pth格式转换为.pt格式
    
    Args:
        pth_path: 输入的.pth文件路径
        pt_path: 输出的.pt文件路径，如果为None则自动生成
        model_type: 模型类型，'auto'为自动检测
    """
    
    print("=" * 50)
    print("RF-DETR模型格式转换工具")
    print(".pth格式 -> .pt格式")
    print("=" * 50)
    
    # 检查输入文件
    pth_path = Path(pth_path)
    if not pth_path.exists():
        print(f"❌ 输入文件不存在: {pth_path}")
        return False
    
    if not pth_path.suffix == '.pth':
        print(f"⚠ 输入文件不是.pth格式: {pth_path}")
    
    # 生成输出文件路径
    if pt_path is None:
        pt_path = pth_path.with_suffix('.pt')
    else:
        pt_path = Path(pt_path)
    
    print(f"输入文件: {pth_path}")
    print(f"输出文件: {pt_path}")
    
    try:
        # 加载.pth文件
        print(f"\n正在加载.pth文件...")
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
        
        # 检查checkpoint结构
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("✓ 检测到训练checkpoint格式 (包含'model'键)")
                
                # 尝试获取其他信息
                if 'args' in checkpoint:
                    args = checkpoint['args']
                    if hasattr(args, 'num_classes'):
                        print(f"✓ 检测到类别数: {args.num_classes}")
                    if hasattr(args, 'encoder'):
                        print(f"✓ 检测到编码器类型: {args.encoder}")
                
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✓ 检测到模型state_dict格式")
            else:
                # 假设整个字典就是state_dict
                state_dict = checkpoint
                print("✓ 假设整个字典为state_dict")
        else:
            print("❌ 无法识别的checkpoint格式")
            return False
        
        # 确定模型类型
        if model_type == 'auto':
            model_class = get_model_class(pth_path.stem)
        else:
            model_mapping = {
                'medium': RFDETRMedium,
                'base': RFDETRBase,
                'large': RFDETRLarge,
                'nano': RFDETRNano,
                'small': RFDETRSmall
            }
            model_class = model_mapping.get(model_type.lower(), RFDETRMedium)
        
        print(f"✓ 使用模型类型: {model_class.__name__}")
        
        # 获取正确的类别数量
        num_classes = 4  # 默认值
        if 'model' in checkpoint and 'class_embed.weight' in checkpoint['model']:
            actual_num_classes = checkpoint['model']['class_embed.weight'].shape[0]
            # 由于RF-DETR内部会+1，所以我们设置为actual_num_classes-1
            num_classes = actual_num_classes - 1 if actual_num_classes > 1 else actual_num_classes
            print(f"✓ 从权重形状推断: 实际权重{actual_num_classes}类，设置num_classes={num_classes}")
        
        # 创建模型
        print(f"\n正在创建模型...")
        model = model_class(num_classes=num_classes, pretrain_weights=None)
        
        # 加载权重
        print(f"正在加载权重...")
        
        # 过滤不匹配的权重
        model_state_dict = model.model.model.state_dict()
        filtered_state_dict = {}
        
        for k, v in state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"⚠ 跳过形状不匹配的权重: {k} {v.shape} vs {model_state_dict[k].shape}")
            else:
                print(f"⚠ 跳过不存在的权重: {k}")
        
        # 尝试加载兼容的权重
        try:
            model.model.model.load_state_dict(filtered_state_dict, strict=False)
            print("✓ 权重加载成功")
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            return False
        
        # 设置为评估模式
        model.model.model.eval()
        
        # 保存为.pt格式
        print(f"\n正在保存为.pt格式...")
        
        # 创建输出目录
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        torch.save(model.model.model.state_dict(), pt_path)
        
        print(f"✅ 转换完成！")
        print(f"✓ 输出文件: {pt_path}")
        print(f"✓ 文件大小: {pt_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # 验证转换结果
        print(f"\n验证转换结果...")
        try:
            test_model = model_class(num_classes=num_classes, pretrain_weights=None)
            test_model.model.model.load_state_dict(torch.load(pt_path, map_location='cpu'))
            print("✅ 转换验证成功！.pt文件可以正常加载")
        except Exception as e:
            print(f"⚠ 转换验证失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换过程中出错: {e}")
        return False


def batch_convert(input_dir, output_dir=None):
    """批量转换目录中的所有.pth文件"""
    
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有.pth文件
    pth_files = list(input_dir.glob('*.pth'))
    
    if not pth_files:
        print(f"❌ 在目录中未找到.pth文件: {input_dir}")
        return
    
    print(f"找到 {len(pth_files)} 个.pth文件")
    
    success_count = 0
    for pth_file in pth_files:
        pt_file = output_dir / pth_file.with_suffix('.pt').name
        print(f"\n{'='*30}")
        print(f"转换: {pth_file.name}")
        
        if convert_pth_to_pt(pth_file, pt_file):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"批量转换完成: {success_count}/{len(pth_files)} 个文件转换成功")


def main():
    parser = argparse.ArgumentParser(description='RF-DETR模型格式转换工具 (.pth -> .pt)')
    parser.add_argument('input', help='输入的.pth文件路径或目录')
    parser.add_argument('-o', '--output', help='输出的.pt文件路径或目录')
    parser.add_argument('-m', '--model', default='auto', 
                       choices=['auto', 'nano', 'small', 'medium', 'base', 'large'],
                       help='模型类型 (默认: auto)')
    parser.add_argument('-b', '--batch', action='store_true',
                       help='批量转换模式')
    
    args = parser.parse_args()
    
    if args.batch or Path(args.input).is_dir():
        batch_convert(args.input, args.output)
    else:
        convert_pth_to_pt(args.input, args.output, args.model)


def quick_convert_output():
    """快速转换训练完成的模型"""
    
    print("=" * 50)
    print("快速转换训练完成的模型")
    print("搜索训练模型文件")
    print("=" * 50)
    
    # 搜索目录 - 优先检查新的runs目录结构
    search_dirs = [
        Path("runs/train_rf-detr-n"),  # 新的目录结构
        Path("output")                 # 旧的目录结构（向后兼容）
    ]
    
    found_dir = None
    for search_dir in search_dirs:
        if search_dir.exists():
            model_files = {
                "最佳EMA模型": search_dir / "checkpoint_best_ema.pth",
                "最佳常规模型": search_dir / "checkpoint_best_regular.pth", 
                "最新检查点": search_dir / "checkpoint.pth"
            }
            
            # 检查是否有模型文件
            if any(path.exists() for path in model_files.values()):
                found_dir = search_dir
                print(f"✓ 在 {search_dir} 中发现训练模型")
                break
    
    if found_dir is None:
        print("❌ 没有找到训练完成的模型文件")
        print("请确保已完成训练，或检查以下目录:")
        for search_dir in search_dirs:
            print(f"  - {search_dir}")
        return
    
    # 查找可转换的模型文件
    model_files = {
        "最佳EMA模型": found_dir / "checkpoint_best_ema.pth",
        "最佳常规模型": found_dir / "checkpoint_best_regular.pth", 
        "最新检查点": found_dir / "checkpoint.pth"
    }
    
    available_models = []
    for name, path in model_files.items():
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            available_models.append((name, path, size_mb))
    
    if not available_models:
        print(f"❌ {found_dir}目录中没有找到可转换的模型文件")
        return
    
    print("发现以下模型文件:")
    for i, (name, path, size_mb) in enumerate(available_models, 1):
        print(f"  {i}. {name}")
        print(f"     文件: {path.name}")
        print(f"     大小: {size_mb:.1f} MB")
        print()
    
    # 用户选择
    while True:
        try:
            choice = input(f"请选择要转换的模型 (1-{len(available_models)}, 0=全部转换): ").strip()
            if choice == '0':
                # 转换所有模型
                for name, path, _ in available_models:
                    pt_path = path.with_suffix('.pt')
                    print(f"\n转换 {name}...")
                    if convert_pth_to_pt(path, pt_path, 'nano'):  # 使用nano因为用户训练的是nano
                        print(f"✅ {name} 转换完成: {pt_path}")
                    else:
                        print(f"❌ {name} 转换失败")
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(available_models):
                # 转换单个模型
                idx = int(choice) - 1
                name, path, _ = available_models[idx]
                pt_path = path.with_suffix('.pt')
                print(f"\n转换 {name}...")
                if convert_pth_to_pt(path, pt_path, 'nano'):
                    print(f"✅ 转换完成: {pt_path}")
                else:
                    print(f"❌ 转换失败")
                break
            else:
                print("无效选择，请重新输入")
        except KeyboardInterrupt:
            print("\n用户取消操作")
            break


if __name__ == '__main__':
    # 如果没有命令行参数，使用交互模式
    import sys
    if len(sys.argv) == 1:
        print("=" * 50)
        print("RF-DETR模型格式转换工具")
        print(".pth格式 -> .pt格式")
        print("=" * 50)
        
        # 检查是否有训练完成的模型
        search_dirs = [
            Path("runs/train_rf-detr-n"),  # 新的目录结构
        ]
        
        found_models = False
        for search_dir in search_dirs:
            if search_dir.exists() and any(search_dir.glob("checkpoint_best_*.pth")):
                found_models = True
                break
        
        if found_models:
            print("✓ 检测到训练完成的模型")
            choice = input("是否使用快速转换模式? (y/n): ").strip().lower()
            if choice in ['y', 'yes', '是']:
                quick_convert_output()
                exit(0)
        
        # 标准交互式输入
        input_path = input("请输入.pth文件路径: ").strip()
        if not input_path:
            print("❌ 未输入文件路径")
            exit(1)
        
        output_path = input("请输入输出路径 (回车使用默认): ").strip()
        if not output_path:
            output_path = None
        
        model_type = input("请输入模型类型 (nano/small/medium/base/large，回车自动检测): ").strip()
        if not model_type:
            model_type = 'auto'
        
        print()
        convert_pth_to_pt(input_path, output_path, model_type)
    else:
        main()

