"""
诊断训练问题：
1. 检查归一化是否只使用训练集
2. 检查loss function是否使用了weights
3. 分析类别2被错判为类别1的原因
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

def diagnose_normalization():
    """诊断问题1：归一化是否只使用训练集"""
    print("="*60)
    print("问题1: 检查归一化实现")
    print("="*60)
    
    # 检查dataset.py中的归一化实现
    print("\n当前归一化实现（dataset.py preprocess_eeg/preprocess_ecg）:")
    print("  ❌ 问题：每个样本的每个patch的每个通道独立归一化")
    print("  - 这意味着每个样本使用自己的mean和std")
    print("  - 训练集和验证集/测试集使用了不同的归一化标准")
    print("  - 这会导致数据分布不一致，影响模型性能")
    print("\n  ✓ 正确做法：")
    print("  - 应该使用训练集的全局mean和std")
    print("  - 验证集和测试集使用相同的mean和std进行归一化")
    print("  - 这样可以保证数据分布的一致性")
    
    return {
        'issue': 'normalization_per_sample',
        'severity': 'high',
        'description': '每个样本独立归一化，导致训练集和验证集分布不一致'
    }

def diagnose_loss_function():
    """诊断问题2：loss function是否使用了weights"""
    print("\n" + "="*60)
    print("问题2: 检查Loss Function和Weights使用")
    print("="*60)
    
    print("\n检查结果：")
    print("  ✓ Loss function (base/loss_function.py) 支持 sample_weights")
    print("  ✓ Trainer (trainer.py) 在计算loss时传递了 sample_weights")
    print("  ✓ 如果batch中有'weight'字段，会被用于加权loss")
    
    print("\n⚠️  需要注意：")
    print("  - weights来自数据中的weights.npy文件")
    print("  - 需要确认这些weights是否是伪标签的weights")
    print("  - 如果weights是伪标签的置信度，使用它们是正确的")
    print("  - 如果weights是其他含义，可能需要调整")
    
    return {
        'issue': 'weights_usage',
        'severity': 'medium',
        'description': 'Loss function使用了weights，需要确认weights的含义'
    }

def analyze_class_2_misclassification():
    """诊断问题3：类别2被错判为类别1的原因"""
    print("\n" + "="*60)
    print("问题3: 分析类别2被错判为类别1的原因")
    print("="*60)
    
    # 从混淆矩阵描述中提取数据
    # True Label 2: 正确分类42个，错判为Class 1有146个，错判为Class 0有41个
    total_class_2 = 42 + 146 + 41 + 7  # 236个样本
    
    print(f"\n类别2的分布情况：")
    print(f"  - 总样本数: {total_class_2}")
    print(f"  - 正确分类: 42 ({42/total_class_2*100:.1f}%)")
    print(f"  - 错判为类别1: 146 ({146/total_class_2*100:.1f}%) ⚠️")
    print(f"  - 错判为类别0: 41 ({41/total_class_2*100:.1f}%)")
    print(f"  - 错判为类别3: 7 ({7/total_class_2*100:.1f}%)")
    
    print("\n可能的原因：")
    print("  1. 类别不平衡：")
    print("     - 类别2的样本数量可能较少")
    print("     - 模型倾向于预测样本数量多的类别")
    
    print("\n  2. 特征相似性：")
    print("     - 类别2和类别1的生理信号特征可能相似")
    print("     - 模型难以区分这两个类别")
    
    print("\n  3. 数据质量问题：")
    print("     - 类别2的标签可能存在噪声")
    print("     - 部分类别2的样本可能实际上更接近类别1")
    
    print("\n  4. 模型容量不足：")
    print("     - 模型可能没有足够的表达能力来区分这些相似类别")
    print("     - 需要更强的特征提取能力")
    
    print("\n建议的解决方案：")
    print("  1. 使用类别权重（class_weights）平衡类别")
    print("  2. 使用Focal Loss处理类别不平衡")
    print("  3. 增加类别2的样本数量（数据增强或收集更多数据）")
    print("  4. 使用更复杂的模型架构")
    print("  5. 分析类别2和类别1的特征差异，可能需要额外的特征")
    
    return {
        'issue': 'class_2_misclassification',
        'severity': 'high',
        'description': '类别2大量被错判为类别1，可能是类别不平衡或特征相似性导致'
    }

def check_class_distribution():
    """检查类别分布"""
    print("\n" + "="*60)
    print("检查类别分布")
    print("="*60)
    
    # 尝试从数据文件读取标签分布
    data_path = Path('./data/training_dataset_valid')
    labels_path = data_path / 'labels.npy'
    
    if labels_path.exists():
        labels = np.load(labels_path)
        unique, counts = np.unique(labels, return_counts=True)
        
        print("\n数据集类别分布：")
        total = len(labels)
        for cls, count in zip(unique, counts):
            percentage = count / total * 100
            print(f"  类别 {cls}: {count} 个样本 ({percentage:.1f}%)")
        
        # 检查类别不平衡
        max_count = counts.max()
        min_count = counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"\n类别不平衡比例: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 5:
            print("  ⚠️  严重的类别不平衡！")
        elif imbalance_ratio > 2:
            print("  ⚠️  存在类别不平衡")
        else:
            print("  ✓ 类别分布相对平衡")
    else:
        print(f"\n无法找到标签文件: {labels_path}")
        print("请确认数据路径是否正确")

def generate_fix_suggestions():
    """生成修复建议"""
    print("\n" + "="*60)
    print("修复建议总结")
    print("="*60)
    
    print("\n1. 修复归一化问题（高优先级）：")
    print("   - 修改 dataset.py，使用训练集的全局mean和std")
    print("   - 在 init_dataloader 中计算训练集的mean和std")
    print("   - 验证集和测试集使用相同的mean和std")
    
    print("\n2. 处理类别不平衡（高优先级）：")
    print("   - 使用 class_weights 参数（main.py中已有）")
    print("   - 或使用 Focal Loss")
    print("   - 考虑数据增强或重采样")
    
    print("\n3. 改进类别2的分类（中优先级）：")
    print("   - 分析类别2和类别1的特征差异")
    print("   - 可能需要更复杂的模型或额外的特征")
    print("   - 考虑使用集成方法")
    
    print("\n4. 验证weights的使用（低优先级）：")
    print("   - 确认weights.npy的含义")
    print("   - 如果weights是伪标签置信度，当前实现是正确的")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("训练问题诊断报告")
    print("="*60)
    
    # 诊断问题1
    issue1 = diagnose_normalization()
    
    # 诊断问题2
    issue2 = diagnose_loss_function()
    
    # 诊断问题3
    issue3 = analyze_class_2_misclassification()
    
    # 检查类别分布
    check_class_distribution()
    
    # 生成修复建议
    generate_fix_suggestions()
    
    print("\n" + "="*60)
    print("诊断完成")
    print("="*60)

