"""
类别权重计算方法说明和实现

常见的类别权重计算方法：
1. 平衡权重（Balanced）：weight_i = total_samples / (num_classes * count_i)
2. 逆频率权重（Inverse Frequency）：weight_i = total_samples / count_i
3. 平方根逆频率（Sqrt Inverse）：weight_i = sqrt(total_samples / count_i)
4. 归一化权重（Normalized）：先计算逆频率，然后归一化到[0,1]或[1, max_weight]
"""

import numpy as np
from collections import Counter

def compute_balanced_weights(class_counts):
    """
    计算平衡权重（Balanced Weights）
    公式：weight_i = total_samples / (num_classes * count_i)
    
    这种方法确保所有类别的总权重相等，常用于类别不平衡问题。
    
    Args:
        class_counts: 每个类别的样本数量列表，如 [4124, 2158, 1447, 407, 445]
    
    Returns:
        weights: 类别权重列表
    """
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    weights = [total_samples / (num_classes * count) for count in class_counts]
    return weights

def compute_inverse_frequency_weights(class_counts):
    """
    计算逆频率权重（Inverse Frequency Weights）
    公式：weight_i = total_samples / count_i
    
    这种方法直接使用频率的倒数，简单直接。
    
    Args:
        class_counts: 每个类别的样本数量列表
    
    Returns:
        weights: 类别权重列表
    """
    total_samples = sum(class_counts)
    weights = [total_samples / count for count in class_counts]
    return weights

def compute_sqrt_inverse_weights(class_counts):
    """
    计算平方根逆频率权重（Sqrt Inverse Frequency Weights）
    公式：weight_i = sqrt(total_samples / count_i)
    
    这种方法使用平方根，使权重变化更平滑，避免极端值。
    
    Args:
        class_counts: 每个类别的样本数量列表
    
    Returns:
        weights: 类别权重列表
    """
    total_samples = sum(class_counts)
    weights = [np.sqrt(total_samples / count) for count in class_counts]
    return weights

def compute_normalized_weights(class_counts, method='balanced', normalize_to_max=True):
    """
    计算归一化权重
    
    Args:
        class_counts: 每个类别的样本数量列表
        method: 计算方法 ('balanced', 'inverse', 'sqrt')
        normalize_to_max: 如果True，归一化到[1, max_weight]；如果False，归一化到[0, 1]
    
    Returns:
        weights: 归一化后的类别权重列表
    """
    if method == 'balanced':
        weights = compute_balanced_weights(class_counts)
    elif method == 'inverse':
        weights = compute_inverse_frequency_weights(class_counts)
    elif method == 'sqrt':
        weights = compute_sqrt_inverse_weights(class_counts)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if normalize_to_max:
        # 归一化到 [1, max_weight]，最小权重为1
        min_weight = min(weights)
        weights = [w / min_weight for w in weights]
    else:
        # 归一化到 [0, 1]
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
    
    return weights

def compute_class_weights_from_labels(labels, method='balanced', normalize_to_max=True):
    """
    从标签直接计算类别权重
    
    Args:
        labels: 标签数组（numpy array或list）
        method: 计算方法 ('balanced', 'inverse', 'sqrt')
        normalize_to_max: 是否归一化到最大权重
    
    Returns:
        weights_dict: 字典，键为类别，值为权重
        weights_list: 按类别顺序排列的权重列表
    """
    # 统计每个类别的数量
    counter = Counter(labels)
    num_classes = len(counter)
    
    # 获取所有类别（按顺序）
    classes = sorted(counter.keys())
    class_counts = [counter[cls] for cls in classes]
    
    # 计算权重
    if method == 'balanced':
        weights = compute_balanced_weights(class_counts)
    elif method == 'inverse':
        weights = compute_inverse_frequency_weights(class_counts)
    elif method == 'sqrt':
        weights = compute_sqrt_inverse_weights(class_counts)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 归一化
    if normalize_to_max:
        min_weight = min(weights)
        weights = [w / min_weight for w in weights]
    
    # 创建字典和列表
    weights_dict = {cls: weight for cls, weight in zip(classes, weights)}
    weights_list = weights
    
    return weights_dict, weights_list, class_counts

def print_weights_info(class_counts, method='balanced'):
    """打印权重信息"""
    print(f"\n类别分布和权重计算（方法: {method}）")
    print("="*60)
    
    total = sum(class_counts)
    num_classes = len(class_counts)
    
    print(f"总样本数: {total}")
    print(f"类别数: {num_classes}")
    print(f"\n类别分布:")
    for i, count in enumerate(class_counts):
        percentage = count / total * 100
        print(f"  类别 {i}: {count:5d} 个样本 ({percentage:5.1f}%)")
    
    # 计算不同方法的权重
    methods = ['balanced', 'inverse', 'sqrt']
    print(f"\n不同方法的权重:")
    print(f"{'类别':<8} {'Balanced':<12} {'Inverse':<12} {'Sqrt':<12}")
    print("-"*60)
    
    for i in range(num_classes):
        balanced = compute_balanced_weights(class_counts)[i]
        inverse = compute_inverse_frequency_weights(class_counts)[i]
        sqrt = compute_sqrt_inverse_weights(class_counts)[i]
        print(f"{i:<8} {balanced:<12.4f} {inverse:<12.4f} {sqrt:<12.4f}")
    
    # 推荐方法
    print(f"\n推荐使用: Balanced 方法（平衡权重）")
    print(f"原因: 确保所有类别的总权重相等，最适合处理类别不平衡问题")

if __name__ == '__main__':
    # 示例：使用你的数据分布
    class_counts = [4124, 2158, 1447, 407, 445]
    
    print_weights_info(class_counts)
    
    # 计算推荐的权重（Balanced方法）
    weights = compute_balanced_weights(class_counts)
    print(f"\n推荐的类别权重（Balanced方法，已归一化）:")
    normalized_weights = compute_normalized_weights(class_counts, method='balanced', normalize_to_max=True)
    print(f"  {normalized_weights}")
    print(f"\n用于命令行参数:")
    weights_str = ','.join([f'{w:.4f}' for w in normalized_weights])
    print(f"  -class_weights \"{weights_str}\"")

