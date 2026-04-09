#!/usr/bin/env python3
"""
计算IMU数据中ef冲突量与labels的相关性

计算步骤：
1. 从imu_windows.npy中提取索引12-14的ef冲突量
2. 计算每个window下ef模长|ef|的RMS
3. 计算与labels的Spearman相关系数
4. 绘制ef RMS与labels的散点图
"""

import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import argparse
import sys
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置中文字体（处理中文兼容性）
available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
preferred_fonts = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Songti SC', 'SimHei', 'Arial Unicode MS']
for f in preferred_fonts:
    if f in available_fonts:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break
else:
    print("⚠️  警告: 未找到中文字体，图表中的中文可能无法正常显示")


def load_data(dataset_path: Path):
    """加载IMU windows和labels数据"""
    imu_path = dataset_path / "imu_windows.npy"
    labels_path = dataset_path / "labels.npy"
    
    if not imu_path.exists():
        raise FileNotFoundError(f"未找到IMU数据文件: {imu_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"未找到labels文件: {labels_path}")
    
    print(f"📂 加载数据...")
    imu_windows = np.load(imu_path)  # shape: (N, 18, 2500)
    labels = np.load(labels_path)  # shape: (N,)
    
    print(f"   IMU windows形状: {imu_windows.shape}")
    print(f"   Labels形状: {labels.shape}")
    
    # 验证数据形状
    if len(imu_windows.shape) != 3:
        raise ValueError(f"IMU windows应该是3维数组，实际形状: {imu_windows.shape}")
    if imu_windows.shape[1] != 18:
        raise ValueError(f"IMU windows应该有18个通道，实际: {imu_windows.shape[1]}")
    if imu_windows.shape[0] != labels.shape[0]:
        raise ValueError(f"IMU windows和labels的样本数不匹配: {imu_windows.shape[0]} vs {labels.shape[0]}")
    
    return imu_windows, labels


def extract_ef_conflicts(imu_windows: np.ndarray):
    """
    提取ef冲突量（索引12-14）
    
    参数:
        imu_windows: (N, 18, 2500) - IMU windows数据
    
    返回:
        ef_conflicts: (N, 3, 2500) - ef冲突量（X, Y, Z三个维度）
    """
    # 索引12-14
    ef_conflicts = imu_windows[:, 3:6, :]  # (N, 3, 2500)
    print(f"   EF冲突量形状: {ef_conflicts.shape}")
    
    return ef_conflicts


def compute_ef_magnitude_rms(ef_conflicts: np.ndarray):
    """
    计算每个window下ef模长|ef|的RMS
    
    参数:
        ef_conflicts: (N, 3, 2500) - ef冲突量
    
    返回:
        ef_rms: (N,) - 每个window的ef模长RMS值
    """
    N, _, T = ef_conflicts.shape
    
    print(f"\n📊 计算ef模长RMS...")
    
    # 计算每个时间点的ef模长 |ef|
    ef_magnitudes = np.linalg.norm(ef_conflicts, axis=1)  # (N, 2500)
    print(f"   EF模长形状: {ef_magnitudes.shape}")
    
    # 计算每个window的RMS
    ef_rms = np.sqrt(np.mean(ef_magnitudes ** 2, axis=1))  # (N,)
    print(f"   EF RMS形状: {ef_rms.shape}")
    
    # 打印统计信息
    print(f"\n   EF RMS统计信息:")
    print(f"     均值: {np.mean(ef_rms):.6f}")
    print(f"     标准差: {np.std(ef_rms):.6f}")
    print(f"     最小值: {np.min(ef_rms):.6f}")
    print(f"     最大值: {np.max(ef_rms):.6f}")
    print(f"     中位数: {np.median(ef_rms):.6f}")
    
    return ef_rms


def compute_correlation(ef_rms: np.ndarray, labels: np.ndarray):
    """
    计算ef RMS与labels的Spearman相关系数
    
    参数:
        ef_rms: (N,) - ef模长RMS值
        labels: (N,) - 被试评分labels
    
    返回:
        correlation: float - Spearman相关系数
        p_value: float - p值
    """
    print(f"\n📈 计算Spearman相关系数...")
    
    # 检查数据有效性
    valid_mask = ~(np.isnan(ef_rms) | np.isnan(labels) | np.isinf(ef_rms) | np.isinf(labels))
    n_valid = np.sum(valid_mask)
    n_total = len(ef_rms)
    
    if n_valid < n_total:
        print(f"   ⚠️  警告: 发现 {n_total - n_valid} 个无效值，将使用有效数据计算")
    
    if n_valid < 3:
        raise ValueError(f"有效数据点太少 ({n_valid})，无法计算相关系数")
    
    ef_rms_valid = ef_rms[valid_mask]
    labels_valid = labels[valid_mask]
    
    # 计算Spearman相关系数
    correlation, p_value = spearmanr(ef_rms_valid, labels_valid)
    
    print(f"   使用 {n_valid}/{n_total} 个有效数据点")
    print(f"   Spearman相关系数: {correlation:.6f}")
    print(f"   p值: {p_value:.6f}")
    
    # 判断显著性
    if p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif p_value < 0.01:
        significance = "** (p < 0.01)"
    elif p_value < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "不显著 (p >= 0.05)"
    
    print(f"   显著性: {significance}")
    
    return correlation, p_value, n_valid


def balance_sample_by_rating(ef_rms: np.ndarray, labels: np.ndarray, 
                              samples_per_rating: int = 200):
    """
    对每个评分等级进行平衡采样
    
    参数:
        ef_rms: (N,) - ef模长RMS值
        labels: (N,) - 被试评分labels
        samples_per_rating: int - 每个评分等级采样的数量
    
    返回:
        ef_rms_balanced: (M,) - 平衡后的ef RMS值
        labels_balanced: (M,) - 平衡后的labels
        original_counts: dict - 原始每个评分的数量
        sampled_counts: dict - 采样后每个评分的数量
    """
    
    # 过滤无效值
    valid_mask = ~(np.isnan(ef_rms) | np.isnan(labels) | np.isinf(ef_rms) | np.isinf(labels))
    ef_rms_valid = ef_rms[valid_mask]
    labels_valid = labels[valid_mask]
    
    # 获取所有唯一的评分值
    unique_ratings = np.unique(labels_valid)
    unique_ratings = np.sort(unique_ratings)  # 排序
    
    print(f"\n📊 平衡采样 (每个评分随机取 {samples_per_rating} 个)...")
    
    ef_rms_list = []
    labels_list = []
    original_counts = {}
    sampled_counts = {}
    
    for rating in unique_ratings:
        # 找到该评分的所有索引
        rating_mask = (labels_valid == rating)
        rating_indices = np.where(rating_mask)[0]
        n_original = len(rating_indices)
        original_counts[rating] = n_original
        
        # 如果该评分的样本数少于目标数量，使用全部样本
        n_sample = min(samples_per_rating, n_original)
        
        # 随机采样
        if n_original > 0:
            sampled_indices = np.random.choice(rating_indices, size=n_sample, replace=False)
            ef_rms_list.append(ef_rms_valid[sampled_indices])
            labels_list.append(labels_valid[sampled_indices])
            sampled_counts[rating] = n_sample
            print(f"   评分 {rating}: 原始 {n_original} 个 -> 采样 {n_sample} 个")
    
    # 合并所有采样结果
    ef_rms_balanced = np.concatenate(ef_rms_list)
    labels_balanced = np.concatenate(labels_list)
    
    print(f"   总样本数: {len(ef_rms_valid)} -> {len(ef_rms_balanced)}")
    
    return ef_rms_balanced, labels_balanced, original_counts, sampled_counts


def plot_scatter(ef_rms: np.ndarray, labels: np.ndarray, correlation: float, 
                 p_value: float, output_path: Path, figsize=(10, 8)):
    """
    绘制ef RMS与labels的散点图（使用平衡采样后的数据）
    
    参数:
        ef_rms: (N,) - ef模长RMS值（已平衡采样）
        labels: (N,) - 被试评分labels（已平衡采样）
        correlation: float - Spearman相关系数
        p_value: float - p值
        output_path: Path - 输出图片路径
        figsize: tuple - 图表大小
    """
    print(f"\n📊 绘制散点图...")
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 绘制散点图，使用alpha透明度以便看到重叠点
    scatter = ax.scatter(ef_rms, labels, alpha=0.5, s=30, 
                        c=labels, cmap='viridis', edgecolors='none')
    
    # 设置标签和标题
    ax.set_xlabel('EF模长RMS', fontsize=14, fontweight='bold')
    ax.set_ylabel('评分Label', fontsize=14, fontweight='bold')
    ax.set_title(f'EF模长RMS vs 评分Label分布（平衡采样）\nSpearman相关系数: {correlation:.4f} (p={p_value:.2e})', 
                fontsize=14, fontweight='bold', pad=15)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('评分Label', fontsize=12)
    
    # 添加趋势线
    if len(ef_rms) > 1:
        # 计算线性拟合
        z = np.polyfit(ef_rms, labels, 1)
        p = np.poly1d(z)
        x_line = np.linspace(ef_rms.min(), ef_rms.max(), 100)
        y_line = p(x_line)
        ax.plot(x_line, y_line, "r--", alpha=0.8, linewidth=2, label='线性拟合')
        ax.legend(loc='best', fontsize=11)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ 散点图已保存到: {output_path}")


def plot_boxplot(ef_rms: np.ndarray, labels: np.ndarray, correlation: float,
                 p_value: float, output_path: Path, figsize=(10, 8)):
    """
    绘制ef RMS按评分分组的箱线图（使用平衡采样后的数据）
    
    参数:
        ef_rms: (N,) - ef模长RMS值（已平衡采样）
        labels: (N,) - 被试评分labels（已平衡采样）
        correlation: float - Spearman相关系数
        p_value: float - p值
        output_path: Path - 输出图片路径
        figsize: tuple - 图表大小
    """
    print(f"\n📊 绘制箱线图...")
    
    # 按评分分组数据
    unique_ratings = np.sort(np.unique(labels))
    data_by_rating = [ef_rms[labels == rating] for rating in unique_ratings]
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 绘制箱线图
    bp = ax.boxplot(data_by_rating, labels=[f'评分 {int(r)}' for r in unique_ratings],
                    patch_artist=True, showmeans=True, meanline=True)
    
    # 美化箱线图
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_ratings)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 设置标签和标题
    ax.set_xlabel('评分Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('EF模长RMS', fontsize=14, fontweight='bold')
    ax.set_title(f'EF模长RMS按评分分组分布（平衡采样）\nSpearman相关系数: {correlation:.4f} (p={p_value:.2e})', 
                fontsize=14, fontweight='bold', pad=15)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 旋转x轴标签
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ 箱线图已保存到: {output_path}")


def save_results(output_path: Path, correlation: float, p_value: float, 
                 ef_rms: np.ndarray, labels: np.ndarray, n_valid: int):
    """保存结果到文件"""
    results = {
        'correlation': float(correlation),
        'p_value': float(p_value),
        'n_valid': int(n_valid),
        'n_total': int(len(ef_rms)),
        'ef_rms_stats': {
            'mean': float(np.mean(ef_rms)),
            'std': float(np.std(ef_rms)),
            'min': float(np.min(ef_rms)),
            'max': float(np.max(ef_rms)),
            'median': float(np.median(ef_rms))
        },
        'labels_stats': {
            'mean': float(np.mean(labels)),
            'std': float(np.std(labels)),
            'min': float(np.min(labels)),
            'max': float(np.max(labels)),
            'median': float(np.median(labels))
        }
    }
    
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='计算ef冲突量与labels的相关性')
    parser.add_argument('--dataset', type=str, 
                       default='data/training_dataset_random',
                       help='数据集路径（默认: data/training_dataset_random）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON文件路径（默认: {dataset}/ef_correlation_results.json）')
    parser.add_argument('--scatter-output', type=str, default=None,
                       help='散点图输出路径（默认: {dataset}/ef_correlation_scatter.png）')
    parser.add_argument('--boxplot-output', type=str, default=None,
                       help='箱线图输出路径（默认: {dataset}/ef_correlation_boxplot.png）')
    parser.add_argument('--samples-per-rating', type=int, default=300,
                       help='每个评分等级采样的数量（默认: 300）')
    
    args = parser.parse_args()
    
    # 解析路径
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = dataset_path / "ef_correlation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 确定散点图输出路径
    if args.scatter_output:
        scatter_output_path = Path(args.scatter_output)
    else:
        scatter_output_path = dataset_path / "ef_correlation_scatter.png"
    scatter_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 确定箱线图输出路径
    if args.boxplot_output:
        boxplot_output_path = Path(args.boxplot_output)
    else:
        boxplot_output_path = dataset_path / "ef_correlation_boxplot.png"
    boxplot_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"EF冲突量与Labels相关性分析")
    print(f"{'='*60}")
    print(f"数据集路径: {dataset_path}")
    print(f"输出路径: {output_path}")
    
    try:
        # 1. 加载数据
        imu_windows, labels = load_data(dataset_path)
        
        # 2. 提取ef冲突量（索引12-14）
        ef_conflicts = extract_ef_conflicts(imu_windows)
        
        # 3. 计算ef模长RMS
        ef_rms = compute_ef_magnitude_rms(ef_conflicts)
        
        # 4. 计算相关性
        correlation, p_value, n_valid = compute_correlation(ef_rms, labels)
        
        # 5. 保存结果
        save_results(output_path, correlation, p_value, ef_rms, labels, n_valid)
        
        # 6. 平衡采样（每个评分随机取指定数量）
        ef_rms_balanced, labels_balanced, original_counts, sampled_counts = balance_sample_by_rating(
            ef_rms, labels, samples_per_rating=args.samples_per_rating
        )
        
        # 7. 计算平衡采样后的相关系数（用于图表显示）
        correlation_balanced, p_value_balanced = spearmanr(ef_rms_balanced, labels_balanced)
        
        # 8. 绘制散点图
        plot_scatter(ef_rms_balanced, labels_balanced, correlation_balanced, 
                    p_value_balanced, scatter_output_path)
        
        # 9. 绘制箱线图
        plot_boxplot(ef_rms_balanced, labels_balanced, correlation_balanced,
                    p_value_balanced, boxplot_output_path)
        
        print(f"\n{'='*60}")
        print(f"✅ 分析完成！")
        print(f"{'='*60}")
        print(f"\n总结:")
        print(f"  - EF冲突量索引: 12-14")
        print(f"  - 有效样本数: {n_valid}/{len(ef_rms)}")
        print(f"  - 平衡采样后样本数: {len(ef_rms_balanced)}")
        print(f"  - Spearman相关系数: {correlation:.6f} (全部数据)")
        print(f"  - Spearman相关系数: {correlation_balanced:.6f} (平衡采样)")
        print(f"  - p值: {p_value:.6f}")
        print(f"  - 散点图已保存: {scatter_output_path}")
        print(f"  - 箱线图已保存: {boxplot_output_path}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

