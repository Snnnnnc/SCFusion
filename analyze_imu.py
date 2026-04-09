#!/usr/bin/env python3
"""
IMU数据分析脚本
分析IMU数据的质量，包括每个维度的数据范围、统计信息等，并绘制时间序列图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 先尝试查找系统中可用的中文字体
available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
preferred_fonts = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Songti SC', 'SimHei']

for f in preferred_fonts:
    if f in available_fonts:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        print(f"✅ 使用字体: {f}")
        break
else:
    print("⚠️ 未找到中文字体，使用默认字体。可尝试手动安装 PingFang SC。")

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# IMU通道名称（9个通道：顺序与BDF文件中的通道名称对应）
# 实际顺序：GYR-X, GYR-Y, GYR-Z, ACC-X, ACC-Y, ACC-Z, MAG-X, MAG-Y, MAG-Z
IMU_CHANNEL_NAMES = [
    'GYR-X', 'GYR-Y', 'GYR-Z',      # 陀螺仪 (rad/s) - 角速度
    'ACC-X', 'ACC-Y', 'ACC-Z',      # 加速度计 (m/s²)
    'MAG-X', 'MAG-Y', 'MAG-Z'       # 磁力计 (μT)
]


def list_sessions_by_subject_map(processed_root: Path, subject: str, map_id: str):
    """根据subject和map_id查找所有匹配的session目录"""
    base = processed_root / subject
    if not base.exists():
        return []
    sessions = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        parts = p.name.split("_")
        if len(parts) >= 3 and parts[2] == map_id:
            if (p / "_modalities").exists():
                sessions.append(p)
    return sessions


def load_imu_data(session_dir: Path):
    """加载IMU数据"""
    imu_path = session_dir / "_modalities" / "imu.npy"
    if not imu_path.exists():
        raise FileNotFoundError(f"未找到IMU数据文件: {imu_path}")
    
    imu_data = np.load(imu_path)
    return imu_data


def get_sampling_rate(session_dir: Path):
    """从modalities.json或npz文件获取IMU采样率"""
    # 优先从modalities.json读取IMU的采样率
    modalities_path = session_dir / "_modalities" / "modalities.json"
    if modalities_path.exists():
        with open(modalities_path, 'r', encoding='utf-8') as f:
            modalities = json.load(f)
            if 'imu' in modalities and 'srate' in modalities['imu']:
                imu_srate = int(modalities['imu']['srate'])
                return imu_srate
    
    # 回退到npz文件
    npz_path = session_dir / f"{session_dir.name}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        if 'srate' in data:
            srate = int(data['srate'])
            # 如果是1000Hz，IMU可能是100Hz（下采样后）
            if srate == 1000:
                return 100  # IMU数据应该是100Hz
            return srate
    
    return 100  # 默认IMU采样率（因为原始数据是100Hz）


def analyze_imu_quality(imu_data, session_name, srate):
    """分析IMU数据质量"""
    n_channels, n_samples = imu_data.shape
    duration = n_samples / srate  # 时长（秒）
    
    print(f"\n{'='*80}")
    print(f"IMU数据质量分析: {session_name}")
    print(f"{'='*80}")
    print(f"\n基本信息:")
    print(f"  数据形状: {imu_data.shape} (通道数 × 采样点数)")
    print(f"  采样率: {srate} Hz")
    print(f"  数据时长: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    print(f"  数据类型: {imu_data.dtype}")
    
    # 检查缺失值和异常值
    print(f"\n数据完整性检查:")
    nan_count = np.isnan(imu_data).sum()
    inf_count = np.isinf(imu_data).sum()
    print(f"  NaN值数量: {nan_count}")
    print(f"  Inf值数量: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print(f"  ⚠️ 警告: 数据中存在异常值！")
    else:
        print(f"  ✅ 数据完整，无缺失值和异常值")
    
    # 每个通道的统计信息
    print(f"\n各通道统计信息:")
    print(f"{'通道':<10} {'均值':<15} {'标准差':<15} {'最小值':<15} {'最大值':<15} {'中位数':<15}")
    print("-" * 100)
    
    stats_dict = {}
    for i in range(n_channels):
        channel_data = imu_data[i, :]
        channel_name = IMU_CHANNEL_NAMES[i] if i < len(IMU_CHANNEL_NAMES) else f"Channel_{i}"
        
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        median_val = np.median(channel_data)
        
        stats_dict[channel_name] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'median': median_val,
            'data': channel_data
        }
        
        print(f"{channel_name:<10} {mean_val:<15.6f} {std_val:<15.6f} {min_val:<15.6f} {max_val:<15.6f} {median_val:<15.6f}")
    
    # 数据范围分析
    print(f"\n数据范围分析:")
    for i, channel_name in enumerate(IMU_CHANNEL_NAMES[:n_channels]):
        channel_data = imu_data[i, :]
        data_range = np.max(channel_data) - np.min(channel_data)
        print(f"  {channel_name}: 范围 [{np.min(channel_data):.4f}, {np.max(channel_data):.4f}], 跨度 {data_range:.4f}")
    
    # 检查数据是否全为0或常数
    print(f"\n数据变化性检查:")
    for i, channel_name in enumerate(IMU_CHANNEL_NAMES[:n_channels]):
        channel_data = imu_data[i, :]
        if np.all(channel_data == channel_data[0]):
            print(f"  ⚠️ {channel_name}: 数据为常数 ({channel_data[0]:.6f})")
        elif np.std(channel_data) < 1e-6:
            print(f"  ⚠️ {channel_name}: 数据变化极小 (std={np.std(channel_data):.6e})")
        else:
            print(f"  ✅ {channel_name}: 数据正常变化 (std={np.std(channel_data):.6f})")
    
    return stats_dict, duration


def plot_imu_timeseries(imu_data, session_name, srate, output_dir):
    """绘制IMU各维度的时间序列图"""
    n_channels, n_samples = imu_data.shape
    duration = n_samples / srate
    time_axis = np.arange(n_samples) / srate
    
    # 创建子图：3行3列，每个通道一个图
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'IMU数据时间序列分析 - {session_name}', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    for i in range(n_channels):
        ax = axes[i]
        channel_name = IMU_CHANNEL_NAMES[i] if i < len(IMU_CHANNEL_NAMES) else f"Channel_{i}"
        channel_data = imu_data[i, :]
        
        # 绘制时间序列
        ax.plot(time_axis, channel_data, color=colors[i % len(colors)], linewidth=0.5, alpha=0.7)
        ax.set_xlabel('时间 (秒)', fontsize=10)
        ax.set_ylabel(f'{channel_name}', fontsize=10)
        ax.set_title(f'{channel_name} 时间序列', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'均值: {mean_val:.4f}')
        ax.axhline(y=mean_val + std_val, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=mean_val - std_val, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)
    
    # 隐藏多余的子图
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f'{session_name}_imu_timeseries.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 时间序列图已保存: {output_path}")
    plt.close()


def plot_imu_statistics(imu_data, session_name, output_dir):
    """绘制IMU统计信息图"""
    n_channels = imu_data.shape[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'IMU数据统计信息 - {session_name}', fontsize=16, fontweight='bold')
    
    # 1. 各通道均值对比
    ax1 = axes[0, 0]
    means = [np.mean(imu_data[i, :]) for i in range(n_channels)]
    channel_names = [IMU_CHANNEL_NAMES[i] if i < len(IMU_CHANNEL_NAMES) else f"Ch_{i}" 
                     for i in range(n_channels)]
    ax1.barh(channel_names, means, color='skyblue', alpha=0.7)
    ax1.set_xlabel('均值', fontsize=11)
    ax1.set_title('各通道均值', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. 各通道标准差对比
    ax2 = axes[0, 1]
    stds = [np.std(imu_data[i, :]) for i in range(n_channels)]
    ax2.barh(channel_names, stds, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('标准差', fontsize=11)
    ax2.set_title('各通道标准差', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. 各通道数据范围（箱线图）
    ax3 = axes[1, 0]
    data_for_boxplot = [imu_data[i, :] for i in range(n_channels)]
    bp = ax3.boxplot(data_for_boxplot, labels=channel_names, vert=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    ax3.set_ylabel('数值', fontsize=11)
    ax3.set_title('各通道数据分布（箱线图）', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 各通道最大值和最小值对比
    ax4 = axes[1, 1]
    x_pos = np.arange(n_channels)
    max_vals = [np.max(imu_data[i, :]) for i in range(n_channels)]
    min_vals = [np.min(imu_data[i, :]) for i in range(n_channels)]
    width = 0.35
    ax4.bar(x_pos - width/2, max_vals, width, label='最大值', color='coral', alpha=0.7)
    ax4.bar(x_pos + width/2, min_vals, width, label='最小值', color='steelblue', alpha=0.7)
    ax4.set_xlabel('通道', fontsize=11)
    ax4.set_ylabel('数值', fontsize=11)
    ax4.set_title('各通道最大值和最小值', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(channel_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    output_path = output_dir / f'{session_name}_imu_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 统计信息图已保存: {output_path}")
    plt.close()


def plot_imu_correlation(imu_data, session_name, output_dir):
    """绘制IMU通道间相关性热图"""
    n_channels = imu_data.shape[0]
    
    # 计算相关性矩阵
    correlation_matrix = np.corrcoef(imu_data)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    channel_names = [IMU_CHANNEL_NAMES[i] if i < len(IMU_CHANNEL_NAMES) else f"Ch_{i}" 
                     for i in range(n_channels)]
    
    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_channels))
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.set_yticklabels(channel_names)
    ax.set_title(f'IMU通道相关性热图 - {session_name}', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(n_channels):
        for j in range(n_channels):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='相关系数')
    plt.tight_layout()
    output_path = output_dir / f'{session_name}_imu_correlation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 相关性热图已保存: {output_path}")
    plt.close()


def save_statistics_to_file(stats_dict, session_name, duration, srate, output_dir):
    """保存统计数据到文件"""
    stats_file = output_dir / f'{session_name}_imu_statistics.txt'
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"IMU数据统计分析报告 - {session_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"基本信息:\n")
        f.write(f"  采样率: {srate} Hz\n")
        f.write(f"  数据时长: {duration:.2f} 秒 ({duration/60:.2f} 分钟)\n")
        f.write(f"  通道数: {len(stats_dict)}\n\n")
        
        f.write("各通道详细统计:\n")
        f.write("-"*80 + "\n")
        for channel_name, stats in stats_dict.items():
            f.write(f"\n{channel_name}:\n")
            f.write(f"  均值: {stats['mean']:.6f}\n")
            f.write(f"  标准差: {stats['std']:.6f}\n")
            f.write(f"  最小值: {stats['min']:.6f}\n")
            f.write(f"  最大值: {stats['max']:.6f}\n")
            f.write(f"  中位数: {stats['median']:.6f}\n")
            f.write(f"  数据范围: {stats['max'] - stats['min']:.6f}\n")
            
            # 计算分位数
            q25 = np.percentile(stats['data'], 25)
            q75 = np.percentile(stats['data'], 75)
            f.write(f"  25%分位: {q25:.6f}\n")
            f.write(f"  75%分位: {q75:.6f}\n")
            f.write(f"  四分位距(IQR): {q75 - q25:.6f}\n")
    
    print(f"✅ 统计报告已保存: {stats_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="IMU数据分析脚本")
    parser.add_argument("--subject", type=str, required=True, help="被试姓名（如 zzh）")
    parser.add_argument("--map", dest="map_id", type=str, required=True, help="地图编号（如 02）")
    parser.add_argument("--output", type=str, default=None, help="输出目录（默认：processed/{subject}/imu_analysis_{map_id}）")
    args = parser.parse_args()
    
    # 确定路径
    script_dir = Path(__file__).resolve().parent
    processed_root = script_dir / "processed"
    
    if not processed_root.exists():
        raise FileNotFoundError(f"未找到processed目录: {processed_root}")
    
    # 查找匹配的sessions
    sessions = list_sessions_by_subject_map(processed_root, args.subject, args.map_id)
    
    if len(sessions) == 0:
        print(f"[错误] 未找到匹配的会话（subject={args.subject}, map={args.map_id}）")
        return
    
    print(f"[信息] 找到 {len(sessions)} 个匹配的会话")
    
    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = processed_root / args.subject / f"imu_analysis_{args.map_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[信息] 输出目录: {output_dir}")
    
    # 处理每个session
    for session_dir in sessions:
        session_name = session_dir.name
        print(f"\n{'='*80}")
        print(f"处理会话: {session_name}")
        print(f"{'='*80}")
        
        try:
            # 加载数据
            imu_data = load_imu_data(session_dir)
            srate = get_sampling_rate(session_dir)
            
            # 分析数据质量
            stats_dict, duration = analyze_imu_quality(imu_data, session_name, srate)
            
            # 绘制图表
            print(f"\n📈 开始生成分析图表...")
            plot_imu_timeseries(imu_data, session_name, srate, output_dir)
            plot_imu_statistics(imu_data, session_name, output_dir)
            plot_imu_correlation(imu_data, session_name, output_dir)
            
            # 保存统计数据
            save_statistics_to_file(stats_dict, session_name, duration, srate, output_dir)
            
            print(f"\n✅ 会话 {session_name} 分析完成！")
            
        except Exception as e:
            print(f"❌ 处理会话 {session_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ 所有分析完成！结果已保存到: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

