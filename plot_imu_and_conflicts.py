#!/usr/bin/env python3
"""
绘制IMU原始数据和前庭冲突误差的时序图

功能：
1. 绘制原始加速度、角速度时序图
2. 绘制加速度误差、角速度误差时序图
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
import sys

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


def load_data(session_dir: Path):
    """加载IMU数据和冲突数据"""
    modalities_dir = session_dir / "_modalities"
    
    # 加载IMU数据
    imu_path = modalities_dir / "imu.npy"
    if not imu_path.exists():
        raise FileNotFoundError(f"未找到IMU数据文件: {imu_path}")
    imu_data = np.load(imu_path)  # shape: (9, N)
    
    # 加载冲突数据
    conflicts_path = modalities_dir / "vestibular_conflicts.npy"
    if not conflicts_path.exists():
        raise FileNotFoundError(f"未找到冲突数据文件: {conflicts_path}")
    conflicts_data = np.load(conflicts_path)  # shape: (12, N)
    
    # 加载元数据获取采样率
    metadata_path = modalities_dir / "vestibular_conflicts_metadata.json"
    srate = 100  # 默认采样率
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            srate = metadata.get('srate', 100)
    
    return imu_data, conflicts_data, srate


def extract_channels(imu_data: np.ndarray, conflicts_data: np.ndarray):
    """
    提取需要的通道数据
    
    参数:
        imu_data: (9, N) - [GYR-X, GYR-Y, GYR-Z, ACC-X, ACC-Y, ACC-Z, MAG-X, MAG-Y, MAG-Z]
        conflicts_data: (12, N) - [e_scc_x, e_scc_y, e_scc_z, e_oto_x, e_oto_y, e_oto_z, ...]
    
    返回:
        gyro: (3, N) - 角速度 [deg/s]
        acc: (3, N) - 加速度 [G]
        e_scc: (3, N) - 角速度误差
        e_oto: (3, N) - 加速度误差
    """
    # IMU数据：前3个是角速度，接下来3个是加速度
    gyro = imu_data[0:3, :]  # (3, N) GYR-X, GYR-Y, GYR-Z
    acc = imu_data[3:6, :]   # (3, N) ACC-X, ACC-Y, ACC-Z
    
    # 冲突数据：前3个是e_scc（角速度误差），接下来3个是e_oto（加速度误差）
    e_scc = conflicts_data[0:3, :]  # (3, N) e_scc_x, e_scc_y, e_scc_z
    e_oto = conflicts_data[3:6, :]  # (3, N) e_oto_x, e_oto_y, e_oto_z
    
    return gyro, acc, e_scc, e_oto


def plot_timeseries(data: np.ndarray, srate: float, title: str, ylabel: str, 
                    labels: list, output_path: Path, figsize=(15, 6)):
    """
    绘制时序图
    
    参数:
        data: (3, N) - 3个通道的数据
        srate: 采样率 (Hz)
        title: 图表标题
        ylabel: Y轴标签
        labels: 通道标签列表 ['X', 'Y', 'Z']
        output_path: 输出文件路径
        figsize: 图表大小
    """
    n_channels, n_samples = data.shape
    time = np.arange(n_samples) / srate  # 时间轴（秒）
    
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
    
    for i in range(n_channels):
        ax = axes[i]
        ax.plot(time, data[i, :], color=colors[i], linewidth=0.5, alpha=0.8)
        ax.set_ylabel(f'{labels[i]} ({ylabel})', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{title} - {labels[i]}轴', fontsize=12, pad=10)
    
    axes[-1].set_xlabel('时间 (秒)', fontsize=11)
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {output_path}")


def plot_all_channels(data: np.ndarray, srate: float, title: str, ylabel: str,
                     labels: list, output_path: Path, figsize=(15, 8)):
    """
    在同一张图上绘制所有通道的时序图
    
    参数:
        data: (3, N) - 3个通道的数据
        srate: 采样率 (Hz)
        title: 图表标题
        ylabel: Y轴标签
        labels: 通道标签列表 ['X', 'Y', 'Z']
        output_path: 输出文件路径
        figsize: 图表大小
    """
    n_channels, n_samples = data.shape
    time = np.arange(n_samples) / srate  # 时间轴（秒）
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
    channel_names = ['X', 'Y', 'Z']
    
    for i in range(n_channels):
        ax.plot(time, data[i, :], color=colors[i], linewidth=0.5, alpha=0.7, 
                label=f'{channel_names[i]}轴')
    
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制IMU原始数据和前庭冲突误差的时序图')
    parser.add_argument('--session', type=str, required=True,
                       help='会话路径，例如: processed/cyz/20251108182115_cyz_01_cyz')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认：会话目录/_modalities）')
    parser.add_argument('--separate', action='store_true',
                       help='为每个轴创建单独的图表（默认：所有轴在同一张图）')
    
    args = parser.parse_args()
    
    # 解析会话路径
    session_path = Path(args.session)
    if not session_path.exists():
        print(f"❌ 错误: 会话路径不存在: {session_path}")
        sys.exit(1)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = session_path / "_modalities"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 会话目录: {session_path}")
    print(f"📁 输出目录: {output_dir}")
    
    # 加载数据
    print("\n📊 加载数据...")
    try:
        imu_data, conflicts_data, srate = load_data(session_path)
        print(f"   IMU数据形状: {imu_data.shape}")
        print(f"   冲突数据形状: {conflicts_data.shape}")
        print(f"   采样率: {srate} Hz")
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        sys.exit(1)
    
    # 提取通道
    print("\n🔍 提取通道数据...")
    gyro, acc, e_scc, e_oto = extract_channels(imu_data, conflicts_data)
    print(f"   角速度形状: {gyro.shape}")
    print(f"   加速度形状: {acc.shape}")
    print(f"   角速度误差形状: {e_scc.shape}")
    print(f"   加速度误差形状: {e_oto.shape}")
    
    # 检查数据长度是否匹配
    if gyro.shape[1] != e_scc.shape[1]:
        print(f"⚠️  警告: IMU数据长度 ({gyro.shape[1]}) 与冲突数据长度 ({e_scc.shape[1]}) 不匹配")
        print(f"   将使用较短的长度进行绘制")
        min_len = min(gyro.shape[1], e_scc.shape[1])
        gyro = gyro[:, :min_len]
        acc = acc[:, :min_len]
        e_scc = e_scc[:, :min_len]
        e_oto = e_oto[:, :min_len]
    
    # 绘制图表
    print("\n📈 绘制时序图...")
    
    if args.separate:
        # 为每个轴创建单独的图表
        plot_timeseries(gyro, srate, '原始角速度', 'deg/s', ['X', 'Y', 'Z'],
                       output_dir / 'gyro_timeseries.png')
        plot_timeseries(acc, srate, '原始加速度', 'G', ['X', 'Y', 'Z'],
                       output_dir / 'acc_timeseries.png')
        plot_timeseries(e_scc, srate, '角速度误差 (e_scc)', 'deg/s', ['X', 'Y', 'Z'],
                       output_dir / 'e_scc_timeseries.png')
        plot_timeseries(e_oto, srate, '加速度误差 (e_oto)', 'G', ['X', 'Y', 'Z'],
                       output_dir / 'e_oto_timeseries.png')
    else:
        # 所有轴在同一张图
        plot_all_channels(gyro, srate, '原始角速度时序图', '角速度 (deg/s)',
                         ['X', 'Y', 'Z'], output_dir / 'gyro_timeseries.png')
        plot_all_channels(acc, srate, '原始加速度时序图', '加速度 (G)',
                         ['X', 'Y', 'Z'], output_dir / 'acc_timeseries.png')
        plot_all_channels(e_scc, srate, '角速度误差时序图 (e_scc)', '误差 (deg/s)',
                         ['X', 'Y', 'Z'], output_dir / 'e_scc_timeseries.png')
        plot_all_channels(e_oto, srate, '加速度误差时序图 (e_oto)', '误差 (G)',
                         ['X', 'Y', 'Z'], output_dir / 'e_oto_timeseries.png')
    
    print("\n✅ 所有图表绘制完成！")


if __name__ == '__main__':
    main()

