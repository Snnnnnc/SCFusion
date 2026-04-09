#!/usr/bin/env python3
"""
车辆运动状态数据处理和可视化

功能：
1. 从states文件夹读取车辆运动状态数据（地面坐标系）
2. 根据速度计算车头朝向，将速度和加速度转换到车身坐标系
3. 绘制横向、纵向的速度和加速度时序图
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# 设置中文字体
available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
preferred_fonts = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Songti SC', 'SimHei', 'Arial Unicode MS']
for f in preferred_fonts:
    if f in available_fonts:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break


def compute_heading_from_velocity(vx, vy, smoothing_window=5):
    """
    根据速度计算车头朝向角（弧度）
    
    参数:
        vx: X方向速度
        vy: Y方向速度
        smoothing_window: 平滑窗口大小（用于减少噪声）
    
    返回:
        heading: 车头朝向角（弧度），范围[-pi, pi]，0表示沿X轴正方向
    """
    # 计算瞬时朝向角
    speed_horiz = np.sqrt(vx**2 + vy**2)
    
    # 对于速度很小的时刻，使用前一时刻的朝向
    heading = np.zeros_like(vx)
    heading[0] = np.arctan2(vy[0], vx[0]) if speed_horiz[0] > 0.01 else 0.0
    
    for i in range(1, len(vx)):
        if speed_horiz[i] > 0.01:  # 速度阈值，避免噪声
            heading[i] = np.arctan2(vy[i], vx[i])
        else:
            heading[i] = heading[i-1]  # 速度太小时保持前一时刻朝向
    
    # 处理角度不连续性（-pi到pi的跳变）
    # 将角度展开为连续序列，避免-pi到pi的跳变
    heading_unwrapped = np.unwrap(heading)
    
    # 平滑处理（简单移动平均）
    if smoothing_window > 1:
        heading_smooth = np.zeros_like(heading_unwrapped)
        half_window = smoothing_window // 2
        
        for i in range(len(heading_unwrapped)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(heading_unwrapped), i + half_window + 1)
            heading_smooth[i] = np.mean(heading_unwrapped[start_idx:end_idx])
        
        heading_unwrapped = heading_smooth
    
    # 将角度归一化到[-pi, pi]
    heading = np.arctan2(np.sin(heading_unwrapped), np.cos(heading_unwrapped))
    
    return heading


def transform_to_body_frame(vx, vy, ax, ay, heading):
    """
    将速度和加速度从地面坐标系转换到车身坐标系
    
    车身坐标系定义：
    - 纵向（longitudinal）：沿车头方向（行驶方向）
    - 横向（lateral）：垂直于车头方向（左侧为正）
    
    参数:
        vx: X方向速度（地面坐标系）
        vy: Y方向速度（地面坐标系）
        ax: X方向加速度（地面坐标系）
        ay: Y方向加速度（地面坐标系）
        heading: 车头朝向角（弧度）
    
    返回:
        v_long: 纵向速度（车身坐标系）
        v_lat: 横向速度（车身坐标系）
        a_long: 纵向加速度（车身坐标系）
        a_lat: 横向加速度（车身坐标系）
    """
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    
    # 速度转换：从地面坐标系到车身坐标系
    # 车身坐标系的X轴（纵向）沿车头方向，Y轴（横向）垂直于车头方向（左侧为正）
    v_long = vx * cos_heading + vy * sin_heading
    v_lat = -vx * sin_heading + vy * cos_heading
    
    # 加速度转换
    a_long = ax * cos_heading + ay * sin_heading
    a_lat = -ax * sin_heading + ay * cos_heading
    
    return v_long, v_lat, a_long, a_lat


def load_states_file(file_path: Path):
    """加载车辆状态CSV文件"""
    print(f"📂 加载文件: {file_path.name}")
    df = pd.read_csv(file_path)
    
    # 提取需要的数据
    time = df['Time(s)'].values
    vx = df['Vx(m/s)'].values
    vy = df['Vy(m/s)'].values
    ax = df['Ax(m/s²)'].values
    ay = df['Ay(m/s²)'].values
    
    print(f"   数据点数: {len(time)}")
    print(f"   时间范围: {time[0]:.2f}s - {time[-1]:.2f}s")
    print(f"   总时长: {time[-1] - time[0]:.2f}s")
    
    return time, vx, vy, ax, ay


def plot_vehicle_states(time, v_long, v_lat, a_long, a_lat, map_name, output_path: Path, figsize=(15, 10)):
    """
    绘制车辆横向、纵向的速度和加速度时序图
    
    参数:
        time: 时间序列
        v_long: 纵向速度
        v_lat: 横向速度
        a_long: 纵向加速度
        a_lat: 横向加速度
        map_name: 地图名称
        output_path: 输出图片路径
        figsize: 图表大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    
    # 1. 纵向速度
    ax1 = axes[0, 0]
    ax1.plot(time, v_long, 'b-', linewidth=1.0, alpha=0.8)
    ax1.set_ylabel('纵向速度 (m/s)', fontsize=12, fontweight='bold')
    ax1.set_title('纵向速度时序图', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 横向速度
    ax2 = axes[0, 1]
    ax2.plot(time, v_lat, 'r-', linewidth=1.0, alpha=0.8)
    ax2.set_ylabel('横向速度 (m/s)', fontsize=12, fontweight='bold')
    ax2.set_title('横向速度时序图', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 纵向加速度
    ax3 = axes[1, 0]
    ax3.plot(time, a_long, 'b-', linewidth=1.0, alpha=0.8)
    ax3.set_xlabel('时间 (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('纵向加速度 (m/s²)', fontsize=12, fontweight='bold')
    ax3.set_title('纵向加速度时序图', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 横向加速度
    ax4 = axes[1, 1]
    ax4.plot(time, a_lat, 'r-', linewidth=1.0, alpha=0.8)
    ax4.set_xlabel('时间 (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('横向加速度 (m/s²)', fontsize=12, fontweight='bold')
    ax4.set_title('横向加速度时序图', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 添加总标题
    fig.suptitle(f'车辆运动状态 - {map_name}\n（车身坐标系）', 
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ 图表已保存: {output_path}")


def process_states_file(states_file: Path, output_dir: Path, smoothing_window=5):
    """
    处理单个车辆状态文件
    
    参数:
        states_file: 状态文件路径
        output_dir: 输出目录
        smoothing_window: 朝向角平滑窗口大小
    """
    # 提取地图名称
    # 文件名格式: recording_vehicle_tesla_model3_Town01_20251109_014612_states.csv
    parts = states_file.stem.split('_')
    # 找到Town开头的部分
    map_name = None
    for part in parts:
        if part.startswith('Town'):
            map_name = part
            break
    if map_name is None:
        map_name = states_file.stem  # 如果找不到，使用文件名
    
    # 加载数据
    time, vx, vy, ax, ay = load_states_file(states_file)
    
    # 计算车头朝向
    print(f"\n🔄 计算车头朝向...")
    heading = compute_heading_from_velocity(vx, vy, smoothing_window=smoothing_window)
    print(f"   朝向角范围: {np.min(heading)*180/np.pi:.2f}° - {np.max(heading)*180/np.pi:.2f}°")
    
    # 转换到车身坐标系
    print(f"\n🔄 转换到车身坐标系...")
    v_long, v_lat, a_long, a_lat = transform_to_body_frame(vx, vy, ax, ay, heading)
    
    # 打印统计信息
    print(f"\n📊 车身坐标系统计信息:")
    print(f"   纵向速度: 均值={np.mean(v_long):.3f} m/s, 范围=[{np.min(v_long):.3f}, {np.max(v_long):.3f}] m/s")
    print(f"   横向速度: 均值={np.mean(v_lat):.3f} m/s, 范围=[{np.min(v_lat):.3f}, {np.max(v_lat):.3f}] m/s")
    print(f"   纵向加速度: 均值={np.mean(a_long):.3f} m/s², 范围=[{np.min(a_long):.3f}, {np.max(a_long):.3f}] m/s²")
    print(f"   横向加速度: 均值={np.mean(a_lat):.3f} m/s², 范围=[{np.min(a_lat):.3f}, {np.max(a_lat):.3f}] m/s²")
    
    # 绘制图表
    output_path = output_dir / f"{map_name}_vehicle_states.png"
    plot_vehicle_states(time, v_long, v_lat, a_long, a_lat, map_name, output_path)
    
    return {
        'map_name': map_name,
        'time': time,
        'v_long': v_long,
        'v_lat': v_lat,
        'a_long': a_long,
        'a_lat': a_lat,
        'heading': heading
    }


def main():
    parser = argparse.ArgumentParser(description='处理车辆运动状态数据并绘制时序图')
    parser.add_argument('--states-dir', type=str, default='states',
                       help='车辆状态文件目录（默认: states）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认: states/output）')
    parser.add_argument('--smoothing-window', type=int, default=5,
                       help='朝向角平滑窗口大小（默认: 5）')
    parser.add_argument('--map', type=str, default=None,
                       help='指定处理的地图（如Town01），默认处理所有地图')
    
    args = parser.parse_args()
    
    # 解析路径
    states_dir = Path(args.states_dir)
    if not states_dir.exists():
        print(f"❌ 错误: 状态文件目录不存在: {states_dir}")
        sys.exit(1)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = states_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"车辆运动状态数据处理")
    print(f"{'='*60}")
    print(f"状态文件目录: {states_dir}")
    print(f"输出目录: {output_dir}")
    
    # 查找所有状态文件
    if args.map:
        pattern = f"*{args.map}*states.csv"
        states_files = list(states_dir.glob(pattern))
        if not states_files:
            print(f"❌ 错误: 未找到匹配的地图文件: {args.map}")
            sys.exit(1)
    else:
        states_files = sorted(states_dir.glob("*states.csv"))
        if not states_files:
            print(f"❌ 错误: 未找到状态文件 (*states.csv)")
            sys.exit(1)
    
    print(f"\n找到 {len(states_files)} 个状态文件\n")
    
    # 处理每个文件
    results = []
    for states_file in states_files:
        try:
            result = process_states_file(states_file, output_dir, 
                                       smoothing_window=args.smoothing_window)
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ 处理文件 {states_file.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"{'='*60}")
    print(f"✅ 处理完成！")
    print(f"{'='*60}")
    print(f"\n处理了 {len(results)} 个文件")
    print(f"所有图表已保存到: {output_dir}")


if __name__ == '__main__':
    main()

