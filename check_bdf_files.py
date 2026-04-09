#!/usr/bin/env python3
"""
检查data.bdf和被试姓名.bdf文件的区别
分析哪个文件更适合用于IMU数据提取
"""

import sys
sys.path.insert(0, 'EEG_trigger')
import mne
from pathlib import Path
import numpy as np
from scipy import signal
import argparse

def analyze_bdf_file(bdf_path, name):
    """分析单个BDF文件"""
    print(f"\n{'='*80}")
    print(f"{name}: {bdf_path.name}")
    print(f"{'='*80}")
    
    raw = mne.io.read_raw_bdf(bdf_path, preload=True)
    
    print(f"\n基本信息:")
    print(f"  采样率: {raw.info['sfreq']} Hz")
    print(f"  通道数: {raw.info['nchan']}")
    print(f"  数据时长: {raw.n_times / raw.info['sfreq']:.2f} 秒")
    
    # 检查IMU通道
    imu_chs = [ch for ch in raw.info['ch_names'] if 'GYR' in ch or 'ACC' in ch or 'MAG' in ch]
    if not imu_chs:
        print(f"  ⚠️ 未找到IMU通道")
        return None
    
    print(f"\nIMU通道: {imu_chs}")
    imu_idx = [raw.ch_names.index(ch) for ch in imu_chs]
    imu_data = raw.get_data()[imu_idx, :]
    
    print(f"\nIMU数据形状: {imu_data.shape}")
    
    # 分析第一个IMU通道（GYR-X）
    gyr_x = imu_data[0, :]
    
    # 1. 数据变化性
    print(f"\n数据变化性分析:")
    unique_vals = len(np.unique(gyr_x[:10000]))
    print(f"  前10000个采样点的唯一值数量: {unique_vals}")
    print(f"  前10个采样点: {gyr_x[:10]}")
    mean_diff = np.mean(np.abs(np.diff(gyr_x[:10000])))
    print(f"  相邻点差异的平均值: {mean_diff:.6f}")
    
    # 2. 检查重复值模式（上采样特征）
    print(f"\n上采样特征检查:")
    is_repeated = True
    repeat_count = 0
    for i in range(min(100, len(gyr_x)//10)):
        segment = gyr_x[i*10:(i+1)*10]
        if np.allclose(segment, segment[0], atol=1e-6):
            repeat_count += 1
        else:
            is_repeated = False
    print(f"  每10个点重复的组数: {repeat_count}/100")
    print(f"  是否每10个点相同: {is_repeated}")
    
    # 3. 频谱分析
    print(f"\n频谱分析:")
    f, Pxx = signal.welch(gyr_x, raw.info['sfreq'], nperseg=min(8192, len(gyr_x)//4))
    idx_100 = np.argmin(np.abs(f - 100))
    energy_below_100 = np.sum(Pxx[:idx_100]) / np.sum(Pxx) * 100
    energy_above_100 = np.sum(Pxx[idx_100:]) / np.sum(Pxx) * 100
    print(f"  100Hz以下能量占比: {energy_below_100:.2f}%")
    print(f"  100Hz以上能量占比: {energy_above_100:.2f}%")
    peak_idx = np.argmax(Pxx)
    print(f"  峰值频率: {f[peak_idx]:.2f} Hz")
    
    return {
        'raw': raw,
        'imu_data': imu_data,
        'imu_chs': imu_chs,
        'srate': raw.info['sfreq'],
        'unique_vals': unique_vals,
        'mean_diff': mean_diff,
        'energy_above_100': energy_above_100,
        'is_repeated': is_repeated
    }

def main():
    parser = argparse.ArgumentParser(description="比较data.bdf和被试姓名.bdf文件")
    parser.add_argument("--subject", type=str, required=True, help="被试姓名（如 wl）")
    parser.add_argument("--session", type=str, default=None, help="会话目录名（可选）")
    args = parser.parse_args()
    
    # 查找数据目录
    data_root = Path("data")
    sessions = []
    
    for item in data_root.iterdir():
        if not item.is_dir() or not item.name.startswith('2025'):
            continue
        
        nested = item / item.name
        subject_dir = nested / args.subject
        
        # 查找data.bdf和subject.bdf
        data_bdf = None
        subject_bdf = None
        
        # 检查t目录
        t_dir = nested / 't'
        if t_dir.exists():
            if (t_dir / 'data.bdf').exists():
                data_bdf = t_dir / 'data.bdf'
            if (t_dir / f'{args.subject}.bdf').exists():
                subject_bdf = t_dir / f'{args.subject}.bdf'
        
        # 检查subject目录
        if subject_dir.exists():
            if (subject_dir / 'data.bdf').exists():
                data_bdf = subject_dir / 'data.bdf'
            if (subject_dir / f'{args.subject}.bdf').exists():
                subject_bdf = subject_dir / f'{args.subject}.bdf'
        
        if data_bdf and subject_bdf:
            sessions.append({
                'session': item.name,
                'data_bdf': data_bdf,
                'subject_bdf': subject_bdf
            })
    
    if not sessions:
        print(f"未找到包含data.bdf和{args.subject}.bdf的会话")
        return
    
    # 分析第一个找到的会话
    session = sessions[0]
    print(f"\n分析会话: {session['session']}")
    
    # 分析data.bdf
    data_info = analyze_bdf_file(session['data_bdf'], "data.bdf")
    
    # 分析subject.bdf
    subject_info = analyze_bdf_file(session['subject_bdf'], f"{args.subject}.bdf")
    
    # 比较
    if data_info and subject_info:
        print(f"\n{'='*80}")
        print("比较结果")
        print(f"{'='*80}")
        
        print(f"\n数据变化性:")
        print(f"  data.bdf唯一值数量: {data_info['unique_vals']}")
        print(f"  {args.subject}.bdf唯一值数量: {subject_info['unique_vals']}")
        print(f"  data.bdf平均变化率: {data_info['mean_diff']:.6f}")
        print(f"  {args.subject}.bdf平均变化率: {subject_info['mean_diff']:.6f}")
        
        print(f"\n上采样特征:")
        print(f"  data.bdf每10个点重复: {data_info['is_repeated']}")
        print(f"  {args.subject}.bdf每10个点重复: {subject_info['is_repeated']}")
        
        print(f"\n频谱分析:")
        print(f"  data.bdf 100Hz以上能量: {data_info['energy_above_100']:.2f}%")
        print(f"  {args.subject}.bdf 100Hz以上能量: {subject_info['energy_above_100']:.2f}%")
        
        print(f"\n建议:")
        if subject_info['unique_vals'] > data_info['unique_vals'] * 10:
            print(f"  ✅ 建议使用 {args.subject}.bdf：数据变化更连续，可能是更原始的IMU数据")
        elif data_info['is_repeated'] and not subject_info['is_repeated']:
            print(f"  ✅ 建议使用 {args.subject}.bdf：data.bdf有明显的上采样特征（重复值）")
        else:
            print(f"  ⚠️ 两个文件差异不明显，需要进一步分析")

if __name__ == "__main__":
    main()

