"""
调试脚本：检查IMU数据中的NaN问题
"""

import numpy as np
from pathlib import Path
import sys

def check_preprocessed_data(data_dir: Path):
    """检查预处理后的数据"""
    print("=" * 60)
    print("检查预处理后的数据 (data/preprocessed)")
    print("=" * 60)
    
    imu_path = data_dir / "imu_concatenated.npy"
    if not imu_path.exists():
        print(f"❌ 未找到: {imu_path}")
        return
    
    print(f"加载: {imu_path}")
    imu = np.load(imu_path)
    print(f"数据形状: {imu.shape}")
    
    # 检查NaN和Inf
    nan_count = np.isnan(imu).sum()
    inf_count = np.isinf(imu).sum()
    
    print(f"\nNaN检查:")
    print(f"  NaN数量: {nan_count}")
    print(f"  Inf数量: {inf_count}")
    
    if nan_count > 0:
        print(f"\n⚠️  发现NaN值！")
        # 找出哪些通道有NaN
        nan_by_channel = np.isnan(imu).sum(axis=1)
        print(f"  各通道NaN数量:")
        for ch in range(len(nan_by_channel)):
            if nan_by_channel[ch] > 0:
                print(f"    通道 {ch}: {nan_by_channel[ch]} 个NaN")
        
        # 找出哪些时间点有NaN
        nan_by_time = np.isnan(imu).sum(axis=0)
        nan_time_indices = np.where(nan_by_time > 0)[0]
        if len(nan_time_indices) > 0:
            print(f"\n  有NaN的时间点数量: {len(nan_time_indices)}")
            print(f"  前10个有NaN的时间点索引: {nan_time_indices[:10]}")
            print(f"  后10个有NaN的时间点索引: {nan_time_indices[-10:]}")
    
    if inf_count > 0:
        print(f"\n⚠️  发现Inf值！")
        inf_by_channel = np.isinf(imu).sum(axis=1)
        print(f"  各通道Inf数量:")
        for ch in range(len(inf_by_channel)):
            if inf_by_channel[ch] > 0:
                print(f"    通道 {ch}: {inf_by_channel[ch]} 个Inf")
    
    # 统计信息
    print(f"\n数据统计:")
    print(f"  最小值: {np.nanmin(imu):.6f}")
    print(f"  最大值: {np.nanmax(imu):.6f}")
    print(f"  均值: {np.nanmean(imu):.6f}")
    print(f"  标准差: {np.nanstd(imu):.6f}")
    
    # 检查每个通道
    print(f"\n各通道统计:")
    for ch in range(imu.shape[0]):
        ch_data = imu[ch, :]
        ch_nan = np.isnan(ch_data).sum()
        ch_inf = np.isinf(ch_data).sum()
        if ch_nan > 0 or ch_inf > 0:
            print(f"  通道 {ch}: NaN={ch_nan}, Inf={ch_inf}, "
                  f"范围=[{np.nanmin(ch_data):.6f}, {np.nanmax(ch_data):.6f}]")
    
    return imu


def check_training_data(data_dir: Path):
    """检查训练数据"""
    print("\n" + "=" * 60)
    print("检查训练数据 (data/training_dataset)")
    print("=" * 60)
    
    imu_patches_path = data_dir / "imu_patches.npy"
    if not imu_patches_path.exists():
        print(f"❌ 未找到: {imu_patches_path}")
        return
    
    print(f"加载: {imu_patches_path}")
    imu_patches = np.load(imu_patches_path)
    print(f"数据形状: {imu_patches.shape}")
    
    # 检查NaN和Inf
    nan_count = np.isnan(imu_patches).sum()
    inf_count = np.isinf(imu_patches).sum()
    
    print(f"\nNaN检查:")
    print(f"  NaN数量: {nan_count}")
    print(f"  Inf数量: {inf_count}")
    
    if nan_count > 0:
        print(f"\n⚠️  发现NaN值！")
        # 找出哪些窗口有NaN
        nan_by_window = np.isnan(imu_patches).any(axis=(1, 2, 3))
        nan_window_indices = np.where(nan_by_window)[0]
        print(f"  有NaN的窗口数量: {len(nan_window_indices)}")
        print(f"  前10个有NaN的窗口索引: {nan_window_indices[:10]}")
        
        # 检查第一个有NaN的窗口
        if len(nan_window_indices) > 0:
            first_nan_idx = nan_window_indices[0]
            first_nan_window = imu_patches[first_nan_idx]
            print(f"\n  窗口 {first_nan_idx} 的详细信息:")
            print(f"    形状: {first_nan_window.shape}")
            print(f"    NaN数量: {np.isnan(first_nan_window).sum()}")
            print(f"    各patch的NaN数量:")
            for p in range(first_nan_window.shape[0]):
                patch_nan = np.isnan(first_nan_window[p]).sum()
                if patch_nan > 0:
                    print(f"      Patch {p}: {patch_nan} 个NaN")
    
    if inf_count > 0:
        print(f"\n⚠️  发现Inf值！")
        inf_by_window = np.isinf(imu_patches).any(axis=(1, 2, 3))
        inf_window_indices = np.where(inf_by_window)[0]
        print(f"  有Inf的窗口数量: {len(inf_window_indices)}")
    
    return imu_patches


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    
    # 检查预处理数据
    preprocessed_dir = project_root / "data" / "preprocessed"
    if preprocessed_dir.exists():
        imu_preprocessed = check_preprocessed_data(preprocessed_dir)
    else:
        print(f"❌ 预处理数据目录不存在: {preprocessed_dir}")
    
    # 检查训练数据
    training_dir = project_root / "data" / "training_dataset"
    if training_dir.exists():
        imu_training = check_training_data(training_dir)
    else:
        print(f"❌ 训练数据目录不存在: {training_dir}")
    
    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

