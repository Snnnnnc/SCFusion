#!/usr/bin/env python3
"""
检查两个数据集路径下的EEG和ECG数据是否有不同
"""
import numpy as np
import os
import sys

def compare_arrays(arr1, arr2, name="array"):
    """比较两个numpy数组是否相同"""
    if arr1.shape != arr2.shape:
        print(f"  ✗ {name}: 形状不同 - {arr1.shape} vs {arr2.shape}")
        return False
    
    if arr1.dtype != arr2.dtype:
        print(f"  ✗ {name}: 数据类型不同 - {arr1.dtype} vs {arr2.dtype}")
        return False
    
    if np.array_equal(arr1, arr2):
        print(f"  ✓ {name}: 完全相同 (形状: {arr1.shape}, 类型: {arr1.dtype})")
        return True
    else:
        # 检查差异
        diff_mask = arr1 != arr2
        num_diff = np.sum(diff_mask)
        max_diff = np.max(np.abs(arr1.astype(float) - arr2.astype(float))) if num_diff > 0 else 0
        print(f"  ✗ {name}: 有差异 - {num_diff}/{arr1.size} 个元素不同, 最大差异: {max_diff}")
        return False

def check_data_difference(path1, path2):
    """检查两个路径下的数据文件"""
    print("=" * 80)
    print(f"比较数据集:")
    print(f"  路径1: {path1}")
    print(f"  路径2: {path2}")
    print("=" * 80)
    
    # 检查路径是否存在
    if not os.path.exists(path1):
        print(f"✗ 错误: 路径1不存在: {path1}")
        return
    
    if not os.path.exists(path2):
        print(f"✗ 错误: 路径2不存在: {path2}")
        return
    
    # 要检查的文件列表
    files_to_check = [
        'eeg_patches.npy',
        'ecg_patches.npy',
        'imu_patches.npy',
        'labels.npy'
    ]
    
    results = {}
    
    for filename in files_to_check:
        file1 = os.path.join(path1, filename)
        file2 = os.path.join(path2, filename)
        
        print(f"\n检查文件: {filename}")
        print("-" * 80)
        
        # 检查文件是否存在
        exists1 = os.path.exists(file1)
        exists2 = os.path.exists(file2)
        
        if not exists1 and not exists2:
            print(f"  ⚠ 两个路径都不存在此文件")
            results[filename] = "both_missing"
            continue
        
        if not exists1:
            print(f"  ✗ 路径1不存在此文件")
            results[filename] = "path1_missing"
            continue
        
        if not exists2:
            print(f"  ✗ 路径2不存在此文件")
            results[filename] = "path2_missing"
            continue
        
        # 加载并比较文件
        try:
            print(f"  正在加载 {file1}...")
            data1 = np.load(file1, mmap_mode='r')
            print(f"    形状: {data1.shape}, 类型: {data1.dtype}, 大小: {data1.nbytes / 1024 / 1024:.2f} MB")
            
            print(f"  正在加载 {file2}...")
            data2 = np.load(file2, mmap_mode='r')
            print(f"    形状: {data2.shape}, 类型: {data2.dtype}, 大小: {data2.nbytes / 1024 / 1024:.2f} MB")
            
            # 比较数据
            is_same = compare_arrays(data1, data2, filename)
            results[filename] = "same" if is_same else "different"
            
        except Exception as e:
            print(f"  ✗ 加载或比较时出错: {e}")
            import traceback
            traceback.print_exc()
            results[filename] = "error"
    
    # 总结
    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    
    for filename, result in results.items():
        if result == "same":
            print(f"  ✓ {filename}: 相同")
        elif result == "different":
            print(f"  ✗ {filename}: 不同")
        elif result == "path1_missing":
            print(f"  ⚠ {filename}: 路径1不存在")
        elif result == "path2_missing":
            print(f"  ⚠ {filename}: 路径2不存在")
        elif result == "both_missing":
            print(f"  ⚠ {filename}: 两个路径都不存在")
        elif result == "error":
            print(f"  ✗ {filename}: 检查时出错")
    
    # 检查是否有不同
    has_difference = any(r in ["different", "path1_missing", "path2_missing", "error"] for r in results.values())
    
    if has_difference:
        print("\n⚠️  发现差异！")
    else:
        print("\n✓ 所有文件都相同或都不存在")

if __name__ == '__main__':
    # 默认路径
    base_path = './data'
    path1 = os.path.join(base_path, 'training_dataset')
    path2 = os.path.join(base_path, 'training_dataset_valid')
    
    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) >= 3:
        path1 = sys.argv[1]
        path2 = sys.argv[2]
    elif len(sys.argv) == 2:
        print("用法: python check_data_diff.py [path1] [path2]")
        print(f"使用默认路径: {path1} 和 {path2}")
    
    check_data_difference(path1, path2)








