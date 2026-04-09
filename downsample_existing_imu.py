#!/usr/bin/env python3
"""
对已存在的IMU数据进行下采样（1000Hz -> 100Hz）
由于data.bdf中的IMU数据是100Hz原始数据通过零阶保持上采样到1000Hz，
可以直接每10个点取1个来恢复到100Hz
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import List

def downsample_imu_file(imu_path: Path, modalities_json_path: Path):
    """对单个IMU文件进行下采样"""
    print(f"\n处理: {imu_path}")
    
    # 加载原始数据
    imu_data = np.load(imu_path)
    original_shape = imu_data.shape
    print(f"  原始形状: {original_shape}")
    
    # 下采样：每10个点取1个
    imu_100hz = imu_data[:, ::10]
    new_shape = imu_100hz.shape
    print(f"  下采样后形状: {new_shape}")
    print(f"  数据量减少: {original_shape[1]} -> {new_shape[1]} ({new_shape[1]/original_shape[1]*100:.1f}%)")
    
    # 备份原始文件
    backup_path = imu_path.with_suffix('.npy.1000hz_backup')
    if not backup_path.exists():
        print(f"  备份原始文件: {backup_path}")
        np.save(backup_path, imu_data)
    else:
        print(f"  备份文件已存在，跳过备份")
    
    # 保存下采样后的数据
    np.save(imu_path, imu_100hz)
    print(f"  ✅ 已保存下采样后的数据")
    
    # 更新modalities.json
    if modalities_json_path.exists():
        with open(modalities_json_path, 'r', encoding='utf-8') as f:
            modalities = json.load(f)
        
        if 'imu' in modalities:
            modalities['imu']['shape'] = [int(new_shape[0]), int(new_shape[1])]
            modalities['imu']['srate'] = 100  # 更新采样率
            
            with open(modalities_json_path, 'w', encoding='utf-8') as f:
                json.dump(modalities, f, ensure_ascii=False, indent=2)
            print(f"  ✅ 已更新modalities.json中的采样率: 1000Hz -> 100Hz")
    
    return True

def find_imu_files(processed_root: Path, subject: str = None, map_id: str = None) -> List[Path]:
    """查找所有IMU文件"""
    imu_files = []
    
    for subject_dir in processed_root.iterdir():
        if not subject_dir.is_dir():
            continue
        if subject and subject_dir.name != subject:
            continue
        
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            # 检查map_id
            if map_id:
                parts = session_dir.name.split("_")
                if len(parts) < 3 or parts[2] != map_id:
                    continue
            
            imu_path = session_dir / "_modalities" / "imu.npy"
            if imu_path.exists():
                imu_files.append(imu_path)
    
    return imu_files

def main():
    parser = argparse.ArgumentParser(description="对已存在的IMU数据进行下采样（1000Hz -> 100Hz）")
    parser.add_argument("--subject", type=str, default=None, help="被试姓名（如 zzh），如果提供则只处理该被试")
    parser.add_argument("--map", dest="map_id", type=str, default=None, help="地图编号（如 02），如果提供则只处理该地图")
    parser.add_argument("--batch-all", action="store_true", help="批量处理所有会话（忽略--subject和--map）")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要处理的文件，不实际处理")
    args = parser.parse_args()
    
    # 确定路径
    script_dir = Path(__file__).resolve().parent
    processed_root = script_dir / "processed"
    
    if not processed_root.exists():
        raise FileNotFoundError(f"未找到processed目录: {processed_root}")
    
    # 查找IMU文件
    if args.batch_all:
        imu_files = find_imu_files(processed_root)
    else:
        if args.subject is None:
            raise ValueError("需要提供 --subject 参数，或使用 --batch-all")
        imu_files = find_imu_files(processed_root, args.subject, args.map_id)
    
    if len(imu_files) == 0:
        print(f"未找到匹配的IMU文件")
        return
    
    print(f"找到 {len(imu_files)} 个IMU文件")
    
    if args.dry_run:
        print("\n【预览模式】将要处理的文件：")
        for imu_path in imu_files:
            print(f"  - {imu_path}")
        return
    
    # 处理每个文件
    success_count = 0
    for imu_path in imu_files:
        modalities_json_path = imu_path.parent / "modalities.json"
        try:
            if downsample_imu_file(imu_path, modalities_json_path):
                success_count += 1
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"✅ 处理完成！成功处理 {success_count}/{len(imu_files)} 个文件")
    print(f"{'='*80}")
    print(f"\n注意：")
    print(f"  - 原始1000Hz数据已备份为 .npy.1000hz_backup")
    print(f"  - 如需恢复，请手动重命名备份文件")

if __name__ == "__main__":
    main()

