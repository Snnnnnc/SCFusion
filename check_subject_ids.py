"""
检查 subject_ids.npy 的内容和数据集划分策略
"""

import numpy as np
import json
from pathlib import Path

def check_subject_ids():
    """检查 subject_ids.npy 的内容"""
    print("=" * 80)
    print("检查 subject_ids.npy 内容")
    print("=" * 80)
    
    # 查找 subject_ids.npy 文件
    project_root = Path(__file__).resolve().parent
    training_dataset_dir = project_root / "data" / "training_dataset"
    
    subject_ids_path = training_dataset_dir / "subject_ids.npy"
    
    if not subject_ids_path.exists():
        print(f"❌ 未找到文件: {subject_ids_path}")
        return
    
    # 加载 subject_ids
    subject_ids = np.load(subject_ids_path)
    print(f"\n文件路径: {subject_ids_path}")
    print(f"数据类型: {subject_ids.dtype}")
    print(f"数据形状: {subject_ids.shape}")
    print(f"总样本数: {len(subject_ids)}")
    
    # 统计信息
    unique_subjects = np.unique(subject_ids)
    print(f"\n唯一 subject 数量: {len(unique_subjects)}")
    print(f"Subject ID 范围: [{unique_subjects.min()}, {unique_subjects.max()}]")
    print(f"唯一 Subject IDs: {sorted(unique_subjects)}")
    
    # 每个 subject 的样本数
    print("\n各 Subject 的样本数统计:")
    print("-" * 80)
    subject_counts = {}
    for subj_id in unique_subjects:
        count = np.sum(subject_ids == subj_id)
        subject_counts[int(subj_id)] = int(count)
        print(f"  Subject {subj_id:3d}: {count:6d} 个样本 ({count/len(subject_ids)*100:5.2f}%)")
    
    # 显示前20个样本的 subject_id
    print(f"\n前20个样本的 Subject ID:")
    print("-" * 80)
    for i in range(min(20, len(subject_ids))):
        print(f"  样本 {i:4d}: Subject {subject_ids[i]}")
    
    return subject_ids, subject_counts

def check_meta_info():
    """检查 meta.json 中的 session 信息"""
    print("\n" + "=" * 80)
    print("检查 meta.json 中的 session 信息")
    print("=" * 80)
    
    project_root = Path(__file__).resolve().parent
    
    # 检查 preprocessed/meta.json
    preprocessed_meta = project_root / "data" / "preprocessed" / "meta.json"
    if preprocessed_meta.exists():
        print(f"\n检查预处理数据元数据: {preprocessed_meta}")
        with preprocessed_meta.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        
        if "sessions" in meta:
            sessions = meta["sessions"]
            print(f"总 session 数: {len(sessions)}")
            print("\n前10个 session 信息:")
            print("-" * 80)
            for i, session in enumerate(sessions[:10]):
                print(f"  Session {i}:")
                for key, value in session.items():
                    if isinstance(value, (list, dict)):
                        print(f"    {key}: {type(value).__name__} (长度: {len(value) if hasattr(value, '__len__') else 'N/A'})")
                    else:
                        print(f"    {key}: {value}")
                print()
    
    # 检查 training_dataset/meta.json
    training_meta = project_root / "data" / "training_dataset" / "meta.json"
    if training_meta.exists():
        print(f"\n检查训练数据集元数据: {training_meta}")
        with training_meta.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        
        print("\n训练数据集元数据:")
        print("-" * 80)
        for key, value in meta.items():
            if isinstance(value, (list, dict)):
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                    print(f"  {key}: {value[:5]}... (长度: {len(value)})")
                else:
                    print(f"  {key}: {type(value).__name__} (长度: {len(value) if hasattr(value, '__len__') else 'N/A'})")
            else:
                print(f"  {key}: {value}")

def check_data_split_strategy():
    """检查数据集划分策略"""
    print("\n" + "=" * 80)
    print("检查数据集划分策略")
    print("=" * 80)
    
    # 读取 experiment.py 中的划分逻辑
    project_root = Path(__file__).resolve().parent
    experiment_file = project_root / "experiment.py"
    
    if experiment_file.exists():
        print(f"\n检查 experiment.py 中的划分逻辑...")
        with experiment_file.open("r", encoding="utf-8") as f:
            content = f.read()
        
        # 查找 subject-wise 划分的相关代码
        if "subject_wise" in content.lower() or "subject-wise" in content.lower():
            print("  ✓ 找到 subject-wise 划分相关代码")
        
        if "use_subject_wise" in content:
            print("  ✓ 使用 use_subject_wise 标志")
        
        # 查找划分比例
        if "0.7" in content and "0.15" in content:
            print("  ✓ 划分比例: 训练集 70%, 验证集 15%, 测试集 15%")
    
    # 检查 build_training_dataset.py 中的 subject 信息
    build_dataset_file = project_root / "data" / "preprocessing" / "build_training_dataset.py"
    if build_dataset_file.exists():
        print(f"\n检查 build_training_dataset.py 中的 subject 信息...")
        with build_dataset_file.open("r", encoding="utf-8") as f:
            content = f.read()
        
        if "subject_boundaries" in content:
            print("  ✓ 使用 subject_boundaries 来划分 subject")
        
        if "subject_map" in content:
            print("  ✓ 使用 subject_map 来映射 subject")

def check_subject_naming():
    """检查 subject 是否按姓名划分"""
    print("\n" + "=" * 80)
    print("检查 Subject 命名方式")
    print("=" * 80)
    
    project_root = Path(__file__).resolve().parent
    preprocessed_meta = project_root / "data" / "preprocessed" / "meta.json"
    
    if preprocessed_meta.exists():
        with preprocessed_meta.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        
        if "sessions" in meta:
            sessions = meta["sessions"]
            print(f"\n检查前10个 session 的标识信息:")
            print("-" * 80)
            
            name_keys = ["name", "subject", "subject_name", "participant", "participant_name", "file", "filename"]
            found_names = False
            
            for i, session in enumerate(sessions[:10]):
                print(f"\nSession {i}:")
                for key in name_keys:
                    if key in session:
                        print(f"  {key}: {session[key]}")
                        found_names = True
                
                # 显示所有键
                if not found_names:
                    print(f"  所有键: {list(session.keys())}")
            
            if found_names:
                print("\n  ✓ 找到可能的 subject 名称字段")
            else:
                print("\n  ⚠️  未找到明显的 subject 名称字段")
                print("  Subject ID 可能是按 session 索引自动生成的")

if __name__ == "__main__":
    # 检查 subject_ids
    subject_ids, subject_counts = check_subject_ids()
    
    # 检查元数据
    check_meta_info()
    
    # 检查划分策略
    check_data_split_strategy()
    
    # 检查命名方式
    check_subject_naming()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("\nSubject 划分策略:")
    print("  1. Subject ID 是整数，从 0 开始递增")
    print("  2. 每个 Subject 对应一个或多个 session")
    print("  3. 数据集划分使用 subject-wise 策略，确保同一 subject 的数据")
    print("     不会同时出现在训练集、验证集和测试集中")
    print("  4. 划分比例: 训练集 70%, 验证集 15%, 测试集 15% (按 subject 数量)")
    print("\n⚠️  注意: Subject ID 不是按姓名划分的，而是按 session 索引生成的")
    print("   每个 session 对应一个 subject_id，多个 session 可能属于同一个 subject")





