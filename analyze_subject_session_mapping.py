"""
分析真实被试（subject name）和 session 的映射关系
检查是否存在信息泄漏风险
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict

def analyze_subject_session_mapping():
    """分析真实被试和session的映射关系"""
    print("=" * 80)
    print("分析真实被试（Subject Name）和 Session 的映射关系")
    print("=" * 80)
    
    project_root = Path(__file__).resolve().parent
    
    # 加载预处理数据的meta.json
    preprocessed_meta = project_root / "data" / "preprocessed" / "meta.json"
    if not preprocessed_meta.exists():
        print(f"❌ 未找到文件: {preprocessed_meta}")
        return
    
    with preprocessed_meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    
    if "sessions" not in meta:
        print("❌ meta.json 中没有 sessions 信息")
        return
    
    sessions = meta["sessions"]
    print(f"\n总 session 数: {len(sessions)}")
    
    # 统计每个真实被试（subject name）对应的session
    subject_to_sessions = defaultdict(list)
    session_to_subject_name = {}
    
    for idx, session in enumerate(sessions):
        subject_name = session.get("subject", "unknown")
        session_id = session.get("session", f"session_{idx}")
        subject_to_sessions[subject_name].append({
            "session_index": idx,
            "session_id": session_id
        })
        session_to_subject_name[idx] = subject_name
    
    print(f"\n唯一真实被试数: {len(subject_to_sessions)}")
    print(f"真实被试列表: {sorted(subject_to_sessions.keys())}")
    
    # 统计每个真实被试的session数
    print("\n各真实被试的 Session 数量统计:")
    print("-" * 80)
    for subject_name in sorted(subject_to_sessions.keys()):
        session_count = len(subject_to_sessions[subject_name])
        session_indices = [s["session_index"] for s in subject_to_sessions[subject_name]]
        print(f"  {subject_name:15s}: {session_count:3d} 个 session, 索引: {session_indices[:10]}{'...' if len(session_indices) > 10 else ''}")
    
    # 检查是否有多个session属于同一个真实被试
    multi_session_subjects = {k: v for k, v in subject_to_sessions.items() if len(v) > 1}
    print(f"\n⚠️  有 {len(multi_session_subjects)} 个真实被试拥有多个 session:")
    print("-" * 80)
    for subject_name, sessions_list in sorted(multi_session_subjects.items()):
        print(f"  {subject_name}: {len(sessions_list)} 个 session")
        for s in sessions_list[:5]:  # 只显示前5个
            print(f"    - Session {s['session_index']}: {s['session_id']}")
        if len(sessions_list) > 5:
            print(f"    ... 还有 {len(sessions_list) - 5} 个 session")
    
    return subject_to_sessions, session_to_subject_name

def check_current_subject_id_mapping():
    """检查当前的subject_id映射（基于session索引）"""
    print("\n" + "=" * 80)
    print("检查当前的 Subject ID 映射（基于 Session 索引）")
    print("=" * 80)
    
    project_root = Path(__file__).resolve().parent
    
    # 加载预处理数据的meta.json
    preprocessed_meta = project_root / "data" / "preprocessed" / "meta.json"
    with preprocessed_meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    
    sessions = meta["sessions"]
    
    # 加载subject_ids.npy
    training_dataset_dir = project_root / "data" / "training_dataset"
    subject_ids_path = training_dataset_dir / "subject_ids.npy"
    
    if not subject_ids_path.exists():
        print(f"❌ 未找到文件: {subject_ids_path}")
        return
    
    subject_ids = np.load(subject_ids_path)
    
    # 创建映射：session_index -> subject_name
    session_to_subject_name = {}
    for idx, session in enumerate(sessions):
        subject_name = session.get("subject", "unknown")
        session_to_subject_name[idx] = subject_name
    
    # 统计：每个subject_id（session_index）对应的真实被试
    print("\n当前 Subject ID（Session 索引）到真实被试的映射:")
    print("-" * 80)
    unique_subject_ids = np.unique(subject_ids)
    for subj_id in sorted(unique_subject_ids)[:20]:  # 只显示前20个
        real_subject = session_to_subject_name.get(int(subj_id), "unknown")
        print(f"  Subject ID {subj_id:3d} -> 真实被试: {real_subject}")
    
    # 检查问题：同一个真实被试的不同session是否被分配了不同的subject_id
    print("\n⚠️  问题分析:")
    print("-" * 80)
    print("  当前实现中，每个 session 被分配一个独立的 subject_id（0-81）")
    print("  这意味着同一个真实被试的不同 session 被当作不同的 subject")
    print("  这可能导致以下问题：")
    print("    1. Subject-wise 划分时，同一真实被试的不同 session 可能被分到不同数据集")
    print("    2. Per-subject 归一化时，同一真实被试的不同 session 使用不同的统计量")
    print("    3. 这违反了 subject-wise 划分的初衷，可能导致信息泄漏")
    
    return session_to_subject_name

def check_information_leakage_risk():
    """检查信息泄漏风险"""
    print("\n" + "=" * 80)
    print("检查信息泄漏风险")
    print("=" * 80)
    
    project_root = Path(__file__).resolve().parent
    
    # 加载预处理数据的meta.json
    preprocessed_meta = project_root / "data" / "preprocessed" / "meta.json"
    with preprocessed_meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    
    sessions = meta["sessions"]
    
    # 创建真实被试到session索引的映射
    subject_to_session_indices = defaultdict(list)
    for idx, session in enumerate(sessions):
        subject_name = session.get("subject", "unknown")
        subject_to_session_indices[subject_name].append(idx)
    
    # 模拟数据集划分（按当前的subject_id，即session索引）
    print("\n模拟数据集划分（按当前 Subject ID，即 Session 索引）:")
    print("-" * 80)
    
    # 假设按70/15/15划分82个subject（session）
    n_subjects = 82
    train_size = int(0.7 * n_subjects)
    val_size = int(0.15 * n_subjects)
    
    # 模拟划分
    all_session_indices = list(range(n_subjects))
    train_sessions = set(all_session_indices[:train_size])
    val_sessions = set(all_session_indices[train_size:train_size + val_size])
    test_sessions = set(all_session_indices[train_size + val_size:])
    
    print(f"  训练集: {len(train_sessions)} 个 session (subject_id 0-{train_size-1})")
    print(f"  验证集: {len(val_sessions)} 个 session (subject_id {train_size}-{train_size+val_size-1})")
    print(f"  测试集: {len(test_sessions)} 个 session (subject_id {train_size+val_size}-{n_subjects-1})")
    
    # 检查是否有真实被试的多个session被分到不同数据集
    leakage_risks = []
    for subject_name, session_indices in subject_to_session_indices.items():
        if len(session_indices) > 1:
            train_count = len([s for s in session_indices if s in train_sessions])
            val_count = len([s for s in session_indices if s in val_sessions])
            test_count = len([s for s in session_indices if s in test_sessions])
            
            # 如果同一个真实被试的session出现在多个数据集，存在泄漏风险
            datasets_present = sum([train_count > 0, val_count > 0, test_count > 0])
            if datasets_present > 1:
                leakage_risks.append({
                    "subject": subject_name,
                    "sessions": session_indices,
                    "train": train_count,
                    "val": val_count,
                    "test": test_count
                })
    
    if leakage_risks:
        print(f"\n❌ 发现 {len(leakage_risks)} 个真实被试存在信息泄漏风险:")
        print("-" * 80)
        for risk in leakage_risks[:10]:  # 只显示前10个
            print(f"  真实被试: {risk['subject']}")
            print(f"    Session 索引: {risk['sessions']}")
            print(f"    训练集: {risk['train']} 个, 验证集: {risk['val']} 个, 测试集: {risk['test']} 个")
            print()
        if len(leakage_risks) > 10:
            print(f"  ... 还有 {len(leakage_risks) - 10} 个真实被试存在泄漏风险")
    else:
        print("\n✓ 在当前划分下，没有发现明显的泄漏风险")
    
    return leakage_risks

def suggest_solution():
    """建议解决方案"""
    print("\n" + "=" * 80)
    print("建议的解决方案")
    print("=" * 80)
    
    print("\n问题:")
    print("  当前实现中，每个 session 被分配一个独立的 subject_id，")
    print("  但多个 session 可能属于同一个真实被试。")
    
    print("\n解决方案:")
    print("  1. 修改 build_training_dataset.py，使用真实被试名称（subject name）")
    print("     而不是 session 索引来生成 subject_ids")
    print("  2. 将相同真实被试的所有 session 映射到同一个 subject_id")
    print("  3. 这样在 subject-wise 划分时，同一真实被试的所有 session")
    print("     会被分到同一个数据集，避免信息泄漏")
    
    print("\n具体修改:")
    print("  - 在 build_training_dataset.py 中，根据 meta.json 中的 'subject' 字段")
    print("    创建 subject_name -> subject_id 的映射")
    print("  - 相同 subject_name 的所有 session 使用相同的 subject_id")
    print("  - 这样 subject_ids.npy 中的值会反映真实被试，而不是 session 索引")

if __name__ == "__main__":
    # 分析映射关系
    subject_to_sessions, session_to_subject_name = analyze_subject_session_mapping()
    
    # 检查当前的subject_id映射
    check_current_subject_id_mapping()
    
    # 检查信息泄漏风险
    leakage_risks = check_information_leakage_risk()
    
    # 建议解决方案
    suggest_solution()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("\n关键发现:")
    print("  1. 存在多个 session 属于同一个真实被试的情况")
    print("  2. 当前实现中，每个 session 被当作独立的 subject_id")
    print("  3. 这可能导致同一真实被试的不同 session 被分到不同数据集")
    print("  4. 建议修改代码，使用真实被试名称来生成 subject_id")


