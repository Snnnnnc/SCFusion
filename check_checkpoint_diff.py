"""
检查 checkpoint.pkl 和 checkpoint_best.pkl 的区别
"""

import pickle
import os
from pathlib import Path

checkpoint_dir = Path('./results/MotionSickness_ComfortClassificationModel_PhysioFusionNet_v1_fold1_seed42')

checkpoint_file = checkpoint_dir / 'checkpoint.pkl'
best_file = checkpoint_dir / 'checkpoint_best.pkl'

print("="*60)
print("Checkpoint 文件对比")
print("="*60)

# 检查 checkpoint.pkl
if checkpoint_file.exists():
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        print(f"\n📄 checkpoint.pkl (最新checkpoint):")
        print(f"  - Epoch: {checkpoint_data['epoch']}")
        print(f"  - Is Best: {checkpoint_data.get('is_best', False)}")
        print(f"  - Timestamp: {checkpoint_data.get('timestamp', 'N/A')}")
        best_info = checkpoint_data['trainer_state']['best_epoch_info']
        print(f"  - 最佳模型信息:")
        print(f"    * 最佳Epoch: {best_info['epoch']}")
        print(f"    * 最佳准确率: {best_info['accuracy']:.4f}")
        print(f"    * 最佳损失: {best_info['loss']:.4f}")
    except Exception as e:
        print(f"\n⚠️  无法读取 checkpoint.pkl: {e}")
        print(f"   文件大小: {checkpoint_file.stat().st_size / 1024 / 1024:.2f} MB")
else:
    print(f"\n❌ checkpoint.pkl 不存在")

# 检查 checkpoint_best.pkl
if best_file.exists():
    try:
        with open(best_file, 'rb') as f:
            best_data = pickle.load(f)
        
        print(f"\n📄 checkpoint_best.pkl (最佳模型checkpoint):")
        print(f"  - Epoch: {best_data['epoch']}")
        print(f"  - Is Best: {best_data.get('is_best', False)}")
        print(f"  - Timestamp: {best_data.get('timestamp', 'N/A')}")
        best_info = best_data['trainer_state']['best_epoch_info']
        print(f"  - 最佳模型信息:")
        print(f"    * 最佳Epoch: {best_info['epoch']}")
        print(f"    * 最佳准确率: {best_info['accuracy']:.4f}")
        print(f"    * 最佳损失: {best_info['loss']:.4f}")
    except Exception as e:
        print(f"\n⚠️  无法读取 checkpoint_best.pkl: {e}")
        print(f"   文件大小: {best_file.stat().st_size / 1024 / 1024:.2f} MB")
else:
    print(f"\n❌ checkpoint_best.pkl 不存在")

# 对比
if checkpoint_file.exists() and best_file.exists():
    print(f"\n" + "="*60)
    print("对比分析:")
    print("="*60)
    
    checkpoint_epoch = checkpoint_data['epoch']
    best_epoch = best_data['epoch']
    checkpoint_acc = checkpoint_data['trainer_state']['best_epoch_info']['accuracy']
    best_acc = best_data['trainer_state']['best_epoch_info']['accuracy']
    
    print(f"\n1. Epoch 差异:")
    print(f"   checkpoint.pkl: epoch {checkpoint_epoch}")
    print(f"   checkpoint_best.pkl: epoch {best_epoch}")
    
    if checkpoint_epoch == best_epoch:
        print(f"   ✓ 两个文件保存的是同一个epoch的模型")
    else:
        print(f"   ⚠️  两个文件保存的是不同epoch的模型")
        print(f"   - checkpoint.pkl 是最后训练的epoch")
        print(f"   - checkpoint_best.pkl 是验证集上表现最好的epoch")
    
    print(f"\n2. 模型状态:")
    print(f"   checkpoint.pkl 保存的是: epoch {checkpoint_epoch} 的模型状态")
    print(f"   checkpoint_best.pkl 保存的是: epoch {best_epoch} 的模型状态（最佳模型）")
    
    print(f"\n3. 使用建议:")
    if checkpoint_epoch == best_epoch:
        print(f"   ✓ 两个文件相同，可以任选一个继续训练")
    else:
        print(f"   - 如果想从最新状态继续: 使用 checkpoint.pkl")
        print(f"   - 如果想从最佳模型继续: 使用 checkpoint_best.pkl")
        print(f"   - 推荐: 使用 checkpoint_best.pkl（性能最好的模型）")

print(f"\n" + "="*60)

