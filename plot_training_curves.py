"""
绘制训练曲线：根据 checkpoint_log.csv 文件可视化训练过程
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_curves(csv_path, output_path=None):
    """
    绘制训练曲线
    
    Args:
        csv_path: checkpoint_log.csv 文件路径
        output_path: 输出图片路径（如果为None，则保存到CSV文件同目录）
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 移除空行（如果有）
    df = df.dropna(subset=['epoch'])
    
    # 确保epoch是数值类型
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df = df.dropna(subset=['epoch'])
    
    # 排序以确保epoch顺序正确
    df = df.sort_values('epoch').reset_index(drop=True)
    
    # 如果output_path未指定，使用CSV文件同目录
    if output_path is None:
        csv_file = Path(csv_path)
        output_path = csv_file.parent / 'training_curves.png'
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress Visualization', fontsize=16, fontweight='bold')
    
    # 1. Train Loss
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(bottom=0)
    
    # 2. Train Accuracy
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['train_acc'], 'g-', linewidth=2, label='Train Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1])
    
    # 3. Train F1 Score
    ax3 = axes[0, 2]
    # 处理可能的空值
    train_f1 = pd.to_numeric(df['train_f1'], errors='coerce')
    ax3.plot(df['epoch'], train_f1, 'c-', linewidth=2, label='Train F1')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.set_title('Training F1 Score', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 1])
    
    # 4. Validation Accuracy
    ax4 = axes[1, 0]
    ax4.plot(df['epoch'], df['val_acc'], 'r-', linewidth=2, label='Val Accuracy')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Validation Accuracy', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0, 1])
    
    # 5. Validation F1 Score
    ax5 = axes[1, 1]
    # 处理可能的空值
    val_f1 = pd.to_numeric(df['val_f1'], errors='coerce')
    ax5.plot(df['epoch'], val_f1, 'm-', linewidth=2, label='Val F1')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('F1 Score', fontsize=12)
    ax5.set_title('Validation F1 Score', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_ylim([0, 1])
    
    # 6. Combined: Train vs Val Accuracy
    ax6 = axes[1, 2]
    ax6.plot(df['epoch'], df['train_acc'], 'g-', linewidth=2, label='Train Accuracy', alpha=0.7)
    ax6.plot(df['epoch'], df['val_acc'], 'r-', linewidth=2, label='Val Accuracy', alpha=0.7)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Accuracy', fontsize=12)
    ax6.set_title('Train vs Validation Accuracy', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存到: {output_path}")
    
    # 显示统计信息
    print("\n" + "="*60)
    print("训练统计信息")
    print("="*60)
    print(f"总训练轮次: {len(df)}")
    print(f"最佳训练准确率: {df['train_acc'].max():.4f} (Epoch {df.loc[df['train_acc'].idxmax(), 'epoch']})")
    print(f"最佳验证准确率: {df['val_acc'].max():.4f} (Epoch {df.loc[df['val_acc'].idxmax(), 'epoch']})")
    if not train_f1.isna().all():
        print(f"最佳训练F1: {train_f1.max():.4f} (Epoch {df.loc[train_f1.idxmax(), 'epoch']})")
    if not val_f1.isna().all():
        print(f"最佳验证F1: {val_f1.max():.4f} (Epoch {df.loc[val_f1.idxmax(), 'epoch']})")
    print(f"最终训练损失: {df['train_loss'].iloc[-1]:.4f}")
    print(f"最终验证损失: {df['val_loss'].iloc[-1]:.4f}")
    print("="*60)
    
    plt.close()
    
    return fig

if __name__ == '__main__':
    import sys
    
    # 默认路径
    default_csv = './results/MotionSickness_ComfortClassificationModel_PhysioFusionNet_v1_fold1_seed42/checkpoint_log.csv'
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_csv
    
    # 检查文件是否存在
    if not Path(csv_path).exists():
        print(f"错误: 文件不存在: {csv_path}")
        print(f"请提供正确的CSV文件路径")
        sys.exit(1)
    
    # 绘制曲线
    plot_training_curves(csv_path)

