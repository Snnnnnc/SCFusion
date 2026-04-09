#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型测试可视化脚本
加载训练好的模型，将输入的时间序列处理成模型输入的维度和格式，
每10秒输出一次舒适度的预测值，并可视化结果
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from datetime import datetime

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入项目模块
from models.comfort_model import (
    ComfortClassificationModel, IMUClassificationModel, MixClassificationModel,
    SimpleMixClassificationModel, NewMixClassificationModel, AllMixClassificationModel
)


def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    print(f"加载checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def infer_mode_from_folder(folder_path):
    """从文件夹路径推断mode"""
    folder_name = os.path.basename(folder_path)
    # 文件夹名格式: MotionSickness_ModelName_Stamp_mode_fold1_seed42
    parts = folder_name.split('_')
    
    # 检查常见的mode值
    mode_keywords = ['allmix', 'newmix', 'simplemix', 'mix', 'imu', 'rawimu', 'physio', 'eeg', 'ecg']
    for keyword in mode_keywords:
        if keyword in parts:
            return keyword
    
    # 如果没找到，尝试从ModelName推断
    if 'AllMix' in folder_name:
        return 'allmix'
    elif 'NewMix' in folder_name:
        return 'newmix'
    elif 'SimpleMix' in folder_name:
        return 'simplemix'
    elif 'Mix' in folder_name and 'AllMix' not in folder_name and 'NewMix' not in folder_name and 'SimpleMix' not in folder_name:
        return 'mix'
    elif 'IMU' in folder_name:
        return 'imu'
    elif 'Physio' in folder_name or 'Comfort' in folder_name:
        return 'physio'
    
    return 'allmix'  # 默认值


def init_model_from_checkpoint(checkpoint, checkpoint_path, device, mode=None):
    """从checkpoint初始化模型"""
    # 从checkpoint获取模型参数
    trainer_state = checkpoint.get('trainer_state', {})
    model_state_dict = trainer_state.get('model_state_dict', None)
    
    if model_state_dict is None:
        raise ValueError("checkpoint中未找到模型状态")
    
    # 从checkpoint获取参数（checkpoint可能没有保存args）
    checkpoint_args = checkpoint.get('args', {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}
    
    # 如果mode未指定，尝试从文件夹名推断
    if mode is None:
        mode = checkpoint_args.get('mode', None)
        if mode is None:
            mode = infer_mode_from_folder(os.path.dirname(checkpoint_path))
            print(f"从文件夹名推断mode: {mode}")
    
    # 默认参数值
    args_dict = {
        'num_classes': checkpoint_args.get('num_classes', 5),
        'imu_channels': checkpoint_args.get('imu_channels', 18),
        'eeg_channels': checkpoint_args.get('eeg_channels', 59),
        'ecg_channels': checkpoint_args.get('ecg_channels', 1),
        'patch_length': checkpoint_args.get('patch_length', 250),
        'num_patches': checkpoint_args.get('num_patches', 10),
        'encoding_dim': checkpoint_args.get('encoding_dim', 256),
        'num_heads': checkpoint_args.get('num_heads', 8),
        'attention_output_mode': checkpoint_args.get('attention_output_mode', 'global'),
        'hidden_dims': checkpoint_args.get('hidden_dims', [512, 256, 128]),
        'dropout': checkpoint_args.get('dropout', 0.1),
    }
    
    # 解析hidden_dims（可能是字符串）
    if isinstance(args_dict['hidden_dims'], str):
        args_dict['hidden_dims'] = [int(x.strip()) for x in args_dict['hidden_dims'].split(',')]
    
    # 根据mode初始化模型
    num_classes = args_dict['num_classes']
    
    if mode == 'allmix':
        model = AllMixClassificationModel(
            imu_channels=args_dict['imu_channels'],
            eeg_channels=args_dict['eeg_channels'],
            ecg_channels=args_dict['ecg_channels'],
            patch_length=args_dict['patch_length'],
            num_patches=args_dict['num_patches'],
            encoding_dim=args_dict['encoding_dim'],
            num_heads=args_dict['num_heads'],
            num_classes=num_classes,
            attention_output_mode=args_dict['attention_output_mode'],
            hidden_dims=args_dict['hidden_dims'],
            dropout=args_dict['dropout'],
        )
    elif mode == 'newmix':
        model = NewMixClassificationModel(
            imu_channels=args_dict['imu_channels'],
            eeg_channels=args_dict['eeg_channels'],
            ecg_channels=args_dict['ecg_channels'],
            patch_length=args_dict['patch_length'],
            num_patches=args_dict['num_patches'],
            encoding_dim=args_dict['encoding_dim'],
            num_heads=args_dict['num_heads'],
            num_classes=num_classes,
            attention_output_mode=args_dict['attention_output_mode'],
            hidden_dims=args_dict['hidden_dims'],
            dropout=args_dict['dropout'],
        )
    elif mode == 'simplemix':
        model = SimpleMixClassificationModel(
            imu_channels=args_dict['imu_channels'],
            eeg_channels=args_dict['eeg_channels'],
            ecg_channels=args_dict['ecg_channels'],
            patch_length=args_dict['patch_length'],
            num_patches=args_dict['num_patches'],
            encoding_dim=args_dict['encoding_dim'],
            num_heads=args_dict['num_heads'],
            num_classes=num_classes,
            attention_output_mode=args_dict['attention_output_mode'],
            hidden_dims=args_dict['hidden_dims'],
            dropout=args_dict['dropout'],
        )
    elif mode == 'mix':
        model = MixClassificationModel(
            imu_channels=args_dict['imu_channels'],
            eeg_channels=args_dict['eeg_channels'],
            ecg_channels=args_dict['ecg_channels'],
            patch_length=args_dict['patch_length'],
            num_patches=args_dict['num_patches'],
            encoding_dim=args_dict['encoding_dim'],
            num_heads=args_dict['num_heads'],
            num_classes=num_classes,
            attention_output_mode=args_dict['attention_output_mode'],
            hidden_dims=args_dict['hidden_dims'],
            dropout=args_dict['dropout'],
        )
    elif mode == 'imu':
        model = IMUClassificationModel(
            imu_channels=args_dict['imu_channels'],
            patch_length=args_dict['patch_length'],
            num_patches=args_dict['num_patches'],
            encoding_dim=args_dict['encoding_dim'],
            num_heads=args_dict['num_heads'],
            num_classes=num_classes,
            attention_output_mode=args_dict['attention_output_mode'],
            hidden_dims=args_dict['hidden_dims'],
            dropout=args_dict['dropout'],
        )
    elif mode == 'rawimu':
        model = IMUClassificationModel(
            imu_channels=6,  # Raw IMU只使用6维
            patch_length=args_dict['patch_length'],
            num_patches=args_dict['num_patches'],
            encoding_dim=args_dict['encoding_dim'],
            num_heads=args_dict['num_heads'],
            num_classes=num_classes,
            attention_output_mode=args_dict['attention_output_mode'],
            hidden_dims=args_dict['hidden_dims'],
            dropout=args_dict['dropout'],
        )
    else:  # physio
        model = ComfortClassificationModel(
            eeg_channels=args_dict['eeg_channels'],
            ecg_channels=args_dict['ecg_channels'],
            patch_length=args_dict['patch_length'],
            num_patches=args_dict['num_patches'],
            encoding_dim=args_dict['encoding_dim'],
            num_heads=args_dict['num_heads'],
            num_classes=num_classes,
            attention_output_mode=args_dict['attention_output_mode'],
            hidden_dims=args_dict['hidden_dims'],
            dropout=args_dict['dropout'],
        )
    
    # 加载模型权重
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    print(f"✓ 模型已加载 (mode={mode}, num_classes={num_classes})")
    
    return model, mode, args_dict


def create_patches_from_window(window_data, patch_length):
    """
    将窗口数据切分成patches
    输入: window_data (channels, window_length)
    输出: patches (num_patches, channels, patch_length)
    """
    channels, window_length = window_data.shape
    num_patches = window_length // patch_length
    
    if num_patches == 0:
        raise ValueError(f"窗口长度 ({window_length}) 必须大于等于patch长度 ({patch_length})")
    
    # 切分patches: (channels, window_length) -> (channels, num_patches, patch_length) -> (num_patches, channels, patch_length)
    patches = window_data[:, :num_patches * patch_length].reshape(channels, num_patches, patch_length).transpose(1, 0, 2)
    
    return patches


def normalize_data(data, normalization_stats, modality):
    """
    归一化数据
    输入: data (channels, time_points) 或 (num_patches, channels, patch_length)
    输出: normalized_data
    """
    if normalization_stats is None or modality not in normalization_stats:
        return data
    
    stats = normalization_stats[modality]
    mean = stats.get('mean', None)
    std = stats.get('std', None)
    
    if mean is None or std is None:
        return data
    
    # 扩展维度以便广播: (1, channels, 1)
    if len(data.shape) == 3:  # (num_patches, channels, patch_length)
        mean_expanded = mean.reshape(1, -1, 1)
        std_expanded = std.reshape(1, -1, 1)
    elif len(data.shape) == 2:  # (channels, time_points)
        mean_expanded = mean.reshape(-1, 1)
        std_expanded = std.reshape(-1, 1)
    else:
        return data
    
    std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
    normalized = (data - mean_expanded) / (std_expanded + 1e-8)
    
    return normalized


def process_time_series_to_patches(time_series_dict, mode, window_length, hop_length, patch_length, 
                                   normalization_stats=None, sampling_rate=250):
    """
    将时间序列数据处理成模型输入的patches格式
    每10秒（window_length）切分一个窗口，每个窗口切分成patches
    
    输入:
        time_series_dict: dict，包含 'imu', 'eeg', 'ecg' 等键
            每个值格式: (channels, total_time_points)
        mode: 模型模式
        window_length: 窗口长度（采样点数，10秒@250Hz=2500）
        hop_length: 步长（采样点数，10秒@250Hz=2500，即不重叠）
        patch_length: patch长度（采样点数，1秒@250Hz=250）
        normalization_stats: 归一化统计量
        sampling_rate: 采样率（Hz）
    
    输出:
        patches_list: list of dict，每个dict包含一个窗口的patches
            格式: {'imu': (num_patches, channels, patch_length), ...}
    """
    # 确定数据的最小长度
    min_length = float('inf')
    modalities_in_data = []
    
    for modal in ['imu', 'eeg', 'ecg']:
        if modal in time_series_dict and time_series_dict[modal] is not None:
            data = time_series_dict[modal]
            if len(data.shape) == 2:  # (channels, time_points)
                min_length = min(min_length, data.shape[1])
                modalities_in_data.append(modal)
    
    if min_length == float('inf') or min_length < window_length:
        raise ValueError(f"数据长度 ({min_length}) 必须大于等于窗口长度 ({window_length})")
    
    # 使用滑动窗口切分数据
    patches_list = []
    window_starts = []
    
    for start_idx in range(0, min_length - window_length + 1, hop_length):
        window_patches = {}
        
        for modal in modalities_in_data:
            data = time_series_dict[modal]
            # 提取窗口: (channels, window_length)
            window_data = data[:, start_idx:start_idx + window_length]
            
            # 归一化
            if normalization_stats is not None:
                window_data = normalize_data(window_data, normalization_stats, modal)
            
            # 切分成patches: (num_patches, channels, patch_length)
            patches = create_patches_from_window(window_data, patch_length)
            window_patches[modal] = patches
        
        patches_list.append(window_patches)
        window_starts.append(start_idx / sampling_rate)  # 转换为秒
    
    return patches_list, window_starts


def predict_comfort_scores(model, patches_list, mode, device):
    """
    对patches列表进行预测
    输入: patches_list: list of dict，每个dict包含一个窗口的patches
    输出: predictions: (num_windows,), probabilities: (num_windows, num_classes)
    """
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for window_patches in tqdm(patches_list, desc='预测中', unit='窗口'):
            # 转换为torch tensor并添加batch维度
            if mode == 'allmix':
                imu_patches = torch.from_numpy(window_patches['imu']).float().unsqueeze(0).to(device)
                eeg_patches = torch.from_numpy(window_patches['eeg']).float().unsqueeze(0).to(device)
                ecg_patches = torch.from_numpy(window_patches['ecg']).float().unsqueeze(0).to(device)
                outputs = model(imu_patches, eeg_patches, ecg_patches)
            elif mode == 'newmix' or mode == 'simplemix' or mode == 'mix':
                imu_patches = torch.from_numpy(window_patches['imu']).float().unsqueeze(0).to(device)
                eeg_patches = torch.from_numpy(window_patches['eeg']).float().unsqueeze(0).to(device)
                ecg_patches = torch.from_numpy(window_patches['ecg']).float().unsqueeze(0).to(device)
                outputs = model(imu_patches, eeg_patches, ecg_patches)
            elif mode == 'imu' or mode == 'rawimu':
                imu_patches = torch.from_numpy(window_patches['imu']).float().unsqueeze(0).to(device)
                outputs = model(imu_patches)
            else:  # physio
                eeg_patches = torch.from_numpy(window_patches['eeg']).float().unsqueeze(0).to(device)
                ecg_patches = torch.from_numpy(window_patches['ecg']).float().unsqueeze(0).to(device)
                outputs = model(eeg_patches, ecg_patches)
            
            # 计算概率和预测类别
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
            
            predictions.append(pred)
            probabilities.append(probs)
    
    return np.array(predictions), np.array(probabilities)


def visualize_predictions(window_starts, predictions, probabilities, save_path=None):
    """
    可视化预测结果
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 第一个子图：预测类别随时间变化
    axes[0].plot(window_starts, predictions, 'o-', linewidth=2, markersize=8, label='预测舒适度等级')
    axes[0].set_xlabel('时间 (秒)', fontsize=12)
    axes[0].set_ylabel('舒适度等级', fontsize=12)
    axes[0].set_title('舒适度预测随时间变化', fontsize=14, fontweight='bold')
    axes[0].set_ylim([-0.5, 4.5])
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(['0分', '1分', '2分', '3分', '4分'])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # 第二个子图：各类别概率随时间变化
    num_classes = probabilities.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        axes[1].plot(window_starts, probabilities[:, i], '-', linewidth=2, 
                    label=f'{i}分', color=colors[i], alpha=0.7)
    
    axes[1].set_xlabel('时间 (秒)', fontsize=12)
    axes[1].set_ylabel('预测概率', fontsize=12)
    axes[1].set_title('各类别预测概率随时间变化', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化结果已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_predictions_summary(window_starts, predictions, probabilities):
    """打印预测结果摘要"""
    print("\n" + "=" * 60)
    print("预测结果摘要")
    print("=" * 60)
    
    for i, (t, pred, probs) in enumerate(zip(window_starts, predictions, probabilities)):
        max_prob = np.max(probs)
        print(f"时间 {t:6.1f}秒: 预测等级={pred}分 (概率={max_prob:.3f}) | "
              f"各等级概率: " + " | ".join([f"{j}分={probs[j]:.3f}" for j in range(len(probs))]))
    
    print("=" * 60)
    print(f"总窗口数: {len(predictions)}")
    print(f"平均预测等级: {np.mean(predictions):.2f}分")
    print(f"预测等级分布: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='模型测试可视化')
    parser.add_argument('-checkpoint_path', required=True, type=str,
                        help='checkpoint文件路径（.pkl）')
    parser.add_argument('-data_path', required=True, type=str,
                        help='测试数据路径（包含.npy文件的目录）')
    parser.add_argument('-output_path', default=None, type=str,
                        help='输出路径（保存可视化结果，默认为checkpoint目录）')
    parser.add_argument('-gpu', default=0, type=int, help='使用的GPU编号')
    parser.add_argument('-window_length', default=2500, type=int,
                        help='窗口长度（采样点数，10秒@250Hz=2500）')
    parser.add_argument('-hop_length', default=2500, type=int,
                        help='步长（采样点数，10秒@250Hz=2500，即每10秒一个窗口）')
    parser.add_argument('-patch_length', default=250, type=int,
                        help='patch长度（采样点数，1秒@250Hz=250）')
    parser.add_argument('-sampling_rate', default=250, type=int,
                        help='采样率（Hz）')
    parser.add_argument('-normalization_stats_path', default=None, type=str,
                        help='归一化统计量文件路径（.pkl）')
    parser.add_argument('-mode', default=None, type=str,
                        choices=['allmix', 'newmix', 'simplemix', 'mix', 'imu', 'rawimu', 'physio'],
                        help='模型模式（如果不指定，将从checkpoint或文件夹名推断）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载checkpoint
    checkpoint = load_checkpoint(args.checkpoint_path)
    
    # 初始化模型（允许从命令行参数指定mode，否则自动推断）
    mode_arg = getattr(args, 'mode', None)
    model, mode, args_dict = init_model_from_checkpoint(checkpoint, args.checkpoint_path, device, mode=mode_arg)
    
    # 加载归一化统计量（如果提供）
    normalization_stats = None
    if args.normalization_stats_path and os.path.exists(args.normalization_stats_path):
        print(f"加载归一化统计量: {args.normalization_stats_path}")
        with open(args.normalization_stats_path, 'rb') as f:
            normalization_stats = pickle.load(f)
        print("✓ 归一化统计量已加载")
    
    # 加载测试数据
    print(f"\n加载测试数据: {args.data_path}")
    time_series_dict = {}
    
    # 尝试加载各个模态的数据
    for modal in ['imu', 'eeg', 'ecg']:
        data_file = os.path.join(args.data_path, f'{modal}.npy')
        if os.path.exists(data_file):
            data = np.load(data_file)
            # 假设数据格式为 (channels, time_points)
            if len(data.shape) == 2:
                time_series_dict[modal] = data
                print(f"  ✓ 加载{modal.upper()}数据: {data.shape}")
            else:
                print(f"  ⚠️  {modal.upper()}数据格式不正确，跳过")
        else:
            print(f"  ⚠️  未找到{modal.upper()}数据文件: {data_file}")
    
    if len(time_series_dict) == 0:
        raise ValueError("未找到任何测试数据文件")
    
    # 处理时间序列为patches格式
    print(f"\n处理时间序列数据...")
    patches_list, window_starts = process_time_series_to_patches(
        time_series_dict=time_series_dict,
        mode=mode,
        window_length=args.window_length,
        hop_length=args.hop_length,
        patch_length=args.patch_length,
        normalization_stats=normalization_stats,
        sampling_rate=args.sampling_rate
    )
    print(f"✓ 数据已处理: {len(patches_list)}个窗口")
    
    # 进行预测
    print(f"\n开始预测...")
    predictions, probabilities = predict_comfort_scores(model, patches_list, mode, device)
    
    # 打印预测结果摘要
    print_predictions_summary(window_starts, predictions, probabilities)
    
    # 可视化结果
    if args.output_path:
        output_dir = args.output_path
    else:
        output_dir = os.path.dirname(args.checkpoint_path)
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'comfort_predictions_{timestamp}.png')
    
    visualize_predictions(window_starts, predictions, probabilities, save_path=save_path)
    
    # 保存预测结果
    results_file = os.path.join(output_dir, f'comfort_predictions_{timestamp}.npz')
    np.savez(results_file, 
             window_starts=window_starts,
             predictions=predictions,
             probabilities=probabilities)
    print(f"✓ 预测结果已保存到: {results_file}")


if __name__ == '__main__':
    main()
