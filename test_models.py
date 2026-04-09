#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型测试脚本
从data/training_dataset_random路径下load数据集，采用random的方式（即不按照被试划分）分为训练、验证、测试集，
在results路径下找到每个文件夹中的checkpoint_best.pkl进行测试，
结果（包括acc,f1等的数据指标文档以及confusion_matrix图片）保存在相应的文件夹下
"""

import os
import sys
import argparse

# 抑制 macOS MallocStackLogging 警告
# os.environ['MallocStackLogging'] = '0'
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 导入项目模块
from dataset import DataArranger, PhysiologicalDataset, IMUDataset, MixDataset, RawIMUDataset
from models.comfort_model import ComfortClassificationModel, IMUClassificationModel, MixClassificationModel
from base.loss_function import ClassificationLoss


def parse_folder_name(folder_name):
    """
    从文件夹名称解析模型信息
    格式: MotionSickness_{ModelName}_{Stamp}_{mode}_fold{fold}_seed{seed}
    例如: MotionSickness_PhysioClassificationModel_PhysioFusionNet_v1_physio_fold1_seed42_random
    """
    parts = folder_name.split('_')
    info = {
        'model_name': None,
        'mode': 'physio',
        'fold': None,
        'seed': None,
        'is_random': 'random' in folder_name
    }
    
    # 查找model_name（通常在MotionSickness之后）
    if 'MotionSickness' in parts:
        idx = parts.index('MotionSickness')
        if idx + 1 < len(parts):
            info['model_name'] = parts[idx + 1]
    
    # 查找mode
    for mode in ['physio', 'imu', 'mix', 'rawimu']:
        if mode in parts:
            info['mode'] = mode
            break
    
    # 查找fold
    for part in parts:
        if part.startswith('fold'):
            try:
                info['fold'] = int(part.replace('fold', ''))
            except:
                pass
    
    # 查找seed
    for part in parts:
        if part.startswith('seed'):
            try:
                info['seed'] = int(part.replace('seed', ''))
            except:
                pass
    
    return info


def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def init_model_from_info(model_name, mode, args_dict):
    """根据模型信息初始化模型"""
    num_classes = args_dict.get('num_classes', 5)
    
    if mode == 'imu':
        model = IMUClassificationModel(
            imu_channels=args_dict.get('imu_channels', 18),
            patch_length=args_dict.get('patch_length', 250),
            num_patches=args_dict.get('num_patches', 10),
            encoding_dim=args_dict.get('encoding_dim', 256),
            num_heads=args_dict.get('num_heads', 8),
            num_classes=num_classes,
            attention_output_mode=args_dict.get('attention_output_mode', 'global'),
            hidden_dims=args_dict.get('hidden_dims', [512, 256, 128]),
            dropout=args_dict.get('dropout', 0.1),
        )
    elif mode == 'rawimu':
        model = IMUClassificationModel(
            imu_channels=6,  # Raw IMU只使用6维
            patch_length=args_dict.get('patch_length', 250),
            num_patches=args_dict.get('num_patches', 10),
            encoding_dim=args_dict.get('encoding_dim', 256),
            num_heads=args_dict.get('num_heads', 8),
            num_classes=num_classes,
            attention_output_mode=args_dict.get('attention_output_mode', 'global'),
            hidden_dims=args_dict.get('hidden_dims', [512, 256, 128]),
            dropout=args_dict.get('dropout', 0.1),
        )
    elif mode == 'mix':
        model = MixClassificationModel(
            imu_channels=args_dict.get('imu_channels', 18),
            eeg_channels=args_dict.get('eeg_channels', 59),
            ecg_channels=args_dict.get('ecg_channels', 1),
            patch_length=args_dict.get('patch_length', 250),
            num_patches=args_dict.get('num_patches', 10),
            encoding_dim=args_dict.get('encoding_dim', 256),
            num_heads=args_dict.get('num_heads', 8),
            num_classes=num_classes,
            attention_output_mode=args_dict.get('attention_output_mode', 'global'),
            hidden_dims=args_dict.get('hidden_dims', [512, 256, 128]),
            dropout=args_dict.get('dropout', 0.1),
        )
    else:  # physio
        model = ComfortClassificationModel(
            eeg_channels=args_dict.get('eeg_channels', 59),
            ecg_channels=args_dict.get('ecg_channels', 1),
            patch_length=args_dict.get('patch_length', 250),
            num_patches=args_dict.get('num_patches', 10),
            encoding_dim=args_dict.get('encoding_dim', 256),
            num_heads=args_dict.get('num_heads', 8),
            num_classes=num_classes,
            attention_output_mode=args_dict.get('attention_output_mode', 'global'),
            hidden_dims=args_dict.get('hidden_dims', [512, 256, 128]),
            dropout=args_dict.get('dropout', 0.1),
        )
    
    return model


def split_data_random(data, seed=42, train_ratio=0.7, val_ratio=0.15):
    """
    随机划分数据（不按被试划分）
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    test_ratio: 测试集比例 = 1 - train_ratio - val_ratio
    """
    n_samples = len(data.get('labels', []))
    indices = np.arange(n_samples)
    
    # 随机打乱
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    # 计算划分点
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"  数据拆分: 训练集 {len(train_indices)}, 验证集 {len(val_indices)}, 测试集 {len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def subset_data(data, indices):
    """
    Create subset of data based on indices
    优化：对于numpy数组，不复制数据，只保存索引映射，避免复制大量数据（参考experiment.py）
    """
    subset_data = {}
    
    # 保存索引映射，而不是复制数据（转换为numpy数组以便快速索引）
    if isinstance(indices, np.ndarray):
        subset_data['_indices'] = indices
    else:
        subset_data['_indices'] = np.asarray(indices, dtype=np.int64)
    
    for key, value in data.items():
        if key == '_indices':
            continue  # 跳过已添加的索引
        if isinstance(value, list) and len(value) > 0:
            # 列表格式：需要复制（因为需要保持列表结构）
            subset_data[key] = [value[i] for i in indices]
        elif isinstance(value, np.ndarray):
            # numpy数组格式：不复制，直接保存原始数据和索引映射
            # Dataset会通过索引映射访问原始数据
            subset_data[key] = value  # 保存原始数据引用
        else:
            # 其他类型（如标量）：直接复制
            subset_data[key] = value
    
    return subset_data


def compute_global_normalization_stats(train_data, mode):
    """
    计算全局归一化统计量（Random划分时使用，只使用训练集）
    优化：使用索引映射，避免复制数据到内存
    返回格式: {modality: {mean, std}}
    """
    stats = {}
    indices = train_data.get('_indices', None)
    
    def extract_data_for_normalization(modality_data, indices, modality_name):
        """提取数据用于归一化统计量计算
        使用索引映射，避免复制数据
        """
        if isinstance(modality_data, np.ndarray):
            if len(modality_data.shape) == 4:  # (num_windows, num_patches, channels, patch_samples)
                if indices is not None:
                    # 使用索引映射访问数据（不复制）
                    return modality_data[indices]
                else:
                    return modality_data
        return None
    
    if mode == 'imu' or mode == 'rawimu':
        if 'imu' in train_data and train_data['imu'] is not None:
            imu_data = train_data['imu']
            all_imu = extract_data_for_normalization(imu_data, indices, 'imu')
            if all_imu is not None and all_imu.size > 0:
                num_samples, num_patches, num_channels, patch_length = all_imu.shape
                if num_samples > 0:
                    # rawimu模式：只使用前6维
                    if mode == 'rawimu':
                        all_imu = all_imu[:, :, :6, :]
                        num_channels = 6
                    
                    all_imu = np.nan_to_num(all_imu, nan=0.0, posinf=0.0, neginf=0.0)
                    reshaped = all_imu.reshape(-1, num_channels, patch_length)
                    imu_mean = reshaped.mean(axis=(0, 2))
                    imu_std = reshaped.std(axis=(0, 2))
                    zero_std_mask = imu_std < 1e-8
                    if zero_std_mask.any():
                        imu_std[zero_std_mask] = 1.0
                    imu_mean = np.nan_to_num(imu_mean, nan=0.0)
                    imu_std = np.nan_to_num(imu_std, nan=1.0)
                    stats['imu'] = {'mean': imu_mean, 'std': imu_std}
    
    elif mode == 'mix':
        # IMU统计量
        if 'imu' in train_data and train_data['imu'] is not None:
            imu_data = train_data['imu']
            all_imu = extract_data_for_normalization(imu_data, indices, 'imu')
            if all_imu is not None and all_imu.size > 0:
                num_samples, num_patches, num_channels, patch_length = all_imu.shape
                if num_samples > 0:
                    all_imu = np.nan_to_num(all_imu, nan=0.0, posinf=0.0, neginf=0.0)
                    reshaped = all_imu.reshape(-1, num_channels, patch_length)
                    imu_mean = reshaped.mean(axis=(0, 2))
                    imu_std = reshaped.std(axis=(0, 2))
                    zero_std_mask = imu_std < 1e-8
                    if zero_std_mask.any():
                        imu_std[zero_std_mask] = 1.0
                    imu_mean = np.nan_to_num(imu_mean, nan=0.0)
                    imu_std = np.nan_to_num(imu_std, nan=1.0)
                    stats['imu'] = {'mean': imu_mean, 'std': imu_std}
        
        # EEG统计量
        if 'eeg' in train_data and train_data['eeg'] is not None:
            eeg_data = train_data['eeg']
            all_eeg = extract_data_for_normalization(eeg_data, indices, 'eeg')
            if all_eeg is not None and all_eeg.size > 0:
                num_samples, num_patches, num_channels, patch_length = all_eeg.shape
                if num_samples > 0:
                    all_eeg = np.nan_to_num(all_eeg, nan=0.0, posinf=0.0, neginf=0.0)
                    reshaped = all_eeg.reshape(-1, num_channels, patch_length)
                    eeg_mean = reshaped.mean(axis=(0, 2))
                    eeg_std = reshaped.std(axis=(0, 2))
                    zero_std_mask = eeg_std < 1e-8
                    if zero_std_mask.any():
                        eeg_std[zero_std_mask] = 1.0
                    eeg_mean = np.nan_to_num(eeg_mean, nan=0.0)
                    eeg_std = np.nan_to_num(eeg_std, nan=1.0)
                    stats['eeg'] = {'mean': eeg_mean, 'std': eeg_std}
        
        # ECG统计量
        if 'ecg' in train_data and train_data['ecg'] is not None:
            ecg_data = train_data['ecg']
            all_ecg = extract_data_for_normalization(ecg_data, indices, 'ecg')
            if all_ecg is not None and all_ecg.size > 0:
                num_samples, num_patches, num_channels, patch_length = all_ecg.shape
                if num_samples > 0:
                    all_ecg = np.nan_to_num(all_ecg, nan=0.0, posinf=0.0, neginf=0.0)
                    reshaped = all_ecg.reshape(-1, num_channels, patch_length)
                    ecg_mean = reshaped.mean(axis=(0, 2))
                    ecg_std = reshaped.std(axis=(0, 2))
                    zero_std_mask = ecg_std < 1e-8
                    if zero_std_mask.any():
                        ecg_std[zero_std_mask] = 1.0
                    ecg_mean = np.nan_to_num(ecg_mean, nan=0.0)
                    ecg_std = np.nan_to_num(ecg_std, nan=1.0)
                    stats['ecg'] = {'mean': ecg_mean, 'std': ecg_std}
    
    else:  # physio
        # EEG统计量
        if 'eeg' in train_data and train_data['eeg'] is not None:
            eeg_data = train_data['eeg']
            all_eeg = extract_data_for_normalization(eeg_data, indices, 'eeg')
            if all_eeg is not None and all_eeg.size > 0:
                num_samples, num_patches, num_channels, patch_length = all_eeg.shape
                if num_samples > 0:
                    all_eeg = np.nan_to_num(all_eeg, nan=0.0, posinf=0.0, neginf=0.0)
                    reshaped = all_eeg.reshape(-1, num_channels, patch_length)
                    eeg_mean = reshaped.mean(axis=(0, 2))
                    eeg_std = reshaped.std(axis=(0, 2))
                    zero_std_mask = eeg_std < 1e-8
                    if zero_std_mask.any():
                        eeg_std[zero_std_mask] = 1.0
                    eeg_mean = np.nan_to_num(eeg_mean, nan=0.0)
                    eeg_std = np.nan_to_num(eeg_std, nan=1.0)
                    stats['eeg'] = {'mean': eeg_mean, 'std': eeg_std}
        
        # ECG统计量
        if 'ecg' in train_data and train_data['ecg'] is not None:
            ecg_data = train_data['ecg']
            all_ecg = extract_data_for_normalization(ecg_data, indices, 'ecg')
            if all_ecg is not None and all_ecg.size > 0:
                num_samples, num_patches, num_channels, patch_length = all_ecg.shape
                if num_samples > 0:
                    all_ecg = np.nan_to_num(all_ecg, nan=0.0, posinf=0.0, neginf=0.0)
                    reshaped = all_ecg.reshape(-1, num_channels, patch_length)
                    ecg_mean = reshaped.mean(axis=(0, 2))
                    ecg_std = reshaped.std(axis=(0, 2))
                    zero_std_mask = ecg_std < 1e-8
                    if zero_std_mask.any():
                        ecg_std[zero_std_mask] = 1.0
                    ecg_mean = np.nan_to_num(ecg_mean, nan=0.0)
                    ecg_std = np.nan_to_num(ecg_std, nan=1.0)
                    stats['ecg'] = {'mean': ecg_mean, 'std': ecg_std}
    
    return stats


def test_model(model, test_loader, device, criterion, mode):
    """测试模型"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing', unit='batch'):
            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 准备输入数据
            if mode == 'imu' or mode == 'rawimu':
                imu_patches = batch.get('imu', None)
                if imu_patches is not None:
                    if len(imu_patches.shape) == 3:
                        imu_patches = imu_patches.unsqueeze(0)
                    outputs = model(imu_patches)
                else:
                    raise ValueError("IMU model requires 'imu' in batch")
            elif mode == 'mix':
                imu_patches = batch.get('imu', None)
                eeg_patches = batch.get('eeg', None)
                ecg_patches = batch.get('ecg', None)
                if imu_patches is not None and eeg_patches is not None and ecg_patches is not None:
                    if len(imu_patches.shape) == 3:
                        imu_patches = imu_patches.unsqueeze(0)
                    if len(eeg_patches.shape) == 3:
                        eeg_patches = eeg_patches.unsqueeze(0)
                    if len(ecg_patches.shape) == 3:
                        ecg_patches = ecg_patches.unsqueeze(0)
                    outputs = model(imu_patches, eeg_patches, ecg_patches)
                else:
                    raise ValueError("Mix model requires 'imu', 'eeg' and 'ecg' in batch")
            else:  # physio
                eeg_patches = batch.get('eeg', None)
                ecg_patches = batch.get('ecg', None)
                if eeg_patches is not None and ecg_patches is not None:
                    if len(eeg_patches.shape) == 3:
                        eeg_patches = eeg_patches.unsqueeze(0)
                    if len(ecg_patches.shape) == 3:
                        ecg_patches = ecg_patches.unsqueeze(0)
                    outputs = model(eeg_patches, ecg_patches)
                else:
                    raise ValueError("Physio model requires both 'eeg' and 'ecg' in batch")
            
            targets = batch['label'].squeeze()
            
            # 计算loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # 收集预测结果
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 计算指标
    if len(all_targets) == 0:
        print("  ⚠️  警告: 测试集为空，无法计算指标")
        return None
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro', zero_division=0
    )
    
    # 创建混淆矩阵
    try:
        cm = confusion_matrix(all_targets, all_predictions)
    except Exception as e:
        print(f"  ⚠️  警告: 计算混淆矩阵时出错: {e}")
        cm = np.array([])
    
    results = {
        'overall': {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'confusion_matrix': cm,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probabilities)
    }
    
    return results


def save_results(results, save_path):
    """保存测试结果"""
    os.makedirs(save_path, exist_ok=True)
    
    # 保存指标
    metrics = results['overall']
    metrics_file = os.path.join(save_path, 'test_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Test Precision: {metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {metrics['recall']:.4f}\n")
        f.write(f"Test F1: {metrics['f1']:.4f}\n")
        f.write(f"Test Loss: {metrics['loss']:.4f}\n")
    
    print(f"  ✓ 指标已保存到: {metrics_file}")
    
    # 保存混淆矩阵图
    cm = results['confusion_matrix']
    if cm.size > 0 and cm.shape[0] > 0:
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_file = os.path.join(save_path, 'confusion_matrix.png')
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 混淆矩阵图已保存到: {cm_file}")
        except Exception as e:
            print(f"  ⚠️  警告: 保存混淆矩阵图时出错: {e}")
    else:
        print(f"  ⚠️  警告: 混淆矩阵为空，跳过保存混淆矩阵图")
    
    # 保存预测结果
    np.save(os.path.join(save_path, 'test_predictions.npy'), results['predictions'])
    np.save(os.path.join(save_path, 'test_targets.npy'), results['targets'])
    np.save(os.path.join(save_path, 'test_probabilities.npy'), results['probabilities'])
    print(f"  ✓ 预测结果已保存")


def main():
    parser = argparse.ArgumentParser(description='测试训练好的模型')
    parser.add_argument('-dataset_path', default='./data/training_dataset_random', type=str,
                        help='数据集路径')
    parser.add_argument('-results_path', default='./results', type=str,
                        help='结果文件夹路径')
    parser.add_argument('-gpu', default=0, type=int, help='使用的GPU编号')
    parser.add_argument('-batch_size', default=32, type=int, help='批次大小')
    parser.add_argument('-seed', default=42, type=int, help='随机种子（用于数据划分）')
    parser.add_argument('-num_classes', default=5, type=int, help='类别数')
    parser.add_argument('-mode', default='physio', type=str, choices=['physio', 'imu', 'rawimu', 'mix'],
                        help='要测试的模型模式（physio, imu, rawimu, mix），默认physio')
    parser.add_argument('-splits_cache_dir', default='./data/splits', type=str,
                        help='保存/加载数据划分的缓存目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建缓存目录
    os.makedirs(args.splits_cache_dir, exist_ok=True)
    
    # 检查是否已有保存的划分
    splits_cache_file = os.path.join(args.splits_cache_dir, f'test_splits_seed{args.seed}.pkl')
    
    if os.path.exists(splits_cache_file):
        print("\n" + "=" * 60)
        print("加载已保存的数据划分...")
        print("=" * 60)
        with open(splits_cache_file, 'rb') as f:
            splits_cache = pickle.load(f)
        train_indices = splits_cache['train_indices']
        val_indices = splits_cache['val_indices']
        test_indices = splits_cache['test_indices']
        default_mode = splits_cache.get('default_mode', 'physio')
        print(f"  ✓ 数据划分加载完成 (模式: {default_mode})")
        print(f"    训练集: {len(train_indices)}, 验证集: {len(val_indices)}, 测试集: {len(test_indices)}")
        
        # 加载数据（使用mmap模式，不加载到内存）
        print("\n" + "=" * 60)
        print("加载数据集（使用mmap模式，不加载到内存）...")
        print("=" * 60)
        data_arranger = DataArranger(None, args.dataset_path, debug=0)
        print(f"  正在加载{default_mode}模式数据...")
        data = data_arranger.load_data(mode=default_mode, use_cache=False)
        print(f"  ✓ 数据加载完成（mmap模式，不占用内存）")
    else:
        # 加载数据
        print("\n" + "=" * 60)
        print("加载数据集（使用mmap模式，不加载到内存）...")
        print("=" * 60)
        data_arranger = DataArranger(None, args.dataset_path, debug=0)
        
        # 从文件夹名称推断mode（先尝试加载一个checkpoint来获取mode）
        # 这里我们先加载数据，然后根据checkpoint来确定mode
        print("  正在加载数据...")
        # 先尝试physio模式
        try:
            data = data_arranger.load_data(mode='physio', use_cache=False)
            default_mode = 'physio'
        except:
            try:
                data = data_arranger.load_data(mode='imu', use_cache=False)
                default_mode = 'imu'
            except:
                try:
                    data = data_arranger.load_data(mode='mix', use_cache=False)
                    default_mode = 'mix'
                except:
                    print("  ⚠️  无法确定数据模式，将根据checkpoint信息推断")
                    data = None
                    default_mode = None
        
        if data is None:
            print("  ❌ 无法加载数据，请检查数据集路径")
            return
        
        print(f"  ✓ 数据加载完成（mmap模式，不占用内存），默认模式: {default_mode}")
        
        # 随机划分数据
        print("\n随机划分数据...")
        train_indices, val_indices, test_indices = split_data_random(data, seed=args.seed)
        
        # 保存划分
        print(f"  保存数据划分到: {splits_cache_file}")
        splits_cache = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'default_mode': default_mode,
            'seed': args.seed
        }
        with open(splits_cache_file, 'wb') as f:
            pickle.dump(splits_cache, f)
        print(f"  ✓ 数据划分已保存")
    
    # 创建数据子集（使用索引映射，不复制数据）
    print("\n创建数据子集（使用索引映射，不复制数据）...")
    train_data = subset_data(data, train_indices)
    val_data = subset_data(data, val_indices)
    test_data = subset_data(data, test_indices)
    print(f"  ✓ 数据子集创建完成（使用索引映射，节省内存）")
    
    # 保存data_arranger引用，以便后续切换模式时使用
    # data_arranger已在上面定义
    
    # 查找所有checkpoint
    print("\n" + "=" * 60)
    print("查找checkpoint文件...")
    print("=" * 60)
    
    results_path = args.results_path
    if not os.path.exists(results_path):
        print(f"  ❌ 结果路径不存在: {results_path}")
        return
    
    checkpoint_folders = []
    for item in os.listdir(results_path):
        item_path = os.path.join(results_path, item)
        if os.path.isdir(item_path):
            checkpoint_path = os.path.join(item_path, 'checkpoint_best.pkl')
            if os.path.exists(checkpoint_path):
                # 快速检查模式（不加载完整checkpoint）
                model_info = parse_folder_name(item)
                if model_info['mode'] == args.mode:
                    checkpoint_folders.append((item, item_path, checkpoint_path))
                    print(f"  ✓ 找到checkpoint: {item} (模式: {model_info['mode']})")
                else:
                    print(f"  ⏭️  跳过checkpoint: {item} (模式: {model_info['mode']}, 需要: {args.mode})")
    
    if len(checkpoint_folders) == 0:
        print("  ❌ 未找到任何checkpoint_best.pkl文件")
        return
    
    print(f"\n共找到 {len(checkpoint_folders)} 个checkpoint")
    
    # 测试每个checkpoint
    print("\n" + "=" * 60)
    print(f"开始测试模型（模式筛选: {args.mode}）...")
    print("=" * 60)
    
    for folder_name, folder_path, checkpoint_path in checkpoint_folders:
        print(f"\n{'='*60}")
        print(f"测试模型: {folder_name}")
        print(f"{'='*60}")
        
        try:
            # 解析文件夹名称获取模型信息
            model_info = parse_folder_name(folder_name)
            print(f"  模型名称: {model_info['model_name']}")
            print(f"  模式: {model_info['mode']}")
            print(f"  Fold: {model_info['fold']}")
            print(f"  Seed: {model_info['seed']}")
            
            mode = model_info['mode']
            
            # 检查模式是否匹配
            if mode != args.mode:
                print(f"  ⏭️  跳过：模型模式为 {mode}，但指定测试模式为 {args.mode}")
                continue
            
            # 根据mode加载对应的数据（如果需要）
            if mode != default_mode:
                print(f"  切换到{mode}模式，重新加载数据（使用mmap模式）...")
                try:
                    data = data_arranger.load_data(mode=mode, use_cache=False)
                    # 使用已保存的划分（如果存在）或重新划分
                    splits_cache_file_mode = os.path.join(args.splits_cache_dir, f'test_splits_{mode}_seed{args.seed}.pkl')
                    if os.path.exists(splits_cache_file_mode):
                        with open(splits_cache_file_mode, 'rb') as f:
                            splits_cache = pickle.load(f)
                        train_indices = splits_cache['train_indices']
                        val_indices = splits_cache['val_indices']
                        test_indices = splits_cache['test_indices']
                        print(f"  ✓ 使用已保存的{mode}模式数据划分")
                    else:
                        # 重新划分并保存
                        train_indices, val_indices, test_indices = split_data_random(data, seed=args.seed)
                        splits_cache = {
                            'train_indices': train_indices,
                            'val_indices': val_indices,
                            'test_indices': test_indices,
                            'default_mode': mode,
                            'seed': args.seed
                        }
                        with open(splits_cache_file_mode, 'wb') as f:
                            pickle.dump(splits_cache, f)
                        print(f"  ✓ {mode}模式数据划分已保存")
                    
                    train_data = subset_data(data, train_indices)
                    val_data = subset_data(data, val_indices)
                    test_data = subset_data(data, test_indices)
                    print(f"  ✓ {mode}模式数据加载完成（使用索引映射，节省内存）")
                except Exception as e:
                    print(f"  ❌ 无法加载{mode}模式数据: {e}")
                    continue
            
            # 加载checkpoint
            print(f"  加载checkpoint: {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path)
            
            # 从checkpoint获取模型参数
            trainer_state = checkpoint.get('trainer_state', {})
            model_state_dict = trainer_state.get('model_state_dict', None)
            
            if model_state_dict is None:
                print(f"  ⚠️  警告: checkpoint中未找到模型状态，跳过")
                continue
            
            # 从checkpoint获取参数（如果存在）
            # checkpoint可能没有保存args，使用默认值
            checkpoint_args = checkpoint.get('args', {})
            if not isinstance(checkpoint_args, dict):
                checkpoint_args = {}
            
            # 默认参数值（与训练时保持一致）
            args_dict = {
                'num_classes': checkpoint_args.get('num_classes', args.num_classes) if checkpoint_args else args.num_classes,
                'patch_length': checkpoint_args.get('patch_length', 250) if checkpoint_args else 250,
                'num_patches': checkpoint_args.get('num_patches', 10) if checkpoint_args else 10,
                'encoding_dim': checkpoint_args.get('encoding_dim', 256) if checkpoint_args else 256,
                'num_heads': checkpoint_args.get('num_heads', 8) if checkpoint_args else 8,
                'attention_output_mode': checkpoint_args.get('attention_output_mode', 'global') if checkpoint_args else 'global',
                'hidden_dims': checkpoint_args.get('hidden_dims', [512, 256, 128]) if checkpoint_args else [512, 256, 128],
                'dropout': checkpoint_args.get('dropout', 0.1) if checkpoint_args else 0.1,
                'eeg_channels': checkpoint_args.get('eeg_channels', 59) if checkpoint_args else 59,
                'ecg_channels': checkpoint_args.get('ecg_channels', 1) if checkpoint_args else 1,
                'imu_channels': checkpoint_args.get('imu_channels', 18) if checkpoint_args else 18,
                'window_length': checkpoint_args.get('window_length', 2500) if checkpoint_args else 2500,
                'hop_length': checkpoint_args.get('hop_length', 750) if checkpoint_args else 750,
                'normalize_data': checkpoint_args.get('normalize_data', True) if checkpoint_args else True,
                'apply_filter': checkpoint_args.get('apply_filter', False) if checkpoint_args else False,
                'eeg_sampling_rate': checkpoint_args.get('eeg_sampling_rate', 250) if checkpoint_args else 250,
                'ecg_sampling_rate': checkpoint_args.get('ecg_sampling_rate', 250) if checkpoint_args else 250,
                'imu_sampling_rate': checkpoint_args.get('imu_sampling_rate', 250) if checkpoint_args else 250,
            }
            
            # 处理hidden_dims（可能是字符串）
            if isinstance(args_dict['hidden_dims'], str):
                args_dict['hidden_dims'] = [int(x.strip()) for x in args_dict['hidden_dims'].split(',')]
            
            # 初始化模型
            print(f"  初始化模型...")
            model = init_model_from_info(model_info['model_name'], mode, args_dict)
            model.load_state_dict(model_state_dict)
            model.to(device)
            model.eval()
            print(f"  ✓ 模型加载完成")
            
            # 计算归一化统计量（从训练集）
            print(f"  计算归一化统计量（从训练集）...")
            normalization_stats = compute_global_normalization_stats(train_data, mode)
            
            # 创建数据集
            print(f"  创建测试数据集...")
            if mode == 'imu' or mode == 'rawimu':
                test_dataset = IMUDataset(
                    data_dict=test_data,
                    window_length=checkpoint_args.get('window_length', 2500),
                    hop_length=checkpoint_args.get('hop_length', 750),
                    normalize=checkpoint_args.get('normalize_data', True),
                    apply_filter=checkpoint_args.get('apply_filter', False),
                    imu_sampling_rate=checkpoint_args.get('imu_sampling_rate', 250),
                    normalization_stats=normalization_stats
                )
            elif mode == 'mix':
                test_dataset = MixDataset(
                    data_dict=test_data,
                    window_length=checkpoint_args.get('window_length', 2500),
                    hop_length=checkpoint_args.get('hop_length', 750),
                    normalize=checkpoint_args.get('normalize_data', True),
                    apply_filter=checkpoint_args.get('apply_filter', False),
                    imu_sampling_rate=checkpoint_args.get('imu_sampling_rate', 250),
                    eeg_sampling_rate=checkpoint_args.get('eeg_sampling_rate', 250),
                    ecg_sampling_rate=checkpoint_args.get('ecg_sampling_rate', 250),
                    normalization_stats=normalization_stats
                )
            else:  # physio
                modality = ['eeg', 'ecg']
                test_dataset = PhysiologicalDataset(
                    data_dict=test_data,
                    modality=modality,
                    window_length=checkpoint_args.get('window_length', 2500),
                    hop_length=checkpoint_args.get('hop_length', 750),
                    normalize=checkpoint_args.get('normalize_data', True),
                    apply_filter=checkpoint_args.get('apply_filter', False),
                    eeg_sampling_rate=checkpoint_args.get('eeg_sampling_rate', 250),
                    ecg_sampling_rate=checkpoint_args.get('ecg_sampling_rate', 250),
                    normalization_stats=normalization_stats
                )
            
            # 创建DataLoader
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory = False
            )
            
            # 初始化损失函数
            criterion = ClassificationLoss(
                loss_type='cross_entropy',
                num_classes=args_dict['num_classes'],
                class_weights=None
            )
            
            # 测试模型
            print(f"  开始测试...")
            results = test_model(model, test_loader, device, criterion, mode)
            
            if results is None:
                print(f"  ⚠️  测试失败，跳过")
                continue
            
            # 保存结果
            print(f"  保存结果...")
            save_results(results, folder_path)
            
            # 打印结果
            metrics = results['overall']
            print(f"\n  测试结果:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1: {metrics['f1']:.4f}")
            print(f"    Loss: {metrics['loss']:.4f}")
            
            print(f"  ✓ 测试完成")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("所有模型测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

