import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import pickle
from scipy import signal
from sklearn.preprocessing import StandardScaler, LabelEncoder


def map_labels_10_to_5(labels):
    """
    将0-9分的标签映射为5分类：0,1,2,3,4（4分及4分以上都视为4）。
    用于新数据集中保持五分类训练与评估。
    """
    labels = np.asarray(labels, dtype=np.int64)
    return np.minimum(labels, 4)


class ECGDataset(Dataset):
    """
    Dataset class for ECG signals
    支持两种数据格式：
    1. 新格式：patches格式 (num_windows, num_patches, channels, patch_samples)
    2. 旧格式：连续信号格式 (channels, time_points)
    """
    
    def __init__(self, data_dict, modality, window_length=3000, 
                 hop_length=1500, normalize=True, apply_filter=True,
                 ecg_sampling_rate=1000, 
                 normalization_stats=None):
        super().__init__()
        
        self.data_dict = data_dict
        self.modality = modality
        self.window_length = window_length
        self.hop_length = hop_length
        self.normalize = normalize
        self.apply_filter = apply_filter
        self.ecg_sampling_rate = ecg_sampling_rate
        # 归一化统计量（如果提供，则使用全局统计量；否则使用样本内统计量）
        self.normalization_stats = normalization_stats
        
        # Extract data
        # 检查是否有索引映射（用于数据子集，避免复制大数据）
        self.indices = data_dict.get('_indices', None)
        
        self.ecg_data = data_dict.get('ecg', None)
        self.labels = data_dict.get('labels', None)
        self.weights = data_dict.get('weights', None)  # 样本权重
        self.subject_ids = data_dict.get('subject_ids', None)  # 保存subject_ids用于per-subject归一化
        
        # 检测数据格式：新格式（patches）还是旧格式（连续信号）
        self.is_patches_format = False
        self.is_numpy_array = False  # 标记数据是否为numpy数组格式
        if self.ecg_data is not None:
            if isinstance(self.ecg_data, np.ndarray):
                # 数据是numpy数组格式
                self.is_numpy_array = True
                if len(self.ecg_data.shape) == 4:
                    # 新格式：(num_windows, num_patches, channels, patch_samples)
                    self.is_patches_format = True
            elif len(self.ecg_data) > 0:
                # 数据是列表格式
                first_ecg = self.ecg_data[0]
                if isinstance(first_ecg, np.ndarray) and len(first_ecg.shape) == 3:
                    # 新格式：(num_patches, channels, patch_samples)
                    self.is_patches_format = True
        
        # 不在这里预处理！只保存归一化统计量，在 __getitem__ 中按需处理
        # 这样可以避免在初始化时处理整个数据集，节省内存
        self.processed_data = None  # 不再预处理整个数据集
        
        self.segments = None
        
    def _normalize_window(self, window_data, modality, subject_id=None):
        """
        对单个window进行归一化（在线处理）
        window_data: (num_patches, channels, patch_samples)
        subject_id: 该window所属的subject_id，用于per-subject归一化
        """
        num_patches, num_channels, patch_length = window_data.shape
        
        if self.normalization_stats is not None:
            # 检查是否是per-subject归一化格式: {subject_id: {modality: {mean, std}}}
            # 还是旧的全局归一化格式: {modality: {mean, std}}
            first_key = next(iter(self.normalization_stats.keys()))
            
            if isinstance(first_key, (int, np.integer)) or (isinstance(first_key, str) and first_key.isdigit()):
                # Per-subject归一化格式
                if subject_id is not None:
                    subject_id_key = int(subject_id) if isinstance(subject_id, (int, np.integer)) else subject_id
                    subject_stats = self.normalization_stats.get(subject_id_key, None)
                    
                    if subject_stats is not None:
                        stats = subject_stats.get(modality, {})
                        mean = stats.get('mean', None)
                        std = stats.get('std', None)
                        
                        if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                            # 使用该subject的统计量：扩展到 (1, num_channels, 1) 以便广播
                            mean_expanded = mean.reshape(1, num_channels, 1)
                            std_expanded = std.reshape(1, num_channels, 1)
                            std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                            return (window_data - mean_expanded) / (std_expanded + 1e-8)
            else:
                # 旧的全局归一化格式（向后兼容）
                stats = self.normalization_stats.get(modality, {})
                mean = stats.get('mean', None)
                std = stats.get('std', None)
                
                if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                    # 使用全局统计量：扩展到 (1, num_channels, 1) 以便广播
                    mean_expanded = mean.reshape(1, num_channels, 1)
                    std_expanded = std.reshape(1, num_channels, 1)
                    std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                    return (window_data - mean_expanded) / (std_expanded + 1e-8)
        
        # 回退到样本内归一化（对这个window的每个patch进行归一化）
        reshaped = window_data.reshape(-1, patch_length)
        means = reshaped.mean(axis=1, keepdims=True)
        stds = reshaped.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        normalized = (reshaped - means) / (stds + 1e-8)
        return normalized.reshape(num_patches, num_channels, patch_length)
    
    def preprocess_ecg(self, ecg_data):
        """
        Preprocess ECG signals
        支持两种格式：
        1. 新格式：patches (num_patches, channels, patch_samples)
           - 对于已分片的数据，只进行可选的归一化（如果 normalize=True）
           - 归一化不是必须的！如果数据已经在预处理阶段归一化过，可以设置 normalize=False
        2. 旧格式：连续信号 (channels, time_points) - 进行滤波和归一化
        """
        # 如果是numpy数组格式，直接处理整个数组（更快）
        if isinstance(ecg_data, np.ndarray) and self.is_patches_format:
            if self.normalize:
                # 对整个数组进行归一化
                num_windows, num_patches, num_channels, patch_length = ecg_data.shape
                
                if self.normalization_stats is not None:
                    ecg_mean = self.normalization_stats.get('ecg', {}).get('mean', None)
                    ecg_std = self.normalization_stats.get('ecg', {}).get('std', None)
                    
                    if ecg_mean is not None and ecg_std is not None and len(ecg_mean.shape) == 1 and ecg_mean.shape[0] == num_channels:
                        # 使用全局统计量：扩展到 (1, 1, num_channels, 1) 以便广播
                        mean_expanded = ecg_mean.reshape(1, 1, num_channels, 1)
                        std_expanded = ecg_std.reshape(1, 1, num_channels, 1)
                        # 归一化整个数组
                        processed_ecg = (ecg_data - mean_expanded) / (std_expanded + 1e-8)
                    else:
                        # 回退到样本内归一化（优化：避免不必要的复制）
                        reshaped = ecg_data.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        stds = np.where(stds < 1e-8, 1.0, stds)
                        processed_ecg = (reshaped - means) / (stds + 1e-8)
                        processed_ecg = processed_ecg.reshape(num_windows, num_patches, num_channels, patch_length)
                else:
                    # 样本内归一化（优化：避免不必要的复制）
                    reshaped = ecg_data.reshape(-1, patch_length)
                    means = reshaped.mean(axis=1, keepdims=True)
                    stds = reshaped.std(axis=1, keepdims=True)
                    stds = np.where(stds < 1e-8, 1.0, stds)
                    processed_ecg = (reshaped - means) / (stds + 1e-8)
                    processed_ecg = processed_ecg.reshape(num_windows, num_patches, num_channels, patch_length)
            else:
                # 不需要归一化，直接返回（不复制，节省内存）
                processed_ecg = ecg_data
            return processed_ecg
        
        # 列表格式：逐个处理（向后兼容）
        processed_ecg = []
        num_windows = len(ecg_data)
        
        # 使用 tqdm 显示进度（仅在数据量大时）
        use_progress = num_windows > 100
        if use_progress:
            from tqdm import tqdm
            iterator = tqdm(ecg_data, desc="预处理 ECG", unit="窗口")
        else:
            iterator = ecg_data
        
        for window_ecg in iterator:
            if self.is_patches_format:
                # 新格式：patches格式数据
                # window_ecg: (num_patches, channels, patch_samples)
                if self.normalize:
                    window_ecg = window_ecg.copy()
                    num_patches, num_channels, patch_length = window_ecg.shape
                    
                    if self.normalization_stats is not None:
                        # 使用全局统计量（训练集的mean和std）
                        ecg_mean = self.normalization_stats.get('ecg', {}).get('mean', None)
                        ecg_std = self.normalization_stats.get('ecg', {}).get('std', None)
                        
                        if ecg_mean is not None and ecg_std is not None:
                            # 使用全局统计量：对每个通道使用相同的mean和std
                            if len(ecg_mean.shape) == 1 and ecg_mean.shape[0] == num_channels:
                                mean_expanded = ecg_mean.reshape(1, num_channels, 1)
                                std_expanded = ecg_std.reshape(1, num_channels, 1)
                                window_ecg = (window_ecg - mean_expanded) / (std_expanded + 1e-8)
                            else:
                                # 形状不匹配，回退到样本内归一化
                                reshaped = window_ecg.reshape(-1, patch_length)
                                means = reshaped.mean(axis=1, keepdims=True)
                                stds = reshaped.std(axis=1, keepdims=True)
                                reshaped = (reshaped - means) / (stds + 1e-8)
                                window_ecg = reshaped.reshape(num_patches, num_channels, patch_length)
                        else:
                            # 如果没有提供统计量，使用样本内归一化
                            reshaped = window_ecg.reshape(-1, patch_length)
                            means = reshaped.mean(axis=1, keepdims=True)
                            stds = reshaped.std(axis=1, keepdims=True)
                            reshaped = (reshaped - means) / (stds + 1e-8)
                            window_ecg = reshaped.reshape(num_patches, num_channels, patch_length)
                    else:
                        # 没有提供全局统计量，使用样本内归一化（向后兼容）
                        reshaped = window_ecg.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        reshaped = (reshaped - means) / (stds + 1e-8)
                        window_ecg = reshaped.reshape(num_patches, num_channels, patch_length)
                # 如果 normalize=False，直接使用原始数据（不进行归一化）
                processed_ecg.append(window_ecg)
            else:
                # 旧格式：连续信号，进行完整预处理
                # Apply bandpass filter (0.5-40 Hz)
                if self.apply_filter:
                    window_ecg = self.apply_ecg_filter(window_ecg)
                
                # Normalize
                if self.normalize:
                    window_ecg = self.normalize_signal(window_ecg)
                
                processed_ecg.append(window_ecg)
        
        return processed_ecg
    
    def normalize_signal(self, signal_data):
        """
        Normalize signal using z-score normalization
        """
        if len(signal_data.shape) == 1:
            # Single channel
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            if std_val > 0:
                signal_data = (signal_data - mean_val) / std_val
        else:
            # Multiple channels
            for ch in range(signal_data.shape[0]):
                mean_val = np.mean(signal_data[ch])
                std_val = np.std(signal_data[ch])
                if std_val > 0:
                    signal_data[ch] = (signal_data[ch] - mean_val) / std_val
        
        return signal_data
    
    def create_segments(self):
        """
        Create overlapping segments from the signals
        """
        segments = []
        
        # Get the minimum length across all subjects
        min_length = float('inf')
        
        if self.ecg_data is not None:
            min_length = min(min_length, min([ecg.shape[1] for ecg in self.ecg_data]))
        
        # 如果没有找到有效数据，返回空列表
        if min_length == float('inf') or min_length < self.window_length:
            return segments
        
        # Create segments
        # 确保所有参数都是整数
        min_length = int(min_length)
        window_length = int(self.window_length)
        hop_length = int(self.hop_length)
        for start in range(0, min_length - window_length + 1, hop_length):
            end = start + window_length
            segments.append((start, end))
        
        return segments
    
    def __len__(self):
        """Return the number of samples"""
        # 如果有索引映射，使用索引长度
        if self.indices is not None:
            return len(self.indices)
        
        return len(self.data_dict.get('labels', []))
    
    def __getitem__(self, idx):
        """
        Get a sample of data
        """
        sample = {}
        # 如果有索引映射，使用映射后的索引
        if self.indices is not None:
            window_idx = int(self.indices[idx])
        else:
            window_idx = idx
        
        # 获取该window的subject_id（用于per-subject归一化）
        subject_id = None
        if self.subject_ids is not None:
            if isinstance(self.subject_ids, np.ndarray):
                subject_id = int(self.subject_ids[window_idx])
            elif isinstance(self.subject_ids, list):
                subject_id = int(self.subject_ids[window_idx])
        
        if 'ecg' in self.modality and self.ecg_data is not None:
            # 直接从原始数据获取，不复制
            window_ecg = self.ecg_data[window_idx]  # (num_patches, channels, patch_samples)
            # 在线归一化（只对这个window处理）
            if self.normalize:
                window_ecg = self._normalize_window(window_ecg, 'ecg', subject_id=subject_id)
            sample['ecg'] = torch.FloatTensor(window_ecg)
        
        # Add label
        if self.labels is not None:
            label = int(self.labels[window_idx])
            sample['label'] = torch.LongTensor([label])
        
        # Add weight (if available)
        if self.weights is not None:
            weight = float(self.weights[window_idx])
            sample['weight'] = torch.FloatTensor([weight])
        
        # Add subject_id (if available)
        if subject_id is not None:
            sample['subject_id'] = torch.LongTensor([subject_id])
        
        return sample

class EEGDataset(Dataset):
    """
    Dataset class for EEG signals
    支持两种数据格式：
    1. 新格式：patches格式 (num_windows, num_patches, channels, patch_samples)
    2. 旧格式：连续信号格式 (channels, time_points)
    """
    
    def __init__(self, data_dict, modality, window_length=3000, 
                 hop_length=1500, normalize=True, apply_filter=True,
                 eeg_sampling_rate=1000, 
                 normalization_stats=None):
        super().__init__()
        
        self.data_dict = data_dict
        self.modality = modality
        self.window_length = window_length
        self.hop_length = hop_length
        self.normalize = normalize
        self.apply_filter = apply_filter
        self.eeg_sampling_rate = eeg_sampling_rate
        # 归一化统计量（如果提供，则使用全局统计量；否则使用样本内统计量）
        self.normalization_stats = normalization_stats
        
        # Extract data
        # 检查是否有索引映射（用于数据子集，避免复制大数据）
        self.indices = data_dict.get('_indices', None)
        
        self.eeg_data = data_dict.get('eeg', None)
        self.labels = data_dict.get('labels', None)
        self.weights = data_dict.get('weights', None)  # 样本权重
        self.subject_ids = data_dict.get('subject_ids', None)  # 保存subject_ids用于per-subject归一化
        
        # 检测数据格式：新格式（patches）还是旧格式（连续信号）
        self.is_patches_format = False
        self.is_numpy_array = False  # 标记数据是否为numpy数组格式
        if self.eeg_data is not None:
            if isinstance(self.eeg_data, np.ndarray):
                # 数据是numpy数组格式
                self.is_numpy_array = True
                if len(self.eeg_data.shape) == 4:
                    # 新格式：(num_windows, num_patches, channels, patch_samples)
                    self.is_patches_format = True
            elif len(self.eeg_data) > 0:
                # 数据是列表格式
                first_eeg = self.eeg_data[0]
                if isinstance(first_eeg, np.ndarray) and len(first_eeg.shape) == 3:
                    # 新格式：(num_patches, channels, patch_samples)
                    self.is_patches_format = True
        
        # 不在这里预处理！只保存归一化统计量，在 __getitem__ 中按需处理
        # 这样可以避免在初始化时处理整个数据集，节省内存
        self.processed_data = None  # 不再预处理整个数据集
        
        self.segments = None
        
    def _normalize_window(self, window_data, modality, subject_id=None):
        """
        对单个window进行归一化（在线处理）
        window_data: (num_patches, channels, patch_samples)
        subject_id: 该window所属的subject_id，用于per-subject归一化
        """
        num_patches, num_channels, patch_length = window_data.shape
        
        if self.normalization_stats is not None:
            # 检查是否是per-subject归一化格式: {subject_id: {modality: {mean, std}}}
            # 还是旧的全局归一化格式: {modality: {mean, std}}
            first_key = next(iter(self.normalization_stats.keys()))
            
            if isinstance(first_key, (int, np.integer)) or (isinstance(first_key, str) and first_key.isdigit()):
                # Per-subject归一化格式
                if subject_id is not None:
                    subject_id_key = int(subject_id) if isinstance(subject_id, (int, np.integer)) else subject_id
                    subject_stats = self.normalization_stats.get(subject_id_key, None)
                    
                    if subject_stats is not None:
                        stats = subject_stats.get(modality, {})
                        mean = stats.get('mean', None)
                        std = stats.get('std', None)
                        
                        if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                            # 使用该subject的统计量：扩展到 (1, num_channels, 1) 以便广播
                            mean_expanded = mean.reshape(1, num_channels, 1)
                            std_expanded = std.reshape(1, num_channels, 1)
                            std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                            return (window_data - mean_expanded) / (std_expanded + 1e-8)
            else:
                # 旧的全局归一化格式（向后兼容）
                stats = self.normalization_stats.get(modality, {})
                mean = stats.get('mean', None)
                std = stats.get('std', None)
                
                if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                    # 使用全局统计量：扩展到 (1, num_channels, 1) 以便广播
                    mean_expanded = mean.reshape(1, num_channels, 1)
                    std_expanded = std.reshape(1, num_channels, 1)
                    std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                    return (window_data - mean_expanded) / (std_expanded + 1e-8)
        
        # 回退到样本内归一化（对这个window的每个patch进行归一化）
        reshaped = window_data.reshape(-1, patch_length)
        means = reshaped.mean(axis=1, keepdims=True)
        stds = reshaped.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        normalized = (reshaped - means) / (stds + 1e-8)
        return normalized.reshape(num_patches, num_channels, patch_length)
    
    def preprocess_eeg(self, eeg_data):
        """
        Preprocess EEG signals
        支持两种格式：
        1. 新格式：patches (num_patches, channels, patch_samples) 
           - 对于已分片的数据，只进行可选的归一化（如果 normalize=True）
           - 归一化不是必须的！如果数据已经在预处理阶段归一化过，可以设置 normalize=False
           - 归一化的作用：
             * 将每个patch的每个通道标准化到均值0、标准差1
             * 有助于神经网络训练稳定性和收敛速度
             * 但如果数据范围已经合理（如已在[-1,1]或[0,1]范围），可以跳过
        2. 旧格式：连续信号 (channels, time_points) - 进行滤波和归一化
        """
        # 如果是numpy数组格式，直接处理整个数组（更快）
        if isinstance(eeg_data, np.ndarray) and self.is_patches_format:
            if self.normalize:
                # 对整个数组进行归一化
                num_windows, num_patches, num_channels, patch_length = eeg_data.shape
                
                if self.normalization_stats is not None:
                    eeg_mean = self.normalization_stats.get('eeg', {}).get('mean', None)
                    eeg_std = self.normalization_stats.get('eeg', {}).get('std', None)
                    
                    if eeg_mean is not None and eeg_std is not None and len(eeg_mean.shape) == 1 and eeg_mean.shape[0] == num_channels:
                        # 使用全局统计量：扩展到 (1, 1, num_channels, 1) 以便广播
                        mean_expanded = eeg_mean.reshape(1, 1, num_channels, 1)
                        std_expanded = eeg_std.reshape(1, 1, num_channels, 1)
                        # 归一化整个数组
                        processed_eeg = (eeg_data - mean_expanded) / (std_expanded + 1e-8)
                    else:
                        # 回退到样本内归一化（对每个窗口的每个patch进行归一化）
                        # 优化：使用view进行reshape，避免复制
                        reshaped = eeg_data.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        stds = np.where(stds < 1e-8, 1.0, stds)
                        processed_eeg = (reshaped - means) / (stds + 1e-8)
                        processed_eeg = processed_eeg.reshape(num_windows, num_patches, num_channels, patch_length)
                else:
                    # 样本内归一化（优化：避免不必要的复制，使用更高效的计算）
                    # 使用view进行reshape，避免复制
                    reshaped = eeg_data.reshape(-1, patch_length)
                    # 计算mean和std（axis=1表示对每个patch的时间维度计算）
                    means = reshaped.mean(axis=1, keepdims=True)
                    stds = reshaped.std(axis=1, keepdims=True)
                    # 避免除零
                    stds = np.where(stds < 1e-8, 1.0, stds)
                    # 归一化（这里需要复制，因为要修改数据）
                    processed_eeg = (reshaped - means) / (stds + 1e-8)
                    processed_eeg = processed_eeg.reshape(num_windows, num_patches, num_channels, patch_length)
            else:
                # 不需要归一化，直接返回（不复制，节省内存）
                processed_eeg = eeg_data
            return processed_eeg
        
        # 列表格式：逐个处理（向后兼容）
        processed_eeg = []
        num_windows = len(eeg_data)
        
        # 使用 tqdm 显示进度
        from tqdm import tqdm
        iterator = tqdm(eeg_data, desc="预处理 EEG", unit="窗口", disable=num_windows < 10)
        
        for window_eeg in iterator:        
            if self.normalize:
                window_eeg = window_eeg.copy()
                num_patches, num_channels, patch_length = window_eeg.shape
                
                if self.normalization_stats is not None:
                    # 使用全局统计量（训练集的mean和std）
                    # normalization_stats: {'eeg': {'mean': (num_channels,), 'std': (num_channels,)}}
                    eeg_mean = self.normalization_stats.get('eeg', {}).get('mean', None)
                    eeg_std = self.normalization_stats.get('eeg', {}).get('std', None)
                    
                    if eeg_mean is not None and eeg_std is not None:
                        # 使用全局统计量：对每个通道使用相同的mean和std
                        # eeg_mean: (num_channels,), eeg_std: (num_channels,)
                        if len(eeg_mean.shape) == 1 and eeg_mean.shape[0] == num_channels:
                            # 扩展到 (num_patches, num_channels, 1)
                            mean_expanded = eeg_mean.reshape(1, num_channels, 1)
                            std_expanded = eeg_std.reshape(1, num_channels, 1)
                            # 归一化: (x - mean) / std
                            window_eeg = (window_eeg - mean_expanded) / (std_expanded + 1e-8)
                        else:
                            # 如果形状不匹配，回退到样本内归一化
                            reshaped = window_eeg.reshape(-1, patch_length)
                            means = reshaped.mean(axis=1, keepdims=True)
                            stds = reshaped.std(axis=1, keepdims=True)
                            reshaped = (reshaped - means) / (stds + 1e-8)
                            window_eeg = reshaped.reshape(num_patches, num_channels, patch_length)
                    else:
                        # 如果没有提供统计量，使用样本内归一化（向后兼容）
                        reshaped = window_eeg.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        reshaped = (reshaped - means) / (stds + 1e-8)
                        window_eeg = reshaped.reshape(num_patches, num_channels, patch_length)
                else:
                    # 没有提供全局统计量，使用样本内归一化（向后兼容）
                    reshaped = window_eeg.reshape(-1, patch_length)
                    means = reshaped.mean(axis=1, keepdims=True)
                    stds = reshaped.std(axis=1, keepdims=True)
                    reshaped = (reshaped - means) / (stds + 1e-8)
                    window_eeg = reshaped.reshape(num_patches, num_channels, patch_length)
            # 如果 normalize=False，直接使用原始数据（不进行归一化）
            processed_eeg.append(window_eeg)
         
        return processed_eeg
   
    
    def normalize_signal(self, signal_data):
        """
        Normalize signal using z-score normalization
        """
        if len(signal_data.shape) == 1:
            # Single channel
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            if std_val > 0:
                signal_data = (signal_data - mean_val) / std_val
        else:
            # Multiple channels
            for ch in range(signal_data.shape[0]):
                mean_val = np.mean(signal_data[ch])
                std_val = np.std(signal_data[ch])
                if std_val > 0:
                    signal_data[ch] = (signal_data[ch] - mean_val) / std_val
        
        return signal_data
    
    def create_segments(self):
        """
        Create overlapping segments from the signals
        """
        segments = []
        
        # Get the minimum length across all subjects
        min_length = float('inf')
        
        if self.eeg_data is not None:
            min_length = min(min_length, min([eeg.shape[1] for eeg in self.eeg_data]))
        
        # 如果没有找到有效数据，返回空列表
        if min_length == float('inf') or min_length < self.window_length:
            return segments
        
        # Create segments
        # 确保所有参数都是整数
        min_length = int(min_length)
        window_length = int(self.window_length)
        hop_length = int(self.hop_length)
        for start in range(0, min_length - window_length + 1, hop_length):
            end = start + window_length
            segments.append((start, end))
        
        return segments
    
    def __len__(self):
        """Return the number of samples"""
        # 如果有索引映射，使用索引长度
        if self.indices is not None:
            return len(self.indices)
        
        return len(self.data_dict.get('labels', []))
    
    def __getitem__(self, idx):
        """
        Get a sample of data
        """
        sample = {}
        # 如果有索引映射，使用映射后的索引
        if self.indices is not None:
            window_idx = int(self.indices[idx])
        else:
            window_idx = idx
        
        # 获取该window的subject_id（用于per-subject归一化）
        subject_id = None
        if self.subject_ids is not None:
            if isinstance(self.subject_ids, np.ndarray):
                subject_id = int(self.subject_ids[window_idx])
            elif isinstance(self.subject_ids, list):
                subject_id = int(self.subject_ids[window_idx])
        
        if 'eeg' in self.modality and self.eeg_data is not None:
            # 直接从原始数据获取，不复制
            window_eeg = self.eeg_data[window_idx]  # (num_patches, channels, patch_samples)
            # 在线归一化（只对这个window处理）
            if self.normalize:
                window_eeg = self._normalize_window(window_eeg, 'eeg', subject_id=subject_id)
            sample['eeg'] = torch.FloatTensor(window_eeg)
        
        # Add label
        if self.labels is not None:
            label = int(self.labels[window_idx])
            sample['label'] = torch.LongTensor([label])
        
        # Add weight (if available)
        if self.weights is not None:
            weight = float(self.weights[window_idx])
            sample['weight'] = torch.FloatTensor([weight])
        
        # Add subject_id (if available)
        if subject_id is not None:
            sample['subject_id'] = torch.LongTensor([subject_id])
        
        return sample

class PhysiologicalDataset(Dataset):
    """
    Dataset class for physiological signals (EEG and ECG)
    支持两种数据格式：
    1. 新格式：patches格式 (num_windows, num_patches, channels, patch_samples)
    2. 旧格式：连续信号格式 (channels, time_points)
    """
    
    def __init__(self, data_dict, modality, window_length=3000, 
                 hop_length=1500, normalize=True, apply_filter=True,
                 eeg_sampling_rate=1000, ecg_sampling_rate=500,
                 normalization_stats=None):
        super().__init__()
        
        self.data_dict = data_dict
        self.modality = modality
        self.window_length = window_length
        self.hop_length = hop_length
        self.normalize = normalize
        self.apply_filter = apply_filter
        self.eeg_sampling_rate = eeg_sampling_rate
        self.ecg_sampling_rate = ecg_sampling_rate
        # 归一化统计量（如果提供，则使用全局统计量；否则使用样本内统计量）
        self.normalization_stats = normalization_stats
        
        # Extract data
        # 检查是否有索引映射（用于数据子集，避免复制大数据）
        self.indices = data_dict.get('_indices', None)
        
        self.eeg_data = data_dict.get('eeg', None)
        self.ecg_data = data_dict.get('ecg', None)
        self.labels = data_dict.get('labels', None)
        self.weights = data_dict.get('weights', None)  # 样本权重
        self.subject_ids = data_dict.get('subject_ids', None)  # 保存subject_ids用于per-subject归一化
        
        # 检测数据格式：新格式（patches）还是旧格式（连续信号）
        # 只要 EEG 或 ECG 任一为 patches 格式即视为新格式（避免仅 ECG 时误判导致 segments 为空除零）
        self.is_patches_format = False
        self.is_numpy_array = False  # 标记数据是否为numpy数组格式
        if self.eeg_data is not None:
            if isinstance(self.eeg_data, np.ndarray):
                self.is_numpy_array = True
                if len(self.eeg_data.shape) == 4:
                    self.is_patches_format = True
            elif len(self.eeg_data) > 0:
                first_eeg = self.eeg_data[0]
                if isinstance(first_eeg, np.ndarray) and len(first_eeg.shape) == 3:
                    self.is_patches_format = True
        if not self.is_patches_format and self.ecg_data is not None:
            if isinstance(self.ecg_data, np.ndarray) and len(self.ecg_data.shape) == 4:
                self.is_patches_format = True
            elif isinstance(self.ecg_data, list) and len(self.ecg_data) > 0:
                first_ecg = self.ecg_data[0]
                if isinstance(first_ecg, np.ndarray) and len(first_ecg.shape) == 3:
                    self.is_patches_format = True
        
        # 不在这里预处理！只保存归一化统计量，在 __getitem__ 中按需处理
        # 这样可以避免在初始化时处理整个数据集，节省内存
        self.processed_data = None  # 不再预处理整个数据集
        
        # Create segments (仅对旧格式)
        if not self.is_patches_format:
            self.segments = self.create_segments()
        else:
            self.segments = None
        
    def _normalize_window(self, window_data, modality, subject_id=None):
        """
        对单个window进行归一化（在线处理）
        window_data: (num_patches, channels, patch_samples)
        subject_id: 该window所属的subject_id，用于per-subject归一化
        """
        num_patches, num_channels, patch_length = window_data.shape
        
        if self.normalization_stats is not None:
            # 检查是否是per-subject归一化格式: {subject_id: {modality: {mean, std}}}
            # 还是旧的全局归一化格式: {modality: {mean, std}}
            first_key = next(iter(self.normalization_stats.keys()))
            
            if isinstance(first_key, (int, np.integer)) or (isinstance(first_key, str) and first_key.isdigit()):
                # Per-subject归一化格式
                if subject_id is not None:
                    subject_id_key = int(subject_id) if isinstance(subject_id, (int, np.integer)) else subject_id
                    subject_stats = self.normalization_stats.get(subject_id_key, None)
                    
                    if subject_stats is not None:
                        stats = subject_stats.get(modality, {})
                        mean = stats.get('mean', None)
                        std = stats.get('std', None)
                        
                        if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                            # 使用该subject的统计量：扩展到 (1, num_channels, 1) 以便广播
                            mean_expanded = mean.reshape(1, num_channels, 1)
                            std_expanded = std.reshape(1, num_channels, 1)
                            std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                            return (window_data - mean_expanded) / (std_expanded + 1e-8)
            else:
                # 旧的全局归一化格式（向后兼容）
                stats = self.normalization_stats.get(modality, {})
                mean = stats.get('mean', None)
                std = stats.get('std', None)
                
                if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                    # 使用全局统计量：扩展到 (1, num_channels, 1) 以便广播
                    mean_expanded = mean.reshape(1, num_channels, 1)
                    std_expanded = std.reshape(1, num_channels, 1)
                    std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                    return (window_data - mean_expanded) / (std_expanded + 1e-8)
        
        # 回退到样本内归一化（对这个window的每个patch进行归一化）
        reshaped = window_data.reshape(-1, patch_length)
        means = reshaped.mean(axis=1, keepdims=True)
        stds = reshaped.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        normalized = (reshaped - means) / (stds + 1e-8)
        return normalized.reshape(num_patches, num_channels, patch_length)
    
    def preprocess_eeg(self, eeg_data):
        """
        Preprocess EEG signals
        支持两种格式：
        1. 新格式：patches (num_patches, channels, patch_samples) 
           - 对于已分片的数据，只进行可选的归一化（如果 normalize=True）
           - 归一化不是必须的！如果数据已经在预处理阶段归一化过，可以设置 normalize=False
           - 归一化的作用：
             * 将每个patch的每个通道标准化到均值0、标准差1
             * 有助于神经网络训练稳定性和收敛速度
             * 但如果数据范围已经合理（如已在[-1,1]或[0,1]范围），可以跳过
        2. 旧格式：连续信号 (channels, time_points) - 进行滤波和归一化
        """
        # 如果是numpy数组格式，直接处理整个数组（更快）
        if isinstance(eeg_data, np.ndarray) and self.is_patches_format:
            if self.normalize:
                # 对整个数组进行归一化
                num_windows, num_patches, num_channels, patch_length = eeg_data.shape
                
                if self.normalization_stats is not None:
                    eeg_mean = self.normalization_stats.get('eeg', {}).get('mean', None)
                    eeg_std = self.normalization_stats.get('eeg', {}).get('std', None)
                    
                    if eeg_mean is not None and eeg_std is not None and len(eeg_mean.shape) == 1 and eeg_mean.shape[0] == num_channels:
                        # 使用全局统计量：扩展到 (1, 1, num_channels, 1) 以便广播
                        mean_expanded = eeg_mean.reshape(1, 1, num_channels, 1)
                        std_expanded = eeg_std.reshape(1, 1, num_channels, 1)
                        # 归一化整个数组
                        processed_eeg = (eeg_data - mean_expanded) / (std_expanded + 1e-8)
                    else:
                        # 回退到样本内归一化（对每个窗口的每个patch进行归一化）
                        # 优化：使用view进行reshape，避免复制
                        reshaped = eeg_data.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        stds = np.where(stds < 1e-8, 1.0, stds)
                        processed_eeg = (reshaped - means) / (stds + 1e-8)
                        processed_eeg = processed_eeg.reshape(num_windows, num_patches, num_channels, patch_length)
                else:
                    # 样本内归一化（优化：避免不必要的复制，使用更高效的计算）
                    # 使用view进行reshape，避免复制
                    reshaped = eeg_data.reshape(-1, patch_length)
                    # 计算mean和std（axis=1表示对每个patch的时间维度计算）
                    means = reshaped.mean(axis=1, keepdims=True)
                    stds = reshaped.std(axis=1, keepdims=True)
                    # 避免除零
                    stds = np.where(stds < 1e-8, 1.0, stds)
                    # 归一化（这里需要复制，因为要修改数据）
                    processed_eeg = (reshaped - means) / (stds + 1e-8)
                    processed_eeg = processed_eeg.reshape(num_windows, num_patches, num_channels, patch_length)
            else:
                # 不需要归一化，直接返回（不复制，节省内存）
                processed_eeg = eeg_data
            return processed_eeg
        
        # 列表格式：逐个处理（向后兼容）
        processed_eeg = []
        num_windows = len(eeg_data)
        
        # 使用 tqdm 显示进度
        from tqdm import tqdm
        iterator = tqdm(eeg_data, desc="预处理 EEG", unit="窗口", disable=num_windows < 10)
        
        for window_eeg in iterator:
            if self.is_patches_format:
                # 新格式：patches格式数据
                # window_eeg: (num_patches, channels, patch_samples)
                if self.normalize:
                    window_eeg = window_eeg.copy()
                    num_patches, num_channels, patch_length = window_eeg.shape
                    
                    if self.normalization_stats is not None:
                        # 使用全局统计量（训练集的mean和std）
                        # normalization_stats: {'eeg': {'mean': (num_channels,), 'std': (num_channels,)}}
                        eeg_mean = self.normalization_stats.get('eeg', {}).get('mean', None)
                        eeg_std = self.normalization_stats.get('eeg', {}).get('std', None)
                        
                        if eeg_mean is not None and eeg_std is not None:
                            # 使用全局统计量：对每个通道使用相同的mean和std
                            # eeg_mean: (num_channels,), eeg_std: (num_channels,)
                            if len(eeg_mean.shape) == 1 and eeg_mean.shape[0] == num_channels:
                                # 扩展到 (num_patches, num_channels, 1)
                                mean_expanded = eeg_mean.reshape(1, num_channels, 1)
                                std_expanded = eeg_std.reshape(1, num_channels, 1)
                                # 归一化: (x - mean) / std
                                window_eeg = (window_eeg - mean_expanded) / (std_expanded + 1e-8)
                            else:
                                # 如果形状不匹配，回退到样本内归一化
                                reshaped = window_eeg.reshape(-1, patch_length)
                                means = reshaped.mean(axis=1, keepdims=True)
                                stds = reshaped.std(axis=1, keepdims=True)
                                reshaped = (reshaped - means) / (stds + 1e-8)
                                window_eeg = reshaped.reshape(num_patches, num_channels, patch_length)
                        else:
                            # 如果没有提供统计量，使用样本内归一化（向后兼容）
                            reshaped = window_eeg.reshape(-1, patch_length)
                            means = reshaped.mean(axis=1, keepdims=True)
                            stds = reshaped.std(axis=1, keepdims=True)
                            reshaped = (reshaped - means) / (stds + 1e-8)
                            window_eeg = reshaped.reshape(num_patches, num_channels, patch_length)
                    else:
                        # 没有提供全局统计量，使用样本内归一化（向后兼容）
                        reshaped = window_eeg.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        reshaped = (reshaped - means) / (stds + 1e-8)
                        window_eeg = reshaped.reshape(num_patches, num_channels, patch_length)
                # 如果 normalize=False，直接使用原始数据（不进行归一化）
                processed_eeg.append(window_eeg)
            else:
                # 旧格式：连续信号，进行完整预处理
                # Apply bandpass filter (0.5-50 Hz)
                if self.apply_filter:
                    window_eeg = self.apply_eeg_filter(window_eeg)
                
                # Remove artifacts (simple approach)
                window_eeg = self.remove_eeg_artifacts(window_eeg)
                
                # Normalize
                if self.normalize:
                    window_eeg = self.normalize_signal(window_eeg)
                
                processed_eeg.append(window_eeg)
        
        return processed_eeg
    
    def preprocess_ecg(self, ecg_data):
        """
        Preprocess ECG signals
        支持两种格式：
        1. 新格式：patches (num_patches, channels, patch_samples)
           - 对于已分片的数据，只进行可选的归一化（如果 normalize=True）
           - 归一化不是必须的！如果数据已经在预处理阶段归一化过，可以设置 normalize=False
        2. 旧格式：连续信号 (channels, time_points) - 进行滤波和归一化
        """
        # 如果是numpy数组格式，直接处理整个数组（更快）
        if isinstance(ecg_data, np.ndarray) and self.is_patches_format:
            if self.normalize:
                # 对整个数组进行归一化
                num_windows, num_patches, num_channels, patch_length = ecg_data.shape
                
                if self.normalization_stats is not None:
                    ecg_mean = self.normalization_stats.get('ecg', {}).get('mean', None)
                    ecg_std = self.normalization_stats.get('ecg', {}).get('std', None)
                    
                    if ecg_mean is not None and ecg_std is not None and len(ecg_mean.shape) == 1 and ecg_mean.shape[0] == num_channels:
                        # 使用全局统计量：扩展到 (1, 1, num_channels, 1) 以便广播
                        mean_expanded = ecg_mean.reshape(1, 1, num_channels, 1)
                        std_expanded = ecg_std.reshape(1, 1, num_channels, 1)
                        # 归一化整个数组
                        processed_ecg = (ecg_data - mean_expanded) / (std_expanded + 1e-8)
                    else:
                        # 回退到样本内归一化（优化：避免不必要的复制）
                        reshaped = ecg_data.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        stds = np.where(stds < 1e-8, 1.0, stds)
                        processed_ecg = (reshaped - means) / (stds + 1e-8)
                        processed_ecg = processed_ecg.reshape(num_windows, num_patches, num_channels, patch_length)
                else:
                    # 样本内归一化（优化：避免不必要的复制）
                    reshaped = ecg_data.reshape(-1, patch_length)
                    means = reshaped.mean(axis=1, keepdims=True)
                    stds = reshaped.std(axis=1, keepdims=True)
                    stds = np.where(stds < 1e-8, 1.0, stds)
                    processed_ecg = (reshaped - means) / (stds + 1e-8)
                    processed_ecg = processed_ecg.reshape(num_windows, num_patches, num_channels, patch_length)
            else:
                # 不需要归一化，直接返回（不复制，节省内存）
                processed_ecg = ecg_data
            return processed_ecg
        
        # 列表格式：逐个处理（向后兼容）
        processed_ecg = []
        num_windows = len(ecg_data)
        
        # 使用 tqdm 显示进度（仅在数据量大时）
        use_progress = num_windows > 100
        if use_progress:
            from tqdm import tqdm
            iterator = tqdm(ecg_data, desc="预处理 ECG", unit="窗口")
        else:
            iterator = ecg_data
        
        for window_ecg in iterator:
            if self.is_patches_format:
                # 新格式：patches格式数据
                # window_ecg: (num_patches, channels, patch_samples)
                if self.normalize:
                    window_ecg = window_ecg.copy()
                    num_patches, num_channels, patch_length = window_ecg.shape
                    
                    if self.normalization_stats is not None:
                        # 使用全局统计量（训练集的mean和std）
                        ecg_mean = self.normalization_stats.get('ecg', {}).get('mean', None)
                        ecg_std = self.normalization_stats.get('ecg', {}).get('std', None)
                        
                        if ecg_mean is not None and ecg_std is not None:
                            # 使用全局统计量：对每个通道使用相同的mean和std
                            if len(ecg_mean.shape) == 1 and ecg_mean.shape[0] == num_channels:
                                mean_expanded = ecg_mean.reshape(1, num_channels, 1)
                                std_expanded = ecg_std.reshape(1, num_channels, 1)
                                window_ecg = (window_ecg - mean_expanded) / (std_expanded + 1e-8)
                            else:
                                # 形状不匹配，回退到样本内归一化
                                reshaped = window_ecg.reshape(-1, patch_length)
                                means = reshaped.mean(axis=1, keepdims=True)
                                stds = reshaped.std(axis=1, keepdims=True)
                                reshaped = (reshaped - means) / (stds + 1e-8)
                                window_ecg = reshaped.reshape(num_patches, num_channels, patch_length)
                        else:
                            # 如果没有提供统计量，使用样本内归一化
                            reshaped = window_ecg.reshape(-1, patch_length)
                            means = reshaped.mean(axis=1, keepdims=True)
                            stds = reshaped.std(axis=1, keepdims=True)
                            reshaped = (reshaped - means) / (stds + 1e-8)
                            window_ecg = reshaped.reshape(num_patches, num_channels, patch_length)
                    else:
                        # 没有提供全局统计量，使用样本内归一化（向后兼容）
                        reshaped = window_ecg.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        reshaped = (reshaped - means) / (stds + 1e-8)
                        window_ecg = reshaped.reshape(num_patches, num_channels, patch_length)
                # 如果 normalize=False，直接使用原始数据（不进行归一化）
                processed_ecg.append(window_ecg)
            else:
                # 旧格式：连续信号，进行完整预处理
                # Apply bandpass filter (0.5-40 Hz)
                if self.apply_filter:
                    window_ecg = self.apply_ecg_filter(window_ecg)
                
                # Normalize
                if self.normalize:
                    window_ecg = self.normalize_signal(window_ecg)
                
                processed_ecg.append(window_ecg)
        
        return processed_ecg
    
    def apply_eeg_filter(self, eeg_signal):
        """
        Apply bandpass filter to EEG signal
        """
        # Design bandpass filter
        nyquist = self.eeg_sampling_rate / 2
        low = 0.5 / nyquist
        high = 50.0 / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_signal = signal.filtfilt(b, a, eeg_signal, axis=1)
        
        return filtered_signal
    
    def apply_ecg_filter(self, ecg_signal):
        """
        Apply bandpass filter to ECG signal
        """
        # Design bandpass filter
        nyquist = self.ecg_sampling_rate / 2
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_signal = signal.filtfilt(b, a, ecg_signal, axis=1)
        
        return filtered_signal
    
    def remove_eeg_artifacts(self, eeg_signal):
        """
        Remove EEG artifacts using simple thresholding
        """
        # Simple artifact removal using thresholding
        threshold = 3 * np.std(eeg_signal)
        eeg_signal = np.clip(eeg_signal, -threshold, threshold)
        
        return eeg_signal
    
    def normalize_signal(self, signal_data):
        """
        Normalize signal using z-score normalization
        """
        if len(signal_data.shape) == 1:
            # Single channel
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            if std_val > 0:
                signal_data = (signal_data - mean_val) / std_val
        else:
            # Multiple channels
            for ch in range(signal_data.shape[0]):
                mean_val = np.mean(signal_data[ch])
                std_val = np.std(signal_data[ch])
                if std_val > 0:
                    signal_data[ch] = (signal_data[ch] - mean_val) / std_val
        
        return signal_data
    
    def create_segments(self):
        """
        Create overlapping segments from the signals
        """
        segments = []
        
        # Get the minimum length across all subjects
        min_length = float('inf')
        
        if self.eeg_data is not None:
            min_length = min(min_length, min([eeg.shape[1] for eeg in self.eeg_data]))
        
        if self.ecg_data is not None:
            min_length = min(min_length, min([ecg.shape[1] for ecg in self.ecg_data]))
        
        # 如果没有找到有效数据，返回空列表
        if min_length == float('inf') or min_length < self.window_length:
            return segments
        
        # Create segments
        # 确保所有参数都是整数
        min_length = int(min_length)
        window_length = int(self.window_length)
        hop_length = int(self.hop_length)
        for start in range(0, min_length - window_length + 1, hop_length):
            end = start + window_length
            segments.append((start, end))
        
        return segments
    
    def __len__(self):
        """Return the number of samples"""
        # 如果有索引映射，使用索引长度
        if self.indices is not None:
            return len(self.indices)
        
        if self.is_patches_format:
            # 新格式：每个窗口是一个样本
            return len(self.data_dict.get('labels', []))
        else:
            # 旧格式：每个subject的每个segment是一个样本
            return len(self.segments) * len(self.data_dict.get('labels', [0]))
    
    def __getitem__(self, idx):
        """
        Get a sample of data
        """
        sample = {}
        
        if self.is_patches_format:
            # 新格式：直接使用窗口索引，返回patches格式
            # 如果有索引映射，使用映射后的索引
            if self.indices is not None:
                window_idx = int(self.indices[idx])
            else:
                window_idx = idx
            
            # 获取该window的subject_id（用于per-subject归一化）
            subject_id = None
            if self.subject_ids is not None:
                if isinstance(self.subject_ids, np.ndarray):
                    subject_id = int(self.subject_ids[window_idx])
                elif isinstance(self.subject_ids, list):
                    subject_id = int(self.subject_ids[window_idx])
            
            if 'eeg' in self.modality and self.eeg_data is not None:
                # 直接从原始数据获取，不复制
                window_eeg = self.eeg_data[window_idx]  # (num_patches, channels, patch_samples)
                # 在线归一化（只对这个window处理）
                if self.normalize:
                    window_eeg = self._normalize_window(window_eeg, 'eeg', subject_id=subject_id)
                sample['eeg'] = torch.FloatTensor(window_eeg)
            
            if 'ecg' in self.modality and self.ecg_data is not None:
                # 直接从原始数据获取，不复制
                window_ecg = self.ecg_data[window_idx]  # (num_patches, channels, patch_samples)
                # 在线归一化（只对这个window处理）
                if self.normalize:
                    window_ecg = self._normalize_window(window_ecg, 'ecg', subject_id=subject_id)
                sample['ecg'] = torch.FloatTensor(window_ecg)
            
            # Add label
            if self.labels is not None:
                label = int(self.labels[window_idx])
                sample['label'] = torch.LongTensor([label])
            
            # Add weight (if available)
            if self.weights is not None:
                weight = float(self.weights[window_idx])
                sample['weight'] = torch.FloatTensor([weight])
            
            # Add subject_id (if available)
            if subject_id is not None:
                sample['subject_id'] = torch.LongTensor([subject_id])
        else:
            # 旧格式：计算subject和segment索引
            if self.segments is None or len(self.segments) == 0:
                raise ValueError(
                    "数据被识别为旧格式但 segments 为空（无法从当前数据生成滑动窗口）。"
                    "请确认数据集为新格式（eeg_patches.npy/ecg_patches.npy 等）且 EEG 或 ECG 至少有一种被正确加载。"
                )
            num_subjects = len(self.data_dict.get('labels', [0]))
            subject_idx = idx // len(self.segments)
            segment_idx = idx % len(self.segments)
            
            start, end = self.segments[segment_idx]
            
            if 'eeg' in self.modality and self.eeg_data is not None:
                window_eeg = self.eeg_data[subject_idx][:, start:end]
                if self.normalize:
                    window_eeg = self.normalize_signal(window_eeg)
                sample['eeg'] = torch.FloatTensor(window_eeg)
            
            if 'ecg' in self.modality and self.ecg_data is not None:
                window_ecg = self.ecg_data[subject_idx][:, start:end]
                if self.normalize:
                    window_ecg = self.normalize_signal(window_ecg)
                sample['ecg'] = torch.FloatTensor(window_ecg)
            
            # Add label
            if self.labels is not None:
                sample['label'] = torch.LongTensor([self.labels[subject_idx]])
            
            # Add weight (if available, 旧格式可能没有权重)
            if self.weights is not None:
                weight = float(self.weights[subject_idx])
                sample['weight'] = torch.FloatTensor([weight])
            
            # Add subject_id (if available)
            if self.subject_ids is not None:
                if isinstance(self.subject_ids, np.ndarray):
                    subject_id = int(self.subject_ids[subject_idx])
                elif isinstance(self.subject_ids, list):
                    subject_id = int(self.subject_ids[subject_idx])
                else:
                    subject_id = int(subject_idx)  # 如果没有subject_ids，使用subject_idx作为subject_id
                sample['subject_id'] = torch.LongTensor([subject_id])
        
        return sample


class DataArranger:
    """
    Data arrangement and preprocessing utility
    """
    
    def __init__(self, dataset_info, dataset_path, debug=0):
        """
        Args:
            dataset_info: 数据集信息
            dataset_path: 数据集路径
            debug: 调试模式（限制数据量）
        """
        self.dataset_info = dataset_info
        self.dataset_path = dataset_path
        self.debug = debug
        self._cached_data = None  # 缓存已加载的数据，避免重复加载
        
    def load_labels_only(self):
        """
        只加载labels，用于计算类别权重等操作，避免加载大量数据。
        返回已映射为5分类的标签（0-9 中 4 分及以上映射为 4）。
        """
        labels_path = os.path.join(self.dataset_path, 'labels.npy')
        if os.path.exists(labels_path):
            labels = np.load(labels_path)
            return map_labels_10_to_5(labels)
        else:
            # 旧格式：从CSV加载
            labels_path = os.path.join(self.dataset_path, 'raw', 'labels', 'motion_sickness_scores.csv')
            if os.path.exists(labels_path):
                df = pd.read_csv(labels_path)
                if 'score' in df.columns:
                    labels = df['score'].values
                elif 'motion_sickness_score' in df.columns:
                    labels = df['motion_sickness_score'].values
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        labels = df[numeric_cols[0]].values
                    else:
                        raise ValueError("No numeric label column found")
                return map_labels_10_to_5(labels)
            raise FileNotFoundError(f"无法找到labels文件: {labels_path}")
    
    def load_data(self, mode='physio', use_cache=True):
        """
        Load data from files
        支持两种数据格式：
        1. 新格式：从training_dataset目录加载patches格式数据
        2. 旧格式：从data/raw目录加载
        
        Args:
            mode: 'physio', 'imu', 'mix', 'simplemix', 'newmix' 或 'allmix'
            use_cache: 是否使用缓存（默认True，避免重复加载）
        """
        # 检查缓存
        if use_cache and self._cached_data is not None:
            print("  使用缓存的数据（避免重复加载）...")
            return self._cached_data
        
        data = {}
        
        if mode == 'mix':
            # Mix模式：同时加载IMU和physio数据
            imu_patches_path = os.path.join(self.dataset_path, 'imu_patches.npy')
            eeg_patches_path = os.path.join(self.dataset_path, 'eeg_patches.npy')
            ecg_patches_path = os.path.join(self.dataset_path, 'ecg_patches.npy')
            labels_path = os.path.join(self.dataset_path, 'labels.npy')
            
            if (os.path.exists(imu_patches_path) and 
                os.path.exists(eeg_patches_path) and 
                os.path.exists(ecg_patches_path) and 
                os.path.exists(labels_path)):
                # 新格式：从training_dataset目录加载
                import time
                from tqdm import tqdm
                
                print(f"检测到Mix模式新格式数据，从 {self.dataset_path} 加载...")
                
                # 加载numpy文件（使用mmap模式，不加载到内存）
                print("  正在加载numpy文件（使用mmap模式，不加载到内存）...")
                start_time = time.time()
                imu_patches = np.load(imu_patches_path, mmap_mode='r')
                print(f"    ✓ IMU数据加载完成 ({time.time()-start_time:.2f}秒, 形状: {imu_patches.shape}, mmap模式)")
                
                start_time = time.time()
                eeg_patches = np.load(eeg_patches_path, mmap_mode='r')
                print(f"    ✓ EEG数据加载完成 ({time.time()-start_time:.2f}秒, 形状: {eeg_patches.shape}, mmap模式)")
                
                start_time = time.time()
                ecg_patches = np.load(ecg_patches_path, mmap_mode='r')
                print(f"    ✓ ECG数据加载完成 ({time.time()-start_time:.2f}秒, 形状: {ecg_patches.shape}, mmap模式)")
                
                start_time = time.time()
                labels = map_labels_10_to_5(np.load(labels_path))  # labels很小，直接加载并映射为5分类
                print(f"    ✓ Labels加载完成 ({time.time()-start_time:.2f}秒, 形状: {labels.shape})")
                
                # 加载subject_ids（如果存在）
                subject_ids_path = os.path.join(self.dataset_path, 'subject_ids.npy')
                if os.path.exists(subject_ids_path):
                    start_time = time.time()
                    subject_ids = np.load(subject_ids_path)  # subject_ids很小，直接加载
                    print(f"    ✓ Subject IDs加载完成 ({time.time()-start_time:.2f}秒, 形状: {subject_ids.shape})")
                    data['subject_ids'] = subject_ids
                else:
                    print(f"    ⚠️  未找到subject_ids.npy，将使用随机划分")
                
                # 直接使用numpy数组，避免转换为列表（大幅提升性能）
                print("  正在准备数据...")
                num_windows = imu_patches.shape[0]
                start_time = time.time()
                
                # 直接使用numpy数组，不需要转换为列表和复制
                # Dataset类可以直接索引numpy数组
                data['imu'] = imu_patches
                data['eeg'] = eeg_patches
                data['ecg'] = ecg_patches
                data['labels'] = labels
                
                elapsed = time.time() - start_time
                print(f"    ✓ 数据准备完成 ({elapsed:.2f}秒)")
                
                # 加载权重（如果存在）
                weights_path = os.path.join(self.dataset_path, 'weights.npy')
                if os.path.exists(weights_path):
                    data['weights'] = np.load(weights_path)
                
                print(f"\n✓ Mix模式数据加载完成: {num_windows} 个窗口")
                print(f"  IMU patches形状: {imu_patches.shape}")
                print(f"  EEG patches形状: {eeg_patches.shape}")
                print(f"  ECG patches形状: {ecg_patches.shape}")
                print(f"  Labels形状: {labels.shape}")
                print(f"{'='*60}\n")
                
                # 缓存数据
                if use_cache:
                    self._cached_data = data
                return data
            else:
                raise FileNotFoundError(f"Mix模式需要同时存在以下文件: {imu_patches_path}, {eeg_patches_path}, {ecg_patches_path}, {labels_path}")
        
        if mode == 'simplemix':
            # SimpleMix模式：同时加载IMU和ECG数据（不加载EEG）
            imu_patches_path = os.path.join(self.dataset_path, 'imu_patches.npy')
            ecg_patches_path = os.path.join(self.dataset_path, 'ecg_patches.npy')
            labels_path = os.path.join(self.dataset_path, 'labels.npy')
            
            if (os.path.exists(imu_patches_path) and 
                os.path.exists(ecg_patches_path) and 
                os.path.exists(labels_path)):
                # 新格式：从training_dataset目录加载
                import time
                from tqdm import tqdm
                
                print(f"检测到SimpleMix模式新格式数据，从 {self.dataset_path} 加载...")
                
                # 加载numpy文件（使用mmap模式，不加载到内存）
                print("  正在加载numpy文件（使用mmap模式，不加载到内存）...")
                start_time = time.time()
                imu_patches = np.load(imu_patches_path, mmap_mode='r')
                print(f"    ✓ IMU数据加载完成 ({time.time()-start_time:.2f}秒, 形状: {imu_patches.shape}, mmap模式)")
                
                start_time = time.time()
                ecg_patches = np.load(ecg_patches_path, mmap_mode='r')
                print(f"    ✓ ECG数据加载完成 ({time.time()-start_time:.2f}秒, 形状: {ecg_patches.shape}, mmap模式)")
                
                start_time = time.time()
                labels = map_labels_10_to_5(np.load(labels_path))  # labels很小，直接加载并映射为5分类
                print(f"    ✓ Labels加载完成 ({time.time()-start_time:.2f}秒, 形状: {labels.shape})")
                
                # 加载subject_ids（如果存在）
                subject_ids_path = os.path.join(self.dataset_path, 'subject_ids.npy')
                if os.path.exists(subject_ids_path):
                    start_time = time.time()
                    subject_ids = np.load(subject_ids_path)  # subject_ids很小，直接加载
                    print(f"    ✓ Subject IDs加载完成 ({time.time()-start_time:.2f}秒, 形状: {subject_ids.shape})")
                    data['subject_ids'] = subject_ids
                else:
                    print(f"    ⚠️  未找到subject_ids.npy，将使用随机划分")
                
                # 直接使用numpy数组，避免转换为列表（大幅提升性能）
                print("  正在准备数据...")
                num_windows = imu_patches.shape[0]
                start_time = time.time()
                
                # 直接使用numpy数组，不需要转换为列表和复制
                # Dataset类可以直接索引numpy数组
                data['imu'] = imu_patches
                data['ecg'] = ecg_patches
                data['labels'] = labels
                
                elapsed = time.time() - start_time
                print(f"    ✓ 数据准备完成 ({elapsed:.2f}秒)")
                
                # 加载权重（如果存在）
                weights_path = os.path.join(self.dataset_path, 'weights.npy')
                if os.path.exists(weights_path):
                    data['weights'] = np.load(weights_path)
                
                print(f"\n✓ SimpleMix模式数据加载完成: {num_windows} 个窗口")
                print(f"  IMU patches形状: {imu_patches.shape}")
                print(f"  ECG patches形状: {ecg_patches.shape}")
                print(f"  Labels形状: {labels.shape}")
                print(f"{'='*60}\n")
                
                # 缓存数据
                if use_cache:
                    self._cached_data = data
                return data
            else:
                raise FileNotFoundError(f"SimpleMix模式需要同时存在以下文件: {imu_patches_path}, {ecg_patches_path}, {labels_path}")
        
        if mode == 'newmix':
            # NewMix模式：同时加载IMU和ECG数据（不加载EEG），使用特征级融合
            imu_patches_path = os.path.join(self.dataset_path, 'imu_patches.npy')
            ecg_patches_path = os.path.join(self.dataset_path, 'ecg_patches.npy')
            labels_path = os.path.join(self.dataset_path, 'labels.npy')
            
            if (os.path.exists(imu_patches_path) and 
                os.path.exists(ecg_patches_path) and 
                os.path.exists(labels_path)):
                # 新格式：从training_dataset目录加载
                import time
                from tqdm import tqdm
                
                print(f"检测到NewMix模式新格式数据，从 {self.dataset_path} 加载...")
                
                # 加载numpy文件（使用mmap模式，不加载到内存）
                print("  正在加载numpy文件（使用mmap模式，不加载到内存）...")
                start_time = time.time()
                imu_patches = np.load(imu_patches_path, mmap_mode='r')
                print(f"    ✓ IMU数据加载完成 ({time.time()-start_time:.2f}秒, 形状: {imu_patches.shape}, mmap模式)")
                
                start_time = time.time()
                ecg_patches = np.load(ecg_patches_path, mmap_mode='r')
                print(f"    ✓ ECG数据加载完成 ({time.time()-start_time:.2f}秒, 形状: {ecg_patches.shape}, mmap模式)")
                
                start_time = time.time()
                labels = map_labels_10_to_5(np.load(labels_path))  # labels很小，直接加载并映射为5分类
                print(f"    ✓ Labels加载完成 ({time.time()-start_time:.2f}秒, 形状: {labels.shape})")
                
                # 加载subject_ids（如果存在）
                subject_ids_path = os.path.join(self.dataset_path, 'subject_ids.npy')
                if os.path.exists(subject_ids_path):
                    start_time = time.time()
                    subject_ids = np.load(subject_ids_path)  # subject_ids很小，直接加载
                    print(f"    ✓ Subject IDs加载完成 ({time.time()-start_time:.2f}秒, 形状: {subject_ids.shape})")
                    data['subject_ids'] = subject_ids
                else:
                    print(f"    ⚠️  未找到subject_ids.npy，将使用随机划分")
                
                # 直接使用numpy数组，避免转换为列表（大幅提升性能）
                print("  正在准备数据...")
                num_windows = imu_patches.shape[0]
                start_time = time.time()
                
                # 直接使用numpy数组，不需要转换为列表和复制
                # Dataset类可以直接索引numpy数组
                data['imu'] = imu_patches
                data['ecg'] = ecg_patches
                data['labels'] = labels
                
                elapsed = time.time() - start_time
                print(f"    ✓ 数据准备完成 ({elapsed:.2f}秒)")
                
                # 加载权重（如果存在）
                weights_path = os.path.join(self.dataset_path, 'weights.npy')
                if os.path.exists(weights_path):
                    data['weights'] = np.load(weights_path)
                
                print(f"\n✓ NewMix模式数据加载完成: {num_windows} 个窗口")
                print(f"  IMU patches形状: {imu_patches.shape}")
                print(f"  ECG patches形状: {ecg_patches.shape}")
                print(f"  Labels形状: {labels.shape}")
                print(f"{'='*60}\n")
                
                # 缓存数据
                if use_cache:
                    self._cached_data = data
                return data
            else:
                raise FileNotFoundError(f"NewMix模式需要同时存在以下文件: {imu_patches_path}, {ecg_patches_path}, {labels_path}")
        
        if mode == 'allmix':
            # AllMix模式：同时加载IMU、EEG和ECG数据，使用三模态特征级融合
            # 数据加载逻辑和mix模式完全一致
            imu_patches_path = os.path.join(self.dataset_path, 'imu_patches.npy')
            eeg_patches_path = os.path.join(self.dataset_path, 'eeg_patches.npy')
            ecg_patches_path = os.path.join(self.dataset_path, 'ecg_patches.npy')
            labels_path = os.path.join(self.dataset_path, 'labels.npy')
            
            if (os.path.exists(imu_patches_path) and 
                os.path.exists(eeg_patches_path) and 
                os.path.exists(ecg_patches_path) and 
                os.path.exists(labels_path)):
                # 新格式：从training_dataset目录加载
                import time
                from tqdm import tqdm
                
                print(f"检测到AllMix模式新格式数据，从 {self.dataset_path} 加载...")
                
                # 加载numpy文件（使用mmap模式，不加载到内存）
                print("  正在加载numpy文件（使用mmap模式，不加载到内存）...")
                start_time = time.time()
                imu_patches = np.load(imu_patches_path, mmap_mode='r')
                print(f"    ✓ IMU数据加载完成 ({time.time()-start_time:.2f}秒, 形状: {imu_patches.shape}, mmap模式)")
                
                start_time = time.time()
                eeg_patches = np.load(eeg_patches_path, mmap_mode='r')
                print(f"    ✓ EEG数据加载完成 ({time.time()-start_time:.2f}秒, 形状: {eeg_patches.shape}, mmap模式)")
                
                start_time = time.time()
                ecg_patches = np.load(ecg_patches_path, mmap_mode='r')
                print(f"    ✓ ECG数据加载完成 ({time.time()-start_time:.2f}秒, 形状: {ecg_patches.shape}, mmap模式)")
                
                start_time = time.time()
                labels = map_labels_10_to_5(np.load(labels_path))  # labels很小，直接加载并映射为5分类
                print(f"    ✓ Labels加载完成 ({time.time()-start_time:.2f}秒, 形状: {labels.shape})")
                
                # 加载subject_ids（如果存在）
                subject_ids_path = os.path.join(self.dataset_path, 'subject_ids.npy')
                if os.path.exists(subject_ids_path):
                    start_time = time.time()
                    subject_ids = np.load(subject_ids_path)  # subject_ids很小，直接加载
                    print(f"    ✓ Subject IDs加载完成 ({time.time()-start_time:.2f}秒, 形状: {subject_ids.shape})")
                    data['subject_ids'] = subject_ids
                else:
                    print(f"    ⚠️  未找到subject_ids.npy，将使用随机划分")
                
                # 直接使用numpy数组，避免转换为列表（大幅提升性能）
                print("  正在准备数据...")
                num_windows = imu_patches.shape[0]
                start_time = time.time()
                
                # 直接使用numpy数组，不需要转换为列表和复制
                # Dataset类可以直接索引numpy数组
                data['imu'] = imu_patches
                data['eeg'] = eeg_patches
                data['ecg'] = ecg_patches
                data['labels'] = labels
                
                elapsed = time.time() - start_time
                print(f"    ✓ 数据准备完成 ({elapsed:.2f}秒)")
                
                # 加载权重（如果存在）
                weights_path = os.path.join(self.dataset_path, 'weights.npy')
                if os.path.exists(weights_path):
                    data['weights'] = np.load(weights_path)
                
                print(f"\n✓ AllMix模式数据加载完成: {num_windows} 个窗口")
                print(f"  IMU patches形状: {imu_patches.shape}")
                print(f"  EEG patches形状: {eeg_patches.shape}")
                print(f"  ECG patches形状: {ecg_patches.shape}")
                print(f"  Labels形状: {labels.shape}")
                print(f"{'='*60}\n")
                
                # 缓存数据
                if use_cache:
                    self._cached_data = data
                return data
            else:
                raise FileNotFoundError(f"AllMix模式需要同时存在以下文件: {imu_patches_path}, {eeg_patches_path}, {ecg_patches_path}, {labels_path}")
        
        if mode == 'imu' or mode == 'rawimu':
            # IMU数据加载（imu和rawimu模式都只加载IMU数据）
            imu_patches_path = os.path.join(self.dataset_path, 'imu_patches.npy')
            labels_path = os.path.join(self.dataset_path, 'labels.npy')
            
            if os.path.exists(imu_patches_path) and os.path.exists(labels_path):
                # 新格式：从training_dataset目录加载
                import time
                from tqdm import tqdm
                
                print(f"检测到IMU新格式数据，从 {self.dataset_path} 加载...")
                print("  正在加载numpy文件（使用mmap模式，不加载到内存）...")
                start_time = time.time()
                imu_patches = np.load(imu_patches_path, mmap_mode='r')
                labels = map_labels_10_to_5(np.load(labels_path))
                print(f"    ✓ 数据加载完成 ({time.time()-start_time:.2f}秒, IMU形状: {imu_patches.shape})")
                
                # 加载subject_ids（如果存在）
                subject_ids_path = os.path.join(self.dataset_path, 'subject_ids.npy')
                if os.path.exists(subject_ids_path):
                    subject_ids = np.load(subject_ids_path)
                    print(f"    ✓ Subject IDs加载完成 (形状: {subject_ids.shape})")
                    data['subject_ids'] = subject_ids
                else:
                    print(f"    ⚠️  未找到subject_ids.npy，将使用随机划分")
                
                # 直接使用numpy数组，避免转换为列表（大幅提升性能）
                print("  正在准备数据...")
                num_windows = imu_patches.shape[0]
                start_time = time.time()
                
                # 直接使用numpy数组，不需要转换为列表和复制
                data['imu'] = imu_patches
                data['labels'] = labels
                
                elapsed = time.time() - start_time
                print(f"    ✓ 数据准备完成 ({elapsed:.2f}秒)")
                
                # 加载权重（如果存在）
                weights_path = os.path.join(self.dataset_path, 'weights.npy')
                if os.path.exists(weights_path):
                    data['weights'] = np.load(weights_path)
                
                print(f"加载完成: {num_windows} 个窗口")
                print(f"  IMU patches形状: {imu_patches.shape}")
                print(f"  Labels形状: {labels.shape}")
                
                # 缓存数据
                if use_cache:
                    self._cached_data = data
                return data
            
            # 旧格式：从data/raw目录加载
            imu_path = os.path.join(self.dataset_path, 'raw', 'imu')
            if os.path.exists(imu_path):
                data['imu'] = self.load_imu_data(imu_path)
            
            # Load labels
            labels_path = os.path.join(self.dataset_path, 'raw', 'labels', 'motion_sickness_scores.csv')
            if os.path.exists(labels_path):
                data['labels'] = self.load_labels(labels_path)
            
            # 缓存数据
            if use_cache:
                self._cached_data = data
            return data
        
        if mode == 'physio':
            # 生理信号数据加载（原有逻辑）
            # 检查是否是新格式（training_dataset格式）
            eeg_patches_path = os.path.join(self.dataset_path, 'eeg_patches.npy')
            ecg_patches_path = os.path.join(self.dataset_path, 'ecg_patches.npy')
            labels_path = os.path.join(self.dataset_path, 'labels.npy')
            
            if os.path.exists(eeg_patches_path) and os.path.exists(ecg_patches_path) and os.path.exists(labels_path):
                # 新格式：从training_dataset目录加载
                import time
                from tqdm import tqdm
                
                print(f"检测到新格式数据，从 {self.dataset_path} 加载...")
                
                print("  正在加载numpy文件（使用mmap模式，不加载到内存）...")
                start_time = time.time()
                # 使用mmap模式读取，不加载到内存
                eeg_patches = np.load(eeg_patches_path, mmap_mode='r')
                ecg_patches = np.load(ecg_patches_path, mmap_mode='r')
                labels = map_labels_10_to_5(np.load(labels_path))  # 直接加载并映射为5分类
                print(f"    ✓ 数据加载完成 ({time.time()-start_time:.2f}秒, EEG形状: {eeg_patches.shape}, ECG形状: {ecg_patches.shape}, mmap模式)")
                
                # 加载subject_ids（如果存在）
                subject_ids_path = os.path.join(self.dataset_path, 'subject_ids.npy')
                if os.path.exists(subject_ids_path):
                    subject_ids = np.load(subject_ids_path)
                    print(f"    ✓ Subject IDs加载完成 (形状: {subject_ids.shape})")
                    data['subject_ids'] = subject_ids
                else:
                    print(f"    ⚠️  未找到subject_ids.npy，将使用随机划分")
                
                # 直接使用numpy数组，避免转换为列表（大幅提升性能）
                print("  正在准备数据...")
                num_windows = eeg_patches.shape[0]
                start_time = time.time()
                
                # 直接使用numpy数组，不需要转换为列表和复制
                data['eeg'] = eeg_patches
                data['ecg'] = ecg_patches
                data['labels'] = labels
                
                elapsed = time.time() - start_time
                print(f"    ✓ 数据准备完成 ({elapsed:.2f}秒)")
                
                # 加载权重（如果存在）
                weights_path = os.path.join(self.dataset_path, 'weights.npy')
                if os.path.exists(weights_path):
                    data['weights'] = np.load(weights_path)
                
                print(f"加载完成: {num_windows} 个窗口")
                print(f"  EEG patches形状: {eeg_patches.shape}")
                print(f"  ECG patches形状: {ecg_patches.shape}")
                print(f"  Labels形状: {labels.shape}")
                
                # 缓存数据
                if use_cache:
                    self._cached_data = data
                return data
            
            # 旧格式：从data/raw目录加载
            # Load EEG data
            eeg_path = os.path.join(self.dataset_path, 'raw', 'eeg')
            if os.path.exists(eeg_path):
                data['eeg'] = self.load_eeg_data(eeg_path)
            
            # Load ECG data
            ecg_path = os.path.join(self.dataset_path, 'raw', 'ecg')
            if os.path.exists(ecg_path):
                data['ecg'] = self.load_ecg_data(ecg_path)
            
            # Load labels
            labels_path = os.path.join(self.dataset_path, 'raw', 'labels', 'motion_sickness_scores.csv')
            if os.path.exists(labels_path):
                data['labels'] = self.load_labels(labels_path)
            
            # 缓存数据
            if use_cache:
                self._cached_data = data
            return data
        
        if mode == 'eeg':
            # EEG数据加载（原有逻辑）
            # 检查是否是新格式（training_dataset格式）
            eeg_patches_path = os.path.join(self.dataset_path, 'eeg_patches.npy')
            labels_path = os.path.join(self.dataset_path, 'labels.npy')
            
            # 新格式：从training_dataset目录加载
            import time
            from tqdm import tqdm
            
            print(f"检测到新格式数据，从 {self.dataset_path} 加载...")
            
            print("  正在加载numpy文件（使用mmap模式，不加载到内存）...")
            start_time = time.time()
            # 使用mmap模式读取，不加载到内存
            eeg_patches = np.load(eeg_patches_path, mmap_mode='r')
            labels = map_labels_10_to_5(np.load(labels_path))  # 直接加载并映射为5分类
            print(f"    ✓ 数据加载完成 ({time.time()-start_time:.2f}秒, EEG形状: {eeg_patches.shape}, mmap模式)")
            
            # 加载subject_ids（如果存在）
            subject_ids_path = os.path.join(self.dataset_path, 'subject_ids.npy')
            if os.path.exists(subject_ids_path):
                subject_ids = np.load(subject_ids_path)
                print(f"    ✓ Subject IDs加载完成 (形状: {subject_ids.shape})")
                data['subject_ids'] = subject_ids
            else:
                print(f"    ⚠️  未找到subject_ids.npy，将使用随机划分")
            
            # 直接使用numpy数组，避免转换为列表（大幅提升性能）
            print("  正在准备数据...")
            num_windows = eeg_patches.shape[0]
            start_time = time.time()
            
            # 直接使用numpy数组，不需要转换为列表和复制
            data['eeg'] = eeg_patches
            data['labels'] = labels
            
            elapsed = time.time() - start_time
            print(f"    ✓ 数据准备完成 ({elapsed:.2f}秒)")
            
            # 加载权重（如果存在）
            weights_path = os.path.join(self.dataset_path, 'weights.npy')
            if os.path.exists(weights_path):
                data['weights'] = np.load(weights_path)
            
            print(f"加载完成: {num_windows} 个窗口")
            print(f"  EEG patches形状: {eeg_patches.shape}")
            print(f"  Labels形状: {labels.shape}")
            
            # 缓存数据
            if use_cache:
                self._cached_data = data
            return data

        if mode == 'ecg':
            # ECG数据加载（原有逻辑）
            # 检查是否是新格式（training_dataset格式）
            ecg_patches_path = os.path.join(self.dataset_path, 'ecg_patches.npy')
            labels_path = os.path.join(self.dataset_path, 'labels.npy')
            
            # 新格式：从training_dataset目录加载
            import time
            from tqdm import tqdm
            
            print(f"检测到新格式数据，从 {self.dataset_path} 加载...")
            
            print("  正在加载numpy文件（使用mmap模式，不加载到内存）...")
            start_time = time.time()
            # 使用mmap模式读取，不加载到内存
            ecg_patches = np.load(ecg_patches_path, mmap_mode='r')
            labels = map_labels_10_to_5(np.load(labels_path))  # 直接加载并映射为5分类
            print(f"    ✓ 数据加载完成 ({time.time()-start_time:.2f}秒, ECG形状: {ecg_patches.shape}, mmap模式)")
            
            # 加载subject_ids（如果存在）
            subject_ids_path = os.path.join(self.dataset_path, 'subject_ids.npy')
            if os.path.exists(subject_ids_path):
                subject_ids = np.load(subject_ids_path)
                print(f"    ✓ Subject IDs加载完成 (形状: {subject_ids.shape})")
                data['subject_ids'] = subject_ids
            else:
                print(f"    ⚠️  未找到subject_ids.npy，将使用随机划分")
            
            # 直接使用numpy数组，避免转换为列表（大幅提升性能）
            print("  正在准备数据...")
            num_windows = ecg_patches.shape[0]
            start_time = time.time()
            
            # 直接使用numpy数组，不需要转换为列表和复制
            data['ecg'] = ecg_patches
            data['labels'] = labels
            
            elapsed = time.time() - start_time
            print(f"    ✓ 数据准备完成 ({elapsed:.2f}秒)")
            
            # 加载权重（如果存在）
            weights_path = os.path.join(self.dataset_path, 'weights.npy')
            if os.path.exists(weights_path):
                data['weights'] = np.load(weights_path)
            
            print(f"加载完成: {num_windows} 个窗口")
            print(f"  ECG patches形状: {ecg_patches.shape}")
            print(f"  Labels形状: {labels.shape}")
            
            # 缓存数据
            if use_cache:
                self._cached_data = data
            return data
    
    def load_eeg_data(self, eeg_path):
        """
        Load EEG data from files
        """
        eeg_files = sorted([f for f in os.listdir(eeg_path) if f.endswith('.mat') or f.endswith('.npy')])
        
        if self.debug > 0:
            eeg_files = eeg_files[:self.debug]
        
        eeg_data = []
        for file in eeg_files:
            file_path = os.path.join(eeg_path, file)
            
            if file.endswith('.mat'):
                # Load .mat file
                from scipy.io import loadmat
                mat_data = loadmat(file_path)
                # Assume the main variable is named 'eeg' or 'data'
                if 'eeg' in mat_data:
                    eeg_data.append(mat_data['eeg'])
                elif 'data' in mat_data:
                    eeg_data.append(mat_data['data'])
                else:
                    # Use the first variable that's not metadata
                    for key in mat_data.keys():
                        if not key.startswith('__') and mat_data[key].ndim >= 2:
                            eeg_data.append(mat_data[key])
                            break
            
            elif file.endswith('.npy'):
                # Load .npy file
                eeg_data.append(np.load(file_path))
        
        return eeg_data
    
    def load_ecg_data(self, ecg_path):
        """
        Load ECG data from files
        """
        ecg_files = sorted([f for f in os.listdir(ecg_path) if f.endswith('.mat') or f.endswith('.npy')])
        
        if self.debug > 0:
            ecg_files = ecg_files[:self.debug]
        
        ecg_data = []
        for file in ecg_files:
            file_path = os.path.join(ecg_path, file)
            
            if file.endswith('.mat'):
                # Load .mat file
                from scipy.io import loadmat
                mat_data = loadmat(file_path)
                # Assume the main variable is named 'ecg' or 'data'
                if 'ecg' in mat_data:
                    ecg_data.append(mat_data['ecg'])
                elif 'data' in mat_data:
                    ecg_data.append(mat_data['data'])
                else:
                    # Use the first variable that's not metadata
                    for key in mat_data.keys():
                        if not key.startswith('__') and mat_data[key].ndim >= 2:
                            ecg_data.append(mat_data[key])
                            break
            
            elif file.endswith('.npy'):
                # Load .npy file
                ecg_data.append(np.load(file_path))
        
        return ecg_data
    
    def load_imu_data(self, imu_path):
        """
        Load IMU data from files
        """
        imu_files = sorted([f for f in os.listdir(imu_path) if f.endswith('.mat') or f.endswith('.npy')])
        
        if self.debug > 0:
            imu_files = imu_files[:self.debug]
        
        imu_data = []
        for file in imu_files:
            file_path = os.path.join(imu_path, file)
            
            if file.endswith('.mat'):
                # Load .mat file
                from scipy.io import loadmat
                mat_data = loadmat(file_path)
                # Assume the main variable is named 'imu' or 'data'
                if 'imu' in mat_data:
                    imu_data.append(mat_data['imu'])
                elif 'data' in mat_data:
                    imu_data.append(mat_data['data'])
                else:
                    # Use the first variable that's not metadata
                    for key in mat_data.keys():
                        if not key.startswith('__') and mat_data[key].ndim >= 2:
                            imu_data.append(mat_data[key])
                            break
            
            elif file.endswith('.npy'):
                # Load .npy file
                imu_data.append(np.load(file_path))
        
        return imu_data
    
    def load_labels(self, labels_path):
        """
        Load motion sickness labels
        """
        if labels_path.endswith('.csv'):
            df = pd.read_csv(labels_path)
            # Assume the label column is named 'score' or 'motion_sickness_score'
            if 'score' in df.columns:
                labels = df['score'].values
            elif 'motion_sickness_score' in df.columns:
                labels = df['motion_sickness_score'].values
            else:
                # Use the first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    labels = df[numeric_cols[0]].values
                else:
                    raise ValueError("No numeric label column found")
        else:
            labels = np.load(labels_path)
        
        return map_labels_10_to_5(labels)


class IMUDataset(Dataset):
    """
    Dataset class for IMU signals
    支持两种数据格式：
    1. 新格式：patches格式 (num_windows, num_patches, channels, patch_samples)
    2. 旧格式：连续信号格式 (channels, time_points)
    """
    
    def __init__(self, data_dict, window_length=3000, 
                 hop_length=1500, normalize=True, apply_filter=False,
                 imu_sampling_rate=250, normalization_stats=None):
        super().__init__()
        
        self.data_dict = data_dict
        self.window_length = window_length
        self.hop_length = hop_length
        self.normalize = normalize
        self.apply_filter = apply_filter
        self.imu_sampling_rate = imu_sampling_rate
        # 归一化统计量（如果提供，则使用全局统计量；否则使用样本内统计量）
        self.normalization_stats = normalization_stats
        
        # Extract data
        # 检查是否有索引映射（用于数据子集，避免复制大数据）
        self.indices = data_dict.get('_indices', None)
        
        self.imu_data = data_dict.get('imu', None)
        self.labels = data_dict.get('labels', None)
        self.weights = data_dict.get('weights', None)  # 样本权重
        self.subject_ids = data_dict.get('subject_ids', None)  # 保存subject_ids用于per-subject归一化
        
        # 检测数据格式：新格式（patches）还是旧格式（连续信号）
        self.is_patches_format = False
        self.is_numpy_array = False  # 标记数据是否为numpy数组格式
        if self.imu_data is not None:
            if isinstance(self.imu_data, np.ndarray):
                # 数据是numpy数组格式
                self.is_numpy_array = True
                if len(self.imu_data.shape) == 4:
                    # 新格式：(num_windows, num_patches, channels, patch_samples)
                    self.is_patches_format = True
            elif len(self.imu_data) > 0:
                # 数据是列表格式
                first_imu = self.imu_data[0]
                if isinstance(first_imu, np.ndarray) and len(first_imu.shape) == 3:
                    # 新格式：(num_patches, channels, patch_samples)
                    self.is_patches_format = True
        
        # 不在这里预处理！只保存归一化统计量，在 __getitem__ 中按需处理
        # 这样可以避免在初始化时处理整个数据集，节省内存
        self.processed_data = None  # 不再预处理整个数据集
        
        # Create segments (仅对旧格式)
        if not self.is_patches_format:
            self.segments = self.create_segments()
        else:
            self.segments = None
    
    def _normalize_window(self, window_data, modality='imu', subject_id=None):
        """
        对单个window进行归一化（在线处理）
        window_data: (num_patches, channels, patch_samples)
        subject_id: 该window所属的subject_id，用于per-subject归一化
        """
        num_patches, num_channels, patch_length = window_data.shape
        
        if self.normalization_stats is not None:
            # 检查是否是per-subject归一化格式: {subject_id: {modality: {mean, std}}}
            # 还是旧的全局归一化格式: {modality: {mean, std}}
            first_key = next(iter(self.normalization_stats.keys()))
            
            if isinstance(first_key, (int, np.integer)) or (isinstance(first_key, str) and first_key.isdigit()):
                # Per-subject归一化格式
                if subject_id is not None:
                    subject_id_key = int(subject_id) if isinstance(subject_id, (int, np.integer)) else subject_id
                    subject_stats = self.normalization_stats.get(subject_id_key, None)
                    
                    if subject_stats is not None:
                        stats = subject_stats.get(modality, {})
                        mean = stats.get('mean', None)
                        std = stats.get('std', None)
                        
                        if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                            # 使用该subject的统计量：扩展到 (1, num_channels, 1) 以便广播
                            mean_expanded = mean.reshape(1, num_channels, 1)
                            std_expanded = std.reshape(1, num_channels, 1)
                            std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                            return (window_data - mean_expanded) / (std_expanded + 1e-8)
            else:
                # 旧的全局归一化格式（向后兼容）
                stats = self.normalization_stats.get(modality, {})
                mean = stats.get('mean', None)
                std = stats.get('std', None)
                
                if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                    # 使用全局统计量：扩展到 (1, num_channels, 1) 以便广播
                    mean_expanded = mean.reshape(1, num_channels, 1)
                    std_expanded = std.reshape(1, num_channels, 1)
                    std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                    return (window_data - mean_expanded) / (std_expanded + 1e-8)
        
        # 回退到样本内归一化（对这个window的每个patch进行归一化）
        reshaped = window_data.reshape(-1, patch_length)
        means = reshaped.mean(axis=1, keepdims=True)
        stds = reshaped.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        normalized = (reshaped - means) / (stds + 1e-8)
        return normalized.reshape(num_patches, num_channels, patch_length)
    
    def preprocess_imu(self, imu_data):
        """
        Preprocess IMU signals
        支持两种格式：
        1. 新格式：patches (num_patches, channels, patch_samples) 
           - 对于已分片的数据，只进行可选的归一化（如果 normalize=True）
        2. 旧格式：连续信号 (channels, time_points) - 进行归一化
        """
        # 如果是numpy数组格式，直接处理整个数组（更快）
        if isinstance(imu_data, np.ndarray) and self.is_patches_format:
            if self.normalize:
                # 对整个数组进行归一化
                num_windows, num_patches, num_channels, patch_length = imu_data.shape
                
                if self.normalization_stats is not None:
                    imu_mean = self.normalization_stats.get('imu', {}).get('mean', None)
                    imu_std = self.normalization_stats.get('imu', {}).get('std', None)
                    
                    if imu_mean is not None and imu_std is not None and len(imu_mean.shape) == 1 and imu_mean.shape[0] == num_channels:
                            # 使用全局统计量：扩展到 (1, 1, num_channels, 1) 以便广播
                            mean_expanded = imu_mean.reshape(1, 1, num_channels, 1)
                            std_expanded = imu_std.reshape(1, 1, num_channels, 1)
                            std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                            processed_imu = (imu_data - mean_expanded) / (std_expanded + 1e-8)
                    else:
                        # 回退到样本内归一化
                        processed_imu = imu_data.copy()
                        reshaped = processed_imu.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        stds = np.where(stds < 1e-8, 1.0, stds)
                        reshaped = (reshaped - means) / (stds + 1e-8)
                        processed_imu = reshaped.reshape(num_windows, num_patches, num_channels, patch_length)
                else:
                    # 样本内归一化
                    processed_imu = imu_data.copy()
                    reshaped = processed_imu.reshape(-1, patch_length)
                    means = reshaped.mean(axis=1, keepdims=True)
                    stds = reshaped.std(axis=1, keepdims=True)
                    stds = np.where(stds < 1e-8, 1.0, stds)
                    processed_imu = (reshaped - means) / (stds + 1e-8)
                    processed_imu = processed_imu.reshape(num_windows, num_patches, num_channels, patch_length)
            else:
                # 不需要归一化，直接返回（不复制，节省内存）
                processed_imu = imu_data
            return processed_imu
        
        # 列表格式：逐个处理（向后兼容）
        processed_imu = []
        num_windows = len(imu_data)
        
        # 使用 tqdm 显示进度（仅在数据量大时）
        use_progress = num_windows > 100
        if use_progress:
            from tqdm import tqdm
            iterator = tqdm(imu_data, desc="预处理 IMU", unit="窗口")
        else:
            iterator = imu_data
        
        for window_imu in iterator:
            if self.is_patches_format:
                # 新格式：patches格式数据
                # window_imu: (num_patches, channels, patch_samples)
                if self.normalize:
                    window_imu = window_imu.copy()
                    num_patches, num_channels, patch_length = window_imu.shape
                    
                    if self.normalization_stats is not None:
                        # 使用全局统计量（训练集的mean和std）
                        imu_mean = self.normalization_stats.get('imu', {}).get('mean', None)
                        imu_std = self.normalization_stats.get('imu', {}).get('std', None)
                        
                        if imu_mean is not None and imu_std is not None:
                            # 使用全局统计量：对每个通道使用相同的mean和std
                            if len(imu_mean.shape) == 1 and imu_mean.shape[0] == num_channels:
                                    mean_expanded = imu_mean.reshape(1, num_channels, 1)
                                    std_expanded = imu_std.reshape(1, num_channels, 1)
                                    # 确保std不为0
                                    std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                                    window_imu = (window_imu - mean_expanded) / (std_expanded + 1e-8)
                            else:
                                # 如果形状不匹配，回退到样本内归一化
                                reshaped = window_imu.reshape(-1, patch_length)
                                means = reshaped.mean(axis=1, keepdims=True)
                                stds = reshaped.std(axis=1, keepdims=True)
                                # 处理std为0的情况
                                stds = np.where(stds < 1e-8, 1.0, stds)
                                window_imu = (reshaped - means) / (stds + 1e-8)
                                window_imu = window_imu.reshape(num_patches, num_channels, patch_length)
                        else:
                            # 如果没有提供统计量，使用样本内归一化
                            reshaped = window_imu.reshape(-1, patch_length)
                            means = reshaped.mean(axis=1, keepdims=True)
                            stds = reshaped.std(axis=1, keepdims=True)
                            # 处理std为0的情况
                            stds = np.where(stds < 1e-8, 1.0, stds)
                            window_imu = (reshaped - means) / (stds + 1e-8)
                            window_imu = window_imu.reshape(num_patches, num_channels, patch_length)
                    else:
                        # 没有提供全局统计量，使用样本内归一化
                        reshaped = window_imu.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        # 处理std为0的情况
                        stds = np.where(stds < 1e-8, 1.0, stds)
                        window_imu = (reshaped - means) / (stds + 1e-8)
                        window_imu = window_imu.reshape(num_patches, num_channels, patch_length)
                # 如果 normalize=False，直接使用原始数据（不进行归一化）
                processed_imu.append(window_imu)
            else:
                # 旧格式：连续信号，进行归一化
                # Normalize
                if self.normalize:
                    window_imu = self.normalize_signal(window_imu)
                
                processed_imu.append(window_imu)
        
        return processed_imu
    
    def normalize_signal(self, signal_data):
        """
        Normalize signal using z-score normalization
        """
        if len(signal_data.shape) == 1:
            # Single channel
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            if std_val > 0:
                signal_data = (signal_data - mean_val) / std_val
        else:
            # Multiple channels
            for ch in range(signal_data.shape[0]):
                mean_val = np.mean(signal_data[ch])
                std_val = np.std(signal_data[ch])
                if std_val > 0:
                    signal_data[ch] = (signal_data[ch] - mean_val) / std_val
        
        return signal_data
    
    def create_segments(self):
        """
        Create overlapping segments from the signals
        """
        segments = []
        
        # Get the minimum length across all subjects
        min_length = float('inf')
        
        if self.imu_data is not None:
            min_length = min(min_length, min([imu.shape[1] for imu in self.imu_data]))
        
        # 如果没有找到有效数据，返回空列表
        if min_length == float('inf') or min_length < self.window_length:
            return segments
        
        # Create segments
        # 确保所有参数都是整数
        min_length = int(min_length)
        window_length = int(self.window_length)
        hop_length = int(self.hop_length)
        for start in range(0, min_length - window_length + 1, hop_length):
            end = start + window_length
            segments.append((start, end))
        
        return segments
    
    def __len__(self):
        """Return the number of samples"""
        # 如果有索引映射，使用索引长度
        if self.indices is not None:
            return len(self.indices)
        
        if self.is_patches_format:
            # 新格式：每个窗口是一个样本
            return len(self.data_dict.get('labels', []))
        else:
            # 旧格式：每个subject的每个segment是一个样本
            if self.segments is None or len(self.segments) == 0:
                return 0
            return len(self.segments) * len(self.data_dict.get('labels', [0]))
    
    def __getitem__(self, idx):
        """
        Get a sample of data
        """
        sample = {}
        
        if self.is_patches_format:
            # 新格式：直接使用窗口索引，返回patches格式
            # 如果有索引映射，使用映射后的索引
            if self.indices is not None:
                window_idx = int(self.indices[idx])
            else:
                window_idx = idx
            
            # 获取该window的subject_id（用于per-subject归一化）
            subject_id = None
            if self.subject_ids is not None:
                if isinstance(self.subject_ids, np.ndarray):
                    subject_id = int(self.subject_ids[window_idx])
                elif isinstance(self.subject_ids, list):
                    subject_id = int(self.subject_ids[window_idx])
            
            if self.imu_data is not None:
                # 直接从原始数据获取，不复制
                window_imu = self.imu_data[window_idx]  # (num_patches, channels, patch_samples)
                # 在线归一化（只对这个window处理）
                if self.normalize:
                    window_imu = self._normalize_window(window_imu, 'imu', subject_id=subject_id)
                sample['imu'] = torch.FloatTensor(window_imu)
            
            # Add label
            if self.labels is not None:
                label = int(self.labels[window_idx])
                sample['label'] = torch.LongTensor([label])
            
            # Add weight (if available)
            if self.weights is not None:
                weight = float(self.weights[window_idx])
                sample['weight'] = torch.FloatTensor([weight])
            
            # Add subject_id (if available)
            if subject_id is not None:
                sample['subject_id'] = torch.LongTensor([subject_id])
        else:
            # 旧格式：计算subject和segment索引
            if self.segments is None or len(self.segments) == 0:
                raise ValueError(
                    f"无法创建segments：数据长度不足或数据格式不正确。"
                    f"window_length={self.window_length}, "
                    f"数据可能为空或太短。请检查数据格式是否为patches格式。"
                )
            num_subjects = len(self.data_dict.get('labels', [0]))
            subject_idx = idx // len(self.segments)
            segment_idx = idx % len(self.segments)
            
            start, end = self.segments[segment_idx]
            
            if self.imu_data is not None:
                window_imu = self.imu_data[subject_idx][:, start:end]
                if self.normalize:
                    window_imu = self.normalize_signal(window_imu)
                sample['imu'] = torch.FloatTensor(window_imu)
            
            # Add label
            if self.labels is not None:
                sample['label'] = torch.LongTensor([self.labels[subject_idx]])
            
            # Add weight (if available, 旧格式可能没有权重)
            if self.weights is not None:
                weight = float(self.weights[subject_idx])
                sample['weight'] = torch.FloatTensor([weight])
            
            # Add subject_id (if available)
            if self.subject_ids is not None:
                if isinstance(self.subject_ids, np.ndarray):
                    subject_id = int(self.subject_ids[subject_idx])
                elif isinstance(self.subject_ids, list):
                    subject_id = int(self.subject_ids[subject_idx])
                else:
                    subject_id = int(subject_idx)  # 如果没有subject_ids，使用subject_idx作为subject_id
                sample['subject_id'] = torch.LongTensor([subject_id])
        
        return sample


class RawIMUDataset(IMUDataset):
    """
    Dataset class for Raw IMU signals (只使用前6维：加速度和角速度)
    继承自 IMUDataset，但只返回前6维数据（去除conflicts数据）
    用于消融实验：比较使用完整18维数据 vs 仅使用原始6维数据的效果
    注意：数据格式应该是patches格式，先切片前6维，然后用6维统计量归一化
    """
    
    def _normalize_window(self, window_data, modality='imu', subject_id=None):
        """
        对单个window进行归一化（在线处理），只使用前6维
        window_data: (num_patches, channels, patch_samples) - 输入是18维
        subject_id: 该window所属的subject_id，用于per-subject归一化
        """
        # 先切片前6维
        window_data = window_data[:, :6, :]  # (num_patches, 6, patch_samples)
        num_patches, num_channels, patch_length = window_data.shape
        
        if self.normalization_stats is not None and len(self.normalization_stats) > 0:
            # 检查是否是per-subject归一化格式: {subject_id: {modality: {mean, std}}}
            # 还是旧的全局归一化格式: {modality: {mean, std}}
            first_key = next(iter(self.normalization_stats.keys()))
            
            if isinstance(first_key, (int, np.integer)) or (isinstance(first_key, str) and first_key.isdigit()):
                # Per-subject归一化格式
                if subject_id is not None:
                    subject_id_key = int(subject_id) if isinstance(subject_id, (int, np.integer)) else subject_id
                    subject_stats = self.normalization_stats.get(subject_id_key, None)
                    
                    if subject_stats is not None:
                        stats = subject_stats.get(modality, {})
                        mean = stats.get('mean', None)
                        std = stats.get('std', None)
                        
                        # 归一化统计量应该是6维的（在_compute_global_normalization_stats中已经处理了）
                        if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                            # 使用该subject的统计量：扩展到 (1, num_channels, 1) 以便广播
                            mean_expanded = mean.reshape(1, num_channels, 1)
                            std_expanded = std.reshape(1, num_channels, 1)
                            std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                            return (window_data - mean_expanded) / (std_expanded + 1e-8)
            else:
                # 旧的全局归一化格式（向后兼容）
                stats = self.normalization_stats.get(modality, {})
                mean = stats.get('mean', None)
                std = stats.get('std', None)
                
                # 归一化统计量应该是6维的（在_compute_global_normalization_stats中已经处理了）
                if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                    # 使用全局统计量：扩展到 (1, num_channels, 1) 以便广播
                    mean_expanded = mean.reshape(1, num_channels, 1)
                    std_expanded = std.reshape(1, num_channels, 1)
                    std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                    return (window_data - mean_expanded) / (std_expanded + 1e-8)
        
        # 回退到样本内归一化（对这个window的每个patch进行归一化）
        reshaped = window_data.reshape(-1, patch_length)
        means = reshaped.mean(axis=1, keepdims=True)
        stds = reshaped.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        normalized = (reshaped - means) / (stds + 1e-8)
        return normalized.reshape(num_patches, num_channels, patch_length)


class MixDataset(Dataset):
    """
    Dataset class for Mix mode (IMU + Physio signals)
    同时加载IMU和Physio数据，用于决策级融合
    支持两种数据格式：
    1. 新格式：patches格式 (num_windows, num_patches, channels, patch_samples)
    2. 旧格式：连续信号格式 (channels, time_points)
    """
    
    def __init__(self, data_dict, window_length=3000, 
                 hop_length=1500, normalize=True, apply_filter=False,
                 imu_sampling_rate=250, eeg_sampling_rate=250, ecg_sampling_rate=250,
                 normalization_stats=None):
        super().__init__()
        
        self.data_dict = data_dict
        self.window_length = window_length
        self.hop_length = hop_length
        self.normalize = normalize
        self.apply_filter = apply_filter
        self.imu_sampling_rate = imu_sampling_rate
        self.eeg_sampling_rate = eeg_sampling_rate
        self.ecg_sampling_rate = ecg_sampling_rate
        self.normalization_stats = normalization_stats
        
        # Extract data
        # 检查是否有索引映射（用于数据子集，避免复制大数据）
        self.indices = data_dict.get('_indices', None)
        
        self.imu_data = data_dict.get('imu', None)
        self.eeg_data = data_dict.get('eeg', None)
        self.ecg_data = data_dict.get('ecg', None)
        self.labels = data_dict.get('labels', None)
        self.weights = data_dict.get('weights', None)
        self.subject_ids = data_dict.get('subject_ids', None)  # 保存subject_ids用于per-subject归一化
        
        # 检测数据格式：新格式（patches）还是旧格式（连续信号）
        self.is_patches_format = False
        self.is_numpy_array = False  # 标记数据是否为numpy数组格式
        if self.imu_data is not None:
            if isinstance(self.imu_data, np.ndarray):
                # 数据是numpy数组格式
                self.is_numpy_array = True
                if len(self.imu_data.shape) == 4:
                    # 新格式：(num_windows, num_patches, channels, patch_samples)
                    self.is_patches_format = True
            elif len(self.imu_data) > 0:
                # 数据是列表格式
                first_imu = self.imu_data[0]
                if isinstance(first_imu, np.ndarray) and len(first_imu.shape) == 3:
                        # 新格式：(num_patches, channels, patch_samples)
                    self.is_patches_format = True
        
        # 不在这里预处理！只保存归一化统计量，在 __getitem__ 中按需处理
        # 这样可以避免在初始化时处理整个数据集，节省内存
        self.processed_data = None  # 不再预处理整个数据集
        
        # Create segments (仅对旧格式)
        if not self.is_patches_format:
            self.segments = self.create_segments()
        else:
            self.segments = None
    
    def _normalize_window(self, window_data, modality):
        """
        对单个window进行归一化（在线处理）
        window_data: (num_patches, channels, patch_samples)
        """
        num_patches, num_channels, patch_length = window_data.shape
        
        if self.normalization_stats is not None:
            # 使用全局统计量
            stats = self.normalization_stats.get(modality, {})
            mean = stats.get('mean', None)
            std = stats.get('std', None)
            
            if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                # 使用全局统计量：扩展到 (1, num_channels, 1) 以便广播
                mean_expanded = mean.reshape(1, num_channels, 1)
                std_expanded = std.reshape(1, num_channels, 1)
                std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                return (window_data - mean_expanded) / (std_expanded + 1e-8)
        
        # 回退到样本内归一化（对这个window的每个patch进行归一化）
        reshaped = window_data.reshape(-1, patch_length)
        means = reshaped.mean(axis=1, keepdims=True)
        stds = reshaped.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        normalized = (reshaped - means) / (stds + 1e-8)
        return normalized.reshape(num_patches, num_channels, patch_length)
    
    def normalize_signal(self, signal_data):
        """Normalize signal using z-score normalization (for old format)"""
        if len(signal_data.shape) == 1:
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            if std_val > 0:
                signal_data = (signal_data - mean_val) / std_val
        else:
            for ch in range(signal_data.shape[0]):
                mean_val = np.mean(signal_data[ch])
                std_val = np.std(signal_data[ch])
                if std_val > 0:
                    signal_data[ch] = (signal_data[ch] - mean_val) / std_val
        return signal_data
    
    def preprocess_imu(self, imu_data):
        """
        Preprocess IMU signals
        支持两种格式：
        1. 新格式：patches (num_patches, channels, patch_samples) 
           - 对于已分片的数据，只进行可选的归一化（如果 normalize=True）
        2. 旧格式：连续信号 (channels, time_points) - 进行归一化
        """
        # 如果是numpy数组格式，直接处理整个数组（更快）
        if isinstance(imu_data, np.ndarray) and self.is_patches_format:
            if self.normalize:
                # 对整个数组进行归一化
                num_windows, num_patches, num_channels, patch_length = imu_data.shape
                
                if self.normalization_stats is not None:
                    imu_mean = self.normalization_stats.get('imu', {}).get('mean', None)
                    imu_std = self.normalization_stats.get('imu', {}).get('std', None)
                    
                    if imu_mean is not None and imu_std is not None and len(imu_mean.shape) == 1 and imu_mean.shape[0] == num_channels:
                        # 使用全局统计量：扩展到 (1, 1, num_channels, 1) 以便广播
                        mean_expanded = imu_mean.reshape(1, 1, num_channels, 1)
                        std_expanded = imu_std.reshape(1, 1, num_channels, 1)
                        std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                        processed_imu = (imu_data - mean_expanded) / (std_expanded + 1e-8)
                    else:
                        # 回退到样本内归一化
                        processed_imu = imu_data.copy()
                        reshaped = processed_imu.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        stds = np.where(stds < 1e-8, 1.0, stds)
                        reshaped = (reshaped - means) / (stds + 1e-8)
                        processed_imu = reshaped.reshape(num_windows, num_patches, num_channels, patch_length)
                else:
                    # 样本内归一化
                    processed_imu = imu_data.copy()
                    reshaped = processed_imu.reshape(-1, patch_length)
                    means = reshaped.mean(axis=1, keepdims=True)
                    stds = reshaped.std(axis=1, keepdims=True)
                    stds = np.where(stds < 1e-8, 1.0, stds)
                    processed_imu = (reshaped - means) / (stds + 1e-8)
                    processed_imu = processed_imu.reshape(num_windows, num_patches, num_channels, patch_length)
            else:
                # 不需要归一化，直接返回（不复制，节省内存）
                processed_imu = imu_data
            return processed_imu
        
        # 列表格式：逐个处理（向后兼容）
        processed_imu = []
        num_windows = len(imu_data)
        
        # 使用 tqdm 显示进度（仅在数据量大时）
        use_progress = num_windows > 100
        if use_progress:
            from tqdm import tqdm
            iterator = tqdm(imu_data, desc="预处理 IMU (Mix)", unit="窗口")
        else:
            iterator = imu_data
        
        for window_imu in iterator:
            if self.is_patches_format:
                window_imu = window_imu.copy()
                
                if self.normalize:
                    num_patches, num_channels, patch_length = window_imu.shape
                    
                    if self.normalization_stats is not None:
                        imu_mean = self.normalization_stats.get('imu', {}).get('mean', None)
                        imu_std = self.normalization_stats.get('imu', {}).get('std', None)
                        
                        if imu_mean is not None and imu_std is not None:
                            if len(imu_mean.shape) == 1 and imu_mean.shape[0] == num_channels:
                                mean_expanded = imu_mean.reshape(1, num_channels, 1)
                                std_expanded = imu_std.reshape(1, num_channels, 1)
                                std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                                window_imu = (window_imu - mean_expanded) / (std_expanded + 1e-8)
                
                
            processed_imu.append(window_imu)
        
        return processed_imu
    
    def preprocess_eeg(self, eeg_data):
        """
        Preprocess EEG signals
        支持两种格式：
        1. 新格式：patches (num_patches, channels, patch_samples) 
           - 对于已分片的数据，只进行可选的归一化（如果 normalize=True）
        2. 旧格式：连续信号 (channels, time_points) - 进行滤波和归一化
        """
        # 如果是numpy数组格式，直接处理整个数组（更快）
        if isinstance(eeg_data, np.ndarray) and self.is_patches_format:
            if self.normalize:
                # 对整个数组进行归一化
                num_windows, num_patches, num_channels, patch_length = eeg_data.shape
                
                if self.normalization_stats is not None:
                    eeg_mean = self.normalization_stats.get('eeg', {}).get('mean', None)
                    eeg_std = self.normalization_stats.get('eeg', {}).get('std', None)
                    
                    if eeg_mean is not None and eeg_std is not None and len(eeg_mean.shape) == 1 and eeg_mean.shape[0] == num_channels:
                        # 使用全局统计量：扩展到 (1, 1, num_channels, 1) 以便广播
                        mean_expanded = eeg_mean.reshape(1, 1, num_channels, 1)
                        std_expanded = eeg_std.reshape(1, 1, num_channels, 1)
                        std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                        # 归一化整个数组
                        processed_eeg = (eeg_data - mean_expanded) / (std_expanded + 1e-8)
                    else:
                        # 回退到样本内归一化（对每个窗口的每个patch进行归一化）
                        processed_eeg = eeg_data.copy()
                        reshaped = processed_eeg.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        stds = np.where(stds < 1e-8, 1.0, stds)
                        reshaped = (reshaped - means) / (stds + 1e-8)
                        processed_eeg = reshaped.reshape(num_windows, num_patches, num_channels, patch_length)
                else:
                    # 样本内归一化
                    processed_eeg = eeg_data.copy()
                    reshaped = processed_eeg.reshape(-1, patch_length)
                    means = reshaped.mean(axis=1, keepdims=True)
                    stds = reshaped.std(axis=1, keepdims=True)
                    stds = np.where(stds < 1e-8, 1.0, stds)
                    reshaped = (reshaped - means) / (stds + 1e-8)
                    processed_eeg = reshaped.reshape(num_windows, num_patches, num_channels, patch_length)
            else:
                # 不需要归一化，直接返回（不复制，节省内存）
                processed_eeg = eeg_data
            return processed_eeg
        
        # 列表格式：逐个处理（向后兼容）
        processed_eeg = []
        num_windows = len(eeg_data)
        
        from tqdm import tqdm
        iterator = tqdm(eeg_data, desc="预处理 EEG (Mix)", unit="窗口", disable=num_windows < 10)
        
        for window_eeg in iterator:
            if self.is_patches_format:
                window_eeg = window_eeg.copy()
                
                if self.normalize:
                    num_patches, num_channels, patch_length = window_eeg.shape
                    
                    if self.normalization_stats is not None:
                        eeg_mean = self.normalization_stats.get('eeg', {}).get('mean', None)
                        eeg_std = self.normalization_stats.get('eeg', {}).get('std', None)
                        
                        if eeg_mean is not None and eeg_std is not None:
                            if len(eeg_mean.shape) == 1 and eeg_mean.shape[0] == num_channels:
                                mean_expanded = eeg_mean.reshape(1, num_channels, 1)
                                std_expanded = eeg_std.reshape(1, num_channels, 1)
                                std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                                window_eeg = (window_eeg - mean_expanded) / (std_expanded + 1e-8)
                
            processed_eeg.append(window_eeg)
        
        return processed_eeg
    
    def preprocess_ecg(self, ecg_data):
        """
        Preprocess ECG signals
        支持两种格式：
        1. 新格式：patches (num_patches, channels, patch_samples) 
           - 对于已分片的数据，只进行可选的归一化（如果 normalize=True）
        2. 旧格式：连续信号 (channels, time_points) - 进行滤波和归一化
        """
        # 如果是numpy数组格式，直接处理整个数组（更快）
        if isinstance(ecg_data, np.ndarray) and self.is_patches_format:
            if self.normalize:
                # 对整个数组进行归一化
                num_windows, num_patches, num_channels, patch_length = ecg_data.shape
                
                if self.normalization_stats is not None:
                    ecg_mean = self.normalization_stats.get('ecg', {}).get('mean', None)
                    ecg_std = self.normalization_stats.get('ecg', {}).get('std', None)
                    
                    if ecg_mean is not None and ecg_std is not None and len(ecg_mean.shape) == 1 and ecg_mean.shape[0] == num_channels:
                        # 使用全局统计量：扩展到 (1, 1, num_channels, 1) 以便广播
                        mean_expanded = ecg_mean.reshape(1, 1, num_channels, 1)
                        std_expanded = ecg_std.reshape(1, 1, num_channels, 1)
                        std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                        # 归一化整个数组
                        processed_ecg = (ecg_data - mean_expanded) / (std_expanded + 1e-8)
                    else:
                        # 回退到样本内归一化
                        processed_ecg = ecg_data.copy()
                        reshaped = processed_ecg.reshape(-1, patch_length)
                        means = reshaped.mean(axis=1, keepdims=True)
                        stds = reshaped.std(axis=1, keepdims=True)
                        stds = np.where(stds < 1e-8, 1.0, stds)
                        reshaped = (reshaped - means) / (stds + 1e-8)
                        processed_ecg = reshaped.reshape(num_windows, num_patches, num_channels, patch_length)
                else:
                    # 样本内归一化
                    processed_ecg = ecg_data.copy()
                    reshaped = processed_ecg.reshape(-1, patch_length)
                    means = reshaped.mean(axis=1, keepdims=True)
                    stds = reshaped.std(axis=1, keepdims=True)
                    stds = np.where(stds < 1e-8, 1.0, stds)
                    reshaped = (reshaped - means) / (stds + 1e-8)
                    processed_ecg = reshaped.reshape(num_windows, num_patches, num_channels, patch_length)
            else:
                # 不需要归一化，直接返回（不复制，节省内存）
                processed_ecg = ecg_data
            return processed_ecg
        
        # 列表格式：逐个处理（向后兼容）
        processed_ecg = []
        num_windows = len(ecg_data)
        
        from tqdm import tqdm
        iterator = tqdm(ecg_data, desc="预处理 ECG (Mix)", unit="窗口", disable=num_windows < 10)
        
        for window_ecg in iterator:
            if self.is_patches_format:
                window_ecg = window_ecg.copy()
                
                if self.normalize:
                    num_patches, num_channels, patch_length = window_ecg.shape
                    
                    if self.normalization_stats is not None:
                        ecg_mean = self.normalization_stats.get('ecg', {}).get('mean', None)
                        ecg_std = self.normalization_stats.get('ecg', {}).get('std', None)
                        
                        if ecg_mean is not None and ecg_std is not None:
                            if len(ecg_mean.shape) == 1 and ecg_mean.shape[0] == num_channels:
                                mean_expanded = ecg_mean.reshape(1, num_channels, 1)
                                std_expanded = ecg_std.reshape(1, num_channels, 1)
                                std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                                window_ecg = (window_ecg - mean_expanded) / (std_expanded + 1e-8)
                
            processed_ecg.append(window_ecg)
        
        return processed_ecg
    
    def _normalize_window(self, window_data, modality, subject_id=None):
        """
        对单个window进行归一化（在线处理）
        window_data: (num_patches, channels, patch_samples)
        subject_id: 该window所属的subject_id，用于per-subject归一化
        """
        num_patches, num_channels, patch_length = window_data.shape
        
        if self.normalization_stats is not None:
            # 检查是否是per-subject归一化格式: {subject_id: {modality: {mean, std}}}
            # 还是旧的全局归一化格式: {modality: {mean, std}}
            first_key = next(iter(self.normalization_stats.keys()))
            
            if isinstance(first_key, (int, np.integer)) or (isinstance(first_key, str) and first_key.isdigit()):
                # Per-subject归一化格式
                if subject_id is not None:
                    subject_id_key = int(subject_id) if isinstance(subject_id, (int, np.integer)) else subject_id
                    subject_stats = self.normalization_stats.get(subject_id_key, None)
                    
                    if subject_stats is not None:
                        stats = subject_stats.get(modality, {})
                        mean = stats.get('mean', None)
                        std = stats.get('std', None)
                        
                        if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                            # 使用该subject的统计量：扩展到 (1, num_channels, 1) 以便广播
                            mean_expanded = mean.reshape(1, num_channels, 1)
                            std_expanded = std.reshape(1, num_channels, 1)
                            std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                            return (window_data - mean_expanded) / (std_expanded + 1e-8)
            else:
                # 旧的全局归一化格式（向后兼容）
                stats = self.normalization_stats.get(modality, {})
                mean = stats.get('mean', None)
                std = stats.get('std', None)
                
                if mean is not None and std is not None and len(mean.shape) == 1 and mean.shape[0] == num_channels:
                    # 使用全局统计量：扩展到 (1, num_channels, 1) 以便广播
                    mean_expanded = mean.reshape(1, num_channels, 1)
                    std_expanded = std.reshape(1, num_channels, 1)
                    std_expanded = np.where(std_expanded < 1e-8, 1.0, std_expanded)
                    return (window_data - mean_expanded) / (std_expanded + 1e-8)
        
        # 回退到样本内归一化（对这个window的每个patch进行归一化）
        reshaped = window_data.reshape(-1, patch_length)
        means = reshaped.mean(axis=1, keepdims=True)
        stds = reshaped.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        normalized = (reshaped - means) / (stds + 1e-8)
        return normalized.reshape(num_patches, num_channels, patch_length)
    
    def create_segments(self):
        """Create overlapping segments from the signals"""
        segments = []
        min_length = float('inf')
        
        if self.imu_data is not None:
            min_length = min(min_length, min([imu.shape[1] for imu in self.imu_data]))
        
        if self.eeg_data is not None:
            min_length = min(min_length, min([eeg.shape[1] for eeg in self.eeg_data]))
        
        if self.ecg_data is not None:
            min_length = min(min_length, min([ecg.shape[1] for ecg in self.ecg_data]))
        
        for start in range(0, int(min_length) - self.window_length + 1, self.hop_length):
            end = start + self.window_length
            segments.append((start, end))
        
        return segments
    
    def __len__(self):
        """Return the number of samples"""
        # 如果有索引映射，使用索引长度
        if self.indices is not None:
            return len(self.indices)
        
        if self.is_patches_format:
            return len(self.data_dict.get('labels', []))
        else:
            return len(self.segments) * len(self.data_dict.get('labels', [0]))
    
    def __getitem__(self, idx):
        """Get a sample of data - 在线处理，只处理单个样本"""
        sample = {}
        
        if self.is_patches_format:
            # 如果有索引映射，使用映射后的索引
            if self.indices is not None:
                window_idx = int(self.indices[idx])
            else:
                window_idx = idx
            
            # 获取该window的subject_id（用于per-subject归一化）
            subject_id = None
            if self.subject_ids is not None:
                if isinstance(self.subject_ids, np.ndarray):
                    subject_id = int(self.subject_ids[window_idx])
                elif isinstance(self.subject_ids, list):
                    subject_id = int(self.subject_ids[window_idx])
            
            if self.imu_data is not None:
                # 直接从原始数据获取，不复制
                window_imu = self.imu_data[window_idx]  # (num_patches, channels, patch_samples)
                # 在线归一化（只对这个window处理）
                if self.normalize:
                    window_imu = self._normalize_window(window_imu, 'imu', subject_id=subject_id)
                sample['imu'] = torch.FloatTensor(window_imu)
            
            if self.eeg_data is not None:
                # 直接从原始数据获取，不复制
                window_eeg = self.eeg_data[window_idx]  # (num_patches, channels, patch_samples)
                # 在线归一化（只对这个window处理）
                if self.normalize:
                    window_eeg = self._normalize_window(window_eeg, 'eeg', subject_id=subject_id)
                sample['eeg'] = torch.FloatTensor(window_eeg)
            
            if self.ecg_data is not None:
                # 直接从原始数据获取，不复制
                window_ecg = self.ecg_data[window_idx]  # (num_patches, channels, patch_samples)
                # 在线归一化（只对这个window处理）
                if self.normalize:
                    window_ecg = self._normalize_window(window_ecg, 'ecg', subject_id=subject_id)
                sample['ecg'] = torch.FloatTensor(window_ecg)
            
            if self.labels is not None:
                label = int(self.labels[window_idx])
                sample['label'] = torch.LongTensor([label])
            
            if self.weights is not None:
                weight = float(self.weights[window_idx])
                sample['weight'] = torch.FloatTensor([weight])
            
            # Add subject_id (if available)
            if subject_id is not None:
                sample['subject_id'] = torch.LongTensor([subject_id])
        else:
            num_subjects = len(self.data_dict.get('labels', [0]))
            subject_idx = idx // len(self.segments)
            segment_idx = idx % len(self.segments)
            
            start, end = self.segments[segment_idx]
            
            if self.imu_data is not None:
                window_imu = self.imu_data[subject_idx][:, start:end]
                if self.normalize:
                    window_imu = self.normalize_signal(window_imu)
                sample['imu'] = torch.FloatTensor(window_imu)
            
            if self.eeg_data is not None:
                window_eeg = self.eeg_data[subject_idx][:, start:end]
                if self.normalize:
                    window_eeg = self.normalize_signal(window_eeg)
                sample['eeg'] = torch.FloatTensor(window_eeg)
            
            if self.ecg_data is not None:
                window_ecg = self.ecg_data[subject_idx][:, start:end]
                if self.normalize:
                    window_ecg = self.normalize_signal(window_ecg)
                sample['ecg'] = torch.FloatTensor(window_ecg)
            
            if self.labels is not None:
                sample['label'] = torch.LongTensor([self.labels[subject_idx]])
            
            if self.weights is not None:
                weight = float(self.weights[subject_idx])
                sample['weight'] = torch.FloatTensor([weight])
            
            # Add subject_id (if available)
            if self.subject_ids is not None:
                if isinstance(self.subject_ids, np.ndarray):
                    subject_id = int(self.subject_ids[subject_idx])
                elif isinstance(self.subject_ids, list):
                    subject_id = int(self.subject_ids[subject_idx])
                else:
                    subject_id = int(subject_idx)  # 如果没有subject_ids，使用subject_idx作为subject_id
                sample['subject_id'] = torch.LongTensor([subject_id])
        
        return sample
