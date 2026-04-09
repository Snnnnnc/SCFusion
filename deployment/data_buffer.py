"""
数据缓冲区模块
使用滑动窗口缓冲区累积数据
"""

import numpy as np
import time
from typing import Dict, Optional
from collections import deque
import threading


class SlidingWindowBuffer:
    """滑动窗口缓冲区"""
    
    def __init__(self, 
                 window_length: int = 2500,
                 target_sampling_rate: int = 250,
                 imu_channels: int = 6,
                 eeg_channels: int = 59,
                 ecg_channels: int = 1,
                 imu_sampling_rate: int = 100,
                 eeg_sampling_rate: int = 250,
                 ecg_sampling_rate: int = 250):
        """
        初始化滑动窗口缓冲区
        
        Args:
            window_length: 窗口长度（采样点数，默认10秒@250Hz=2500）
            target_sampling_rate: 目标采样率（Hz）
            imu_channels: IMU通道数
            eeg_channels: EEG通道数
            ecg_channels: ECG通道数
            imu_sampling_rate: IMU原始采样率
            eeg_sampling_rate: EEG原始采样率
            ecg_sampling_rate: ECG原始采样率
        """
        self.window_length = window_length
        self.target_sampling_rate = target_sampling_rate
        self.imu_channels = imu_channels
        self.eeg_channels = eeg_channels
        self.ecg_channels = ecg_channels
        
        # 原始采样率
        self.imu_sampling_rate = imu_sampling_rate
        self.eeg_sampling_rate = eeg_sampling_rate
        self.ecg_sampling_rate = ecg_sampling_rate
        
        # 针对不同采样率计算各自的窗口长度（均为10秒）
        duration_sec = window_length / target_sampling_rate
        self.imu_win_len = int(imu_sampling_rate * duration_sec)
        self.eeg_win_len = int(eeg_sampling_rate * duration_sec)
        self.ecg_win_len = int(ecg_sampling_rate * duration_sec)
        
        # 使用deque存储数据，每个元素是一帧数据 (channels, 1)
        # 同时存储时间戳用于同步
        self.imu_buffer = deque(maxlen=self.imu_win_len)
        self.eeg_buffer = deque(maxlen=self.eeg_win_len)
        self.ecg_buffer = deque(maxlen=self.ecg_win_len)
        
        self.imu_timestamps = deque(maxlen=self.imu_win_len)
        self.eeg_timestamps = deque(maxlen=self.eeg_win_len)
        self.ecg_timestamps = deque(maxlen=self.ecg_win_len)
        
        # 线程锁（如果多线程采集）
        self.lock = threading.Lock()
        
        # 统计信息
        self.total_frames_received = 0
    
    def add_frames(self, 
                   imu: Optional[np.ndarray] = None,
                   eeg: Optional[np.ndarray] = None,
                   ecg: Optional[np.ndarray] = None,
                   timestamps: Optional[np.ndarray] = None) -> bool:
        """
        添加多帧数据
        
        Args:
            imu: IMU数据 (imu_channels, N_imu)
            eeg: EEG数据 (eeg_channels, N_eeg)
            ecg: ECG数据 (ecg_channels, N_ecg)
            timestamps: 时间戳数组（可选）
        """
        with self.lock:
            if imu is not None:
                if len(imu.shape) == 1: imu = imu.reshape(-1, 1)
                N = imu.shape[1]
                for i in range(N):
                    self.imu_buffer.append(imu[:, i:i+1].copy())
                    ts = timestamps[i] if timestamps is not None and i < len(timestamps) else time.time()
                    self.imu_timestamps.append(ts)
            
            if eeg is not None:
                if len(eeg.shape) == 1: eeg = eeg.reshape(-1, 1)
                N = eeg.shape[1]
                for i in range(N):
                    self.eeg_buffer.append(eeg[:, i:i+1].copy())
                    ts = timestamps[i] if timestamps is not None and i < len(timestamps) else time.time()
                    self.eeg_timestamps.append(ts)
                    
            if ecg is not None:
                if len(ecg.shape) == 1: ecg = ecg.reshape(-1, 1)
                N = ecg.shape[1]
                for i in range(N):
                    self.ecg_buffer.append(ecg[:, i:i+1].copy())
                    ts = timestamps[i] if timestamps is not None and i < len(timestamps) else time.time()
                    self.ecg_timestamps.append(ts)
            
            self.total_frames_received += 1
            return True
    
    def add_frame(self, 
                  imu: Optional[np.ndarray] = None,
                  eeg: Optional[np.ndarray] = None,
                  ecg: Optional[np.ndarray] = None,
                  timestamp: Optional[float] = None) -> bool:
        """
        添加新一帧数据
        
        Args:
            imu: IMU数据帧 (imu_channels, 1) 或 (6, 1)
            eeg: EEG数据帧 (eeg_channels, 1)
            ecg: ECG数据帧 (ecg_channels, 1)
            timestamp: 时间戳（如果None则使用当前时间）
        
        Returns:
            bool: 是否成功添加
        """
        if timestamp is None:
            import time
            timestamp = time.time()
        
        with self.lock:
            if imu is not None:
                # 验证数据维度
                if imu.shape[0] != self.imu_channels:
                    print(f"⚠️  IMU数据维度不匹配: 期望{self.imu_channels}, 实际{imu.shape[0]}")
                    return False
                # 确保是 (channels, 1) 格式
                if len(imu.shape) == 1:
                    imu = imu.reshape(-1, 1)
                self.imu_buffer.append(imu.copy())
                self.imu_timestamps.append(timestamp)
            
            if eeg is not None:
                if eeg.shape[0] != self.eeg_channels:
                    print(f"⚠️  EEG数据维度不匹配: 期望{self.eeg_channels}, 实际{eeg.shape[0]}")
                    return False
                if len(eeg.shape) == 1:
                    eeg = eeg.reshape(-1, 1)
                self.eeg_buffer.append(eeg.copy())
                self.eeg_timestamps.append(timestamp)
            
            if ecg is not None:
                if ecg.shape[0] != self.ecg_channels:
                    print(f"⚠️  ECG数据维度不匹配: 期望{self.ecg_channels}, 实际{ecg.shape[0]}")
                    return False
                if len(ecg.shape) == 1:
                    ecg = ecg.reshape(-1, 1)
                self.ecg_buffer.append(ecg.copy())
                self.ecg_timestamps.append(timestamp)
            
            self.total_frames_received += 1
            return True
    
    def is_ready(self) -> bool:
        """
        检查是否已累积足够数据（所有模态都达到各自采样率下的10秒长度）
        
        Returns:
            bool: 是否准备好
        """
        with self.lock:
            # 只有当所有启用的模态都填满了各自的缓冲区时，才认为Ready
            ready = True
            if self.imu_win_len > 0 and len(self.imu_buffer) < self.imu_win_len:
                ready = False
            if ready and self.eeg_win_len > 0 and len(self.eeg_buffer) < self.eeg_win_len:
                ready = False
            if ready and self.ecg_win_len > 0 and len(self.ecg_buffer) < self.ecg_win_len:
                ready = False
            
            return ready
    
    def get_window(self) -> Dict[str, np.ndarray]:
        """
        获取当前窗口数据
        
        Returns:
            dict: {'imu': (channels, imu_win_len), 'eeg': (...), 'ecg': (...)}
        """
        with self.lock:
            window_data = {}
            
            if len(self.imu_buffer) >= self.imu_win_len:
                imu_list = list(self.imu_buffer)
                window_data['imu'] = np.concatenate(imu_list, axis=1)
                window_data['imu_timestamps'] = np.array(list(self.imu_timestamps))
            
            if len(self.eeg_buffer) >= self.eeg_win_len:
                eeg_list = list(self.eeg_buffer)
                window_data['eeg'] = np.concatenate(eeg_list, axis=1)
                window_data['eeg_timestamps'] = np.array(list(self.eeg_timestamps))
            
            if len(self.ecg_buffer) >= self.ecg_win_len:
                ecg_list = list(self.ecg_buffer)
                window_data['ecg'] = np.concatenate(ecg_list, axis=1)
                window_data['ecg_timestamps'] = np.array(list(self.ecg_timestamps))
            
            return window_data
    
    def reset(self):
        """重置缓冲区"""
        with self.lock:
            self.imu_buffer.clear()
            self.eeg_buffer.clear()
            self.ecg_buffer.clear()
            self.imu_timestamps.clear()
            self.eeg_timestamps.clear()
            self.ecg_timestamps.clear()
            self.total_frames_received = 0
    
    def get_buffer_status(self) -> Dict:
        """获取缓冲区状态信息"""
        with self.lock:
            return {
                'imu_length': len(self.imu_buffer),
                'imu_target': self.imu_win_len,
                'eeg_length': len(self.eeg_buffer),
                'eeg_target': self.eeg_win_len,
                'ecg_length': len(self.ecg_buffer),
                'ecg_target': self.ecg_win_len,
                'is_ready': self.is_ready(),
                'total_frames_received': self.total_frames_received
            }
