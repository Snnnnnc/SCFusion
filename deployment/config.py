"""
部署配置模块
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeploymentConfig:
    """部署配置类"""
    
    # 模型相关
    checkpoint_path: str
    normalization_stats_path: Optional[str] = None
    mode: Optional[str] = None  # 自动推断或手动指定
    device: str = 'cuda:0'  # 或 'cpu'
    
    # 数据窗口参数
    window_length: int = 2500  # 10秒@250Hz
    hop_length: int = 2500  # 窗口步长（默认不重叠）
    patch_length: int = 250  # 1秒@250Hz
    num_patches: int = 10  # 窗口内的patch数量
    target_sampling_rate: int = 250  # 目标采样率（Hz）
    
    # 数据采集参数
    imu_sampling_rate: int = 100  # IMU原始采样率（Hz）
    eeg_sampling_rate: int = 250  # EEG原始采样率（Hz）
    ecg_sampling_rate: int = 250  # ECG原始采样率（Hz）
    
    # 数据通道数
    imu_channels: int = 18  # IMU通道数（6原始 + 12冲突）
    eeg_channels: int = 59  # EEG通道数
    ecg_channels: int = 1  # ECG通道数
    
    # 传感器配置（用于模拟数据生成或原始数据回放）
    use_simulated_data: bool = False  # 是否使用模拟随机数据
    use_original_data: bool = False   # 是否使用真实的原始BDF数据
    subject_id: Optional[str] = None  # 被试ID
    subject_filter: Optional[str] = None
    map_filter: Optional[str] = None
    data_root: str = "data"
    # 原始数据回放：允许直接指定某个 bdf 文件（来自前端文件选择）
    bdf_path: Optional[str] = None

    # 与车端对齐：可选的“统一起跑”绝对时间戳（Unix epoch seconds）
    # 若设置，数据回放与预测会以该时刻为 t=0（倒计时由前端/车端共同遵守）
    start_at: Optional[float] = None
    
    simulated_data_config: Optional[dict] = None
    
    # 日志和输出
    log_dir: str = "./deployment_logs"
    output_dir: str = "./deployment_output"
    verbose: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 验证路径
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint文件不存在: {self.checkpoint_path}")
        
        if self.normalization_stats_path and not os.path.exists(self.normalization_stats_path):
            print(f"⚠️  警告: 归一化统计量文件不存在: {self.normalization_stats_path}")
