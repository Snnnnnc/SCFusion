"""
数据处理流水线模块
将原始数据转换为模型输入格式

复用 data/preprocessing/preprocess_raw.py 中的核心处理逻辑
"""

import numpy as np
import sys
import os
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from scipy import signal as scipy_signal
    from scipy.signal import butter, filtfilt, iirnotch
except ImportError:
    scipy_signal = None
    butter = filtfilt = iirnotch = None
    print("⚠️  警告: scipy未安装，将使用简单抽取进行下采样，且无法进行滤波")

try:
    from models.vestibular_model import VestibularModel
except ImportError:
    VestibularModel = None
    print("⚠️  警告: VestibularModel未找到，无法计算IMU冲突")


def downsample_signal(arr: np.ndarray, orig_srate: int, target_srate: int, axis: int = -1, is_events: bool = False) -> np.ndarray:
    """
    下采样信号数据（复用 preprocess_raw.py 中的 downsample 逻辑）
    
    Args:
        arr: 输入数组 (channels, time_points) 或 (time_points,)
        orig_srate: 原始采样率
        target_srate: 目标采样率
        axis: 下采样的轴（默认-1，即时间轴）
        is_events: 是否为events数据（离散整数编码），如果是则使用最近邻插值
    
    Returns:
        下采样后的数组
    
    参考: data/preprocessing/preprocess_raw.py 中的 downsample() 函数
    """
    if orig_srate == target_srate:
        return arr
    
    # 计算目标长度（与resample保持一致）
    num_samples = int(arr.shape[axis] * target_srate / orig_srate)
    
    if is_events:
        # events是离散整数编码，使用最近邻索引映射
        # 这样可以保持长度与信号一致，同时保持离散编码值不变
        T_orig = arr.shape[axis]
        # 计算目标索引对应的原始索引（最近邻）
        target_indices = np.arange(num_samples)
        # 计算对应的原始索引位置（浮点）
        orig_positions = target_indices * (T_orig - 1) / (num_samples - 1) if num_samples > 1 else np.array([0])
        # 四舍五入到最近的整数索引
        orig_indices = np.round(orig_positions).astype(int)
        # 确保索引在有效范围内
        orig_indices = np.clip(orig_indices, 0, T_orig - 1)
        
        # 使用索引提取值
        if axis == -1:
            # 最简单的情况：axis是最后一个维度
            result = arr[..., orig_indices]
        else:
            # 需要转置处理
            arr_swapped = np.swapaxes(arr, axis, -1)
            result_swapped = arr_swapped[..., orig_indices]
            result = np.swapaxes(result_swapped, axis, -1)
        
        return result.astype(arr.dtype)
    
    if scipy_signal is None:
        # 简单抽取（抗混叠较差，但可用）
        factor = orig_srate // target_srate
        if factor > 1:
            if axis == -1:
                result = arr[..., ::factor]
            else:
                slices = [slice(None)] * arr.ndim
                slices[axis] = slice(None, None, factor)
                result = arr[tuple(slices)]

            # 截断到目标长度
            if result.shape[axis] > num_samples:
                if axis == -1:
                    result = result[..., :num_samples]
                else:
                    slices = [slice(None)] * arr.ndim
                    slices[axis] = slice(None, num_samples)
                    result = result[tuple(slices)]

            return result
        else:
            # 上采样情况，使用线性插值（简单实现）
            # 对于上采样，使用scipy.signal.resample更合适
            raise ValueError(f"上采样 ({orig_srate}Hz -> {target_srate}Hz) 需要 scipy")
    
    # 使用resample（抗混叠更好，适用于连续信号）
    return scipy_signal.resample(arr, num_samples, axis=axis)


def compute_vestibular_conflicts(gyro_dps: np.ndarray, acc_G: np.ndarray, dt: float, 
                                 session_name: str = "", validate: bool = True) -> np.ndarray:
    """
    计算前庭模型冲突数据（复用 preprocess_raw.py 中的 compute_vestibular_conflicts 逻辑）
    
    Args:
        gyro_dps: (N, 3) 角速度 (deg/s)
        acc_G: (N, 3) 加速度 (G单位)
        dt: 时间步长（秒）
        session_name: 会话名称（用于错误报告）
        validate: 是否进行数据验证
    
    Returns:
        conflicts: (12, N) - 冲突数据（e_scc, e_oto, e_v, k_out各3维）
    
    参考: data/preprocessing/preprocess_raw.py 中的 compute_vestibular_conflicts() 函数
    """
    if VestibularModel is None:
        raise RuntimeError("需要 models.vestibular_model.VestibularModel。请确保模型文件存在。")
    
    # 验证输入数据（可选）
    if validate:
        # 检查基本形状
        if gyro_dps.shape[1] != 3 or acc_G.shape[1] != 3:
            raise ValueError(f"角速度和加速度数据形状错误: gyro {gyro_dps.shape}, acc {acc_G.shape}")
        if len(gyro_dps) != len(acc_G):
            raise ValueError(f"角速度和加速度数据长度不一致: {len(gyro_dps)} vs {len(acc_G)}")
        
        # 检查NaN/Inf
        if np.isnan(gyro_dps).any() or np.isinf(gyro_dps).any():
            raise ValueError(f"角速度数据包含NaN/Inf，无法继续计算conflicts。会话: {session_name}")
        if np.isnan(acc_G).any() or np.isinf(acc_G).any():
            raise ValueError(f"加速度数据包含NaN/Inf，无法继续计算conflicts。会话: {session_name}")
    
    vestibular_model = VestibularModel(dt=dt)
    vestibular_model.reset()
    
    N = len(gyro_dps)
    conflicts = {
        'e_scc': [],
        'e_oto': [],
        'e_v': [],
        'k_out': []
    }
    
    # 计算conflicts，逐帧处理
    for i in range(N):
        result = vestibular_model.step(
            acc_world_G=acc_G[i],        # (3,) G单位
            gyro_head_dps=gyro_dps[i],   # (3,) deg/s
            gravity_switch=0
        )
        conflicts['e_scc'].append(result['e_scc'])
        conflicts['e_oto'].append(result['e_oto'])
        conflicts['e_v'].append(result['e_v'])
        conflicts['k_out'].append(result['k_out'])
    
    # 转换为numpy数组并组合
    e_scc = np.array(conflicts['e_scc'])  # (N, 3)
    e_oto = np.array(conflicts['e_oto'])  # (N, 3)
    e_v = np.array(conflicts['e_v'])       # (N, 3)
    k_out = np.array(conflicts['k_out'])  # (N, 3)
    
    # 组合为 (12, N)
    vestibular_conflicts = np.vstack([
        e_scc.T,    # (3, N)
        e_oto.T,    # (3, N)
        e_v.T,      # (3, N)
        k_out.T     # (3, N)
    ])  # (12, N)
    
    # 清理NaN/Inf（防止传播）
    vestibular_conflicts = np.nan_to_num(vestibular_conflicts, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 验证输出数据（可选）
    if validate:
        # 检查输出形状
        if vestibular_conflicts.shape[0] != 12:
            raise ValueError(f"冲突数据通道数错误: 期望12，实际{vestibular_conflicts.shape[0]}")
        
        # 检查NaN/Inf（虽然已经清理，但再次确认）
        if np.isnan(vestibular_conflicts).any() or np.isinf(vestibular_conflicts).any():
            print(f"⚠️  警告: 冲突数据包含NaN/Inf（已清理为0），会话: {session_name}")
    
    return vestibular_conflicts


def normalize_data(data: np.ndarray, normalization_stats: Optional[Dict], modality: str, 
                   subject_id: Optional[str] = None) -> np.ndarray:
    """
    归一化数据。
    默认逻辑：
    1. 如果提供了有效的 normalization_stats，尝试进行全局或按被试归一化。
    2. 如果统计量无效（None、全0/1、维度不匹配），则强制执行“自归一化”（样本内归一化）。
    """
    num_ch = data.shape[-2] if len(data.shape) >= 2 else data.shape[0]

    def _apply_self_normalization(x):
        # 强制自归一化：使当前数据均值为0，标准差为1
        reshaped = x.reshape(num_ch, -1)
        m = reshaped.mean(axis=1)
        s = reshaped.std(axis=1) + 1e-8
        if len(x.shape) == 3:
            return (x - m.reshape(1, -1, 1)) / (s.reshape(1, -1, 1))
        return (x - m.reshape(-1, 1)) / (s.reshape(-1, 1))

    # 如果完全没有提供统计量，直接自归一化
    if normalization_stats is None:
        return _apply_self_normalization(data)
    
    mean, std = None, None
    try:
        # 判定是否为按被试区分的字典
        first_key = next(iter(normalization_stats.keys()))
        is_per_subject = isinstance(normalization_stats[first_key], dict) and \
                         not any(m in normalization_stats for m in ['imu', 'eeg', 'ecg'])

        if is_per_subject and subject_id is not None:
            # 尝试匹配被试
            s_dict = normalization_stats.get(subject_id)
            if s_dict is None and str(subject_id).isdigit():
                s_dict = normalization_stats.get(int(subject_id))
            if s_dict:
                stats = s_dict.get(modality, {})
                mean, std = stats.get('mean'), stats.get('std')
        else:
            # 全局模式
            stats = normalization_stats.get(modality, {})
            mean, std = stats.get('mean'), stats.get('std')
            
    except Exception:
        pass

    # 核心判断：如果提取到的统计量无效或全是占位符（0/1），则回退到自归一化
    is_dummy = False
    if mean is not None and std is not None:
        # 检查是否为标量 0/1 或 数组全 0/1
        m_val = np.mean(mean) if isinstance(mean, np.ndarray) else mean
        s_val = np.mean(std) if isinstance(std, np.ndarray) else std
        if m_val == 0.0 and s_val == 1.0:
            is_dummy = True

    if mean is None or std is None or is_dummy or (isinstance(mean, np.ndarray) and mean.shape[0] != num_ch):
        # 只要不满足条件，就执行自归一化保底
        return _apply_self_normalization(data)
    
    # 执行正规归一化
    if len(data.shape) == 3:
        m_exp, s_exp = mean.reshape(1, -1, 1), std.reshape(1, -1, 1)
    else:
        m_exp, s_exp = mean.reshape(-1, 1), std.reshape(-1, 1)
        
    s_exp = np.where(s_exp < 1e-8, 1.0, s_exp)
    return (data - m_exp) / s_exp


def create_patches_from_window(window_data: np.ndarray, patch_length: int) -> np.ndarray:
    """
    将窗口数据切分成patches（复用 build_training_dataset.py 中的 create_patches_fast 逻辑）
    
    Args:
        window_data: (channels, window_length) 或 (num_windows, channels, window_length)
        patch_length: patch长度（采样点数）
    
    Returns:
        patches: (num_patches, channels, patch_length) 或 (num_windows, num_patches, channels, patch_length)
    
    参考: data/preprocessing/build_training_dataset.py 中的 create_patches_fast() 函数
    """
    if len(window_data.shape) == 2:
        # 单个窗口: (channels, window_length)
        channels, window_length = window_data.shape
        num_patches = window_length // patch_length

        if num_patches == 0:
            raise ValueError(f"窗口长度 ({window_length}) 必须大于等于patch长度 ({patch_length})")

        # 检查是否能整除
        if window_length % patch_length != 0:
            raise ValueError(
                f"window_length ({window_length}) 必须能被 patch_length ({patch_length}) 整除。"
                f"当前 window={window_length}点, patch={patch_length}点"
            )

        # 使用reshape+transpose实现零拷贝（或近似零拷贝）
        # (channels, window_length) -> (channels, num_patches, patch_length) -> (num_patches, channels, patch_length)
        patches = (
            window_data[:, : num_patches * patch_length]
            .reshape(channels, num_patches, patch_length)
            .transpose(1, 0, 2)
        )

        return patches

    elif len(window_data.shape) == 3:
        # 批量窗口: (num_windows, channels, window_length)
        num_windows, channels, window_length = window_data.shape
        num_patches = window_length // patch_length

        if num_patches == 0:
            raise ValueError(f"窗口长度 ({window_length}) 必须大于等于patch长度 ({patch_length})")

        # 检查是否能整除
        if window_length % patch_length != 0:
            raise ValueError(f"window_length ({window_length}) 必须能被 patch_length ({patch_length}) 整除。")

        # 批量处理: (num_windows, channels, window_length) -> (num_windows, channels, num_patches, patch_length) -> (num_windows, num_patches, channels, patch_length)
        patches = (
            window_data[:, :, : num_patches * patch_length]
            .reshape(num_windows, channels, num_patches, patch_length)
            .transpose(0, 2, 1, 3)
        )

        return patches

    else:
        raise ValueError(f"window_data 形状错误: 期望2D或3D，实际{window_data.shape}")


# --- 信号预处理工具函数 (移植自 preprocess_raw.py) ---

def remove_dc(arr: np.ndarray) -> np.ndarray:
    """去直流：减去中位数"""
    med = np.median(arr, axis=1, keepdims=True)
    return arr - med


def apply_notch_bandpass(arr: np.ndarray, srate: int, notch_freq_hz: float, notch_q: float, 
                         lo_hz: float, hi_hz: float, order: int) -> np.ndarray:
    """应用陷波器和带通滤波器"""
    if butter is None or filtfilt is None:
        return arr  # 如果没有scipy，直接返回
    
    x = arr.copy()
    if notch_freq_hz and iirnotch is not None:
        w0 = notch_freq_hz / (srate / 2.0)
        b, a = iirnotch(w0, notch_q)
        x = filtfilt(b, a, x, axis=1)
    
    lo = lo_hz / (srate / 2.0)
    hi = hi_hz / (srate / 2.0)
    b, a = butter(order, [lo, hi], btype="bandpass")
    x = filtfilt(b, a, x, axis=1)
    return x


def detect_bad_channels(eeg: np.ndarray) -> list:
    """简单坏通道检测"""
    if eeg.shape[1] < 2:
        return []
    diffs = np.abs(np.diff(eeg, axis=1))
    flat = np.mean(diffs < 1e-9, axis=1) > 0.98
    var = np.var(eeg, axis=1)
    # 避免空数组或全NaN导致的问题
    if len(var) == 0 or np.isnan(var).all():
        return []
    lo, hi = np.percentile(var, [5, 95])
    bad_var = (var < (0.1 * lo)) | (var > (3.0 * hi))
    bad = np.where(flat | bad_var)[0].tolist()
    return bad


def asr_like_clean(eeg: np.ndarray, srate: int, z_thresh: float = 5.0) -> np.ndarray:
    """ASR近似：抑制瞬时大伪迹"""
    x = eeg.copy()
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True) + 1e-9
    z = (x - med) / (1.4826 * mad)
    # 软阈函数：保留中心，压缩尾部
    gain = np.tanh(z / z_thresh) / (z / z_thresh + 1e-12)
    x_clean = med + (x - med) * np.clip(gain, 0.0, 1.0)
    return x_clean


def remove_eog_artifacts_via_regression(eeg: np.ndarray, eog: np.ndarray) -> np.ndarray:
    """使用EOG线性回归移除眼动分量"""
    if eog is None or eog.size == 0:
        return eeg
    
    # 确保长度一致
    if eog.shape[1] != eeg.shape[1]:
        min_len = min(eog.shape[1], eeg.shape[1])
        eog = eog[:, :min_len]
        eeg = eeg[:, :min_len]
        
    X = eog.T  # [T, k]
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # 加偏置
    Y = eeg.T  # [T, ch]
    XT = X.T                    # [k+1, T]
    XtX = XT @ X               # [k+1, k+1]
    XtY = XT @ Y               # [k+1, ch]
    reg = 1e-6 * np.eye(XtX.shape[0], dtype=X.dtype)
    try:
        beta = np.linalg.pinv(XtX + reg) @ XtY   # [k+1, ch]
        Y_hat = X @ beta                         # [T, ch]
        Y_clean = Y - Y_hat
        return Y_clean.T
    except np.linalg.LinAlgError:
        return eeg


def interpolate_bad_channels(eeg: np.ndarray, bad_idx: list) -> np.ndarray:
    """插值坏通道"""
    if not bad_idx:
        return eeg
    good_idx = [i for i in range(eeg.shape[0]) if i not in bad_idx]
    if not good_idx:
        return eeg
    ref = np.mean(eeg[good_idx, :], axis=0, keepdims=True)
    out = eeg.copy()
    out[bad_idx, :] = ref
    return out


class DataProcessor:
    """数据处理流水线"""
    
    def __init__(self, 
                 config,
                 normalization_stats: Optional[Dict] = None):
        """
        初始化数据处理流水线
        
        Args:
            config: 部署配置对象
            normalization_stats: 归一化统计量字典
        """
        self.config = config
        self.normalization_stats = normalization_stats
        
        self.target_sampling_rate = config.target_sampling_rate
        self.window_length = config.window_length
        self.patch_length = config.patch_length
        self.num_patches = config.num_patches
    
    def process_window(self, window_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        处理一个10秒窗口数据
        
        Args:
            window_data: 窗口数据字典 {'imu': (channels, window_length), 'eeg': (...), 'ecg': (...)}
        
        Returns:
            patches_dict: patches格式数据 {'imu': (10, channels, 250), 'eeg': (...), 'ecg': (...)}
        """
        patches_dict = {}
        
        # 处理IMU数据
        if 'imu' in window_data:
            imu_window = window_data['imu'].copy()
            
            # 1. 采样率对齐（如果需要）
            if self.config.imu_sampling_rate != self.target_sampling_rate:
                imu_window = downsample_signal(
                    imu_window, 
                    self.config.imu_sampling_rate, 
                    self.target_sampling_rate,
                    axis=1
                )
            
            # 2. 计算IMU冲突（如果输入是6维原始数据）
            if imu_window.shape[0] == 6:
                # 提取角速度和加速度
                gyro_dps = imu_window[0:3, :].T  # (N, 3)
                acc_G = imu_window[3:6, :].T     # (N, 3)
                
                # 计算冲突
                dt = 1.0 / self.target_sampling_rate
                conflicts = compute_vestibular_conflicts(
                    gyro_dps, acc_G, dt, 
                    session_name=getattr(self.config, 'session_name', 'deployment'),
                    validate=True
                )  # (12, N)
                
                # 合并为18维: [gyro(3), acc(3), conflicts(12)]
                imu_window = np.vstack([imu_window, conflicts])  # (18, N)
            elif imu_window.shape[0] != 18:
                raise ValueError(f"IMU数据维度错误: 期望6或18维，实际{imu_window.shape[0]}维")
            
            # 3. 归一化
            # 注意：如果使用per-subject归一化，需要提供subject_id
            subject_id = getattr(self.config, 'subject_id', None)
            imu_window = normalize_data(imu_window, self.normalization_stats, 'imu', subject_id=subject_id)
            
            # 调试输出：打印归一化后的统计量
            if getattr(self.config, 'verbose', True):
                print(f"  [DataProcessor] IMU norm mean: {imu_window.mean():.4f} (expected ~0)")
            
            # 4. 创建patches
            imu_patches = create_patches_from_window(imu_window, self.patch_length)
            patches_dict['imu'] = imu_patches
        
        # 处理EEG数据
        if 'eeg' in window_data:
            eeg_window = window_data['eeg'].copy()
            eog_data = window_data.get('eog', None)
            
            # 1. 信号预处理 (参考 preprocess_raw.py)
            srate = self.config.eeg_sampling_rate
            
            # 去直流
            eeg_window = remove_dc(eeg_window)
            
            # 陷波和带通滤波
            eeg_window = apply_notch_bandpass(
                eeg_window, srate, 
                notch_freq_hz=50.0, notch_q=30.0, 
                lo_hz=0.5, hi_hz=45.0, order=4
            )
            
            # 坏通道检测与插值
            bad_idx = detect_bad_channels(eeg_window)
            
            # ASR去伪迹
            eeg_window = asr_like_clean(eeg_window, srate, z_thresh=5.0)
            
            # EOG回归
            if eog_data is not None:
                eeg_window = remove_eog_artifacts_via_regression(eeg_window, eog_data)
                
            # 插值坏通道
            eeg_window = interpolate_bad_channels(eeg_window, bad_idx)
            
            # 2. 采样率对齐（如果需要）
            if self.config.eeg_sampling_rate != self.target_sampling_rate:
                eeg_window = downsample_signal(
                    eeg_window,
                    self.config.eeg_sampling_rate,
                    self.target_sampling_rate,
                    axis=1
                )
            
            # 3. 归一化
            subject_id = getattr(self.config, 'subject_id', None)
            eeg_window = normalize_data(eeg_window, self.normalization_stats, 'eeg', subject_id=subject_id)
            
            # 调试输出：打印归一化后的统计量
            if getattr(self.config, 'verbose', True):
                print(f"  [DataProcessor] EEG norm mean: {eeg_window.mean():.4f} (expected ~0)")
            
            # 4. 创建 patches
            eeg_patches = create_patches_from_window(eeg_window, self.patch_length)
            patches_dict['eeg'] = eeg_patches
        
        # 处理ECG数据
        if 'ecg' in window_data:
            ecg_window = window_data['ecg'].copy()
            
            # 1. 信号预处理 (参考 preprocess_raw.py)
            srate = self.config.ecg_sampling_rate
            
            # 去直流
            ecg_window = remove_dc(ecg_window)
            
            # 陷波和带通滤波
            ecg_window = apply_notch_bandpass(
                ecg_window, srate, 
                notch_freq_hz=50.0, notch_q=30.0, 
                lo_hz=0.5, hi_hz=40.0, order=4
            )
            
            # 2. 采样率对齐（如果需要）
            if self.config.ecg_sampling_rate != self.target_sampling_rate:
                ecg_window = downsample_signal(
                    ecg_window,
                    self.config.ecg_sampling_rate,
                    self.target_sampling_rate,
                    axis=1
                )
            
            # 3. 归一化
            subject_id = getattr(self.config, 'subject_id', None)
            ecg_window = normalize_data(ecg_window, self.normalization_stats, 'ecg', subject_id=subject_id)
            
            # 4. 创建 patches
            ecg_patches = create_patches_from_window(ecg_window, self.patch_length)
            patches_dict['ecg'] = ecg_patches
        
        return patches_dict
