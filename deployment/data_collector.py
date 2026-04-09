"""
数据采集接口模块
从硬件传感器获取IMU和生理信号数据
"""

import numpy as np
import time
import os
from typing import Optional, Dict, Tuple, List
from abc import ABC, abstractmethod
from pathlib import Path

# 尝试导入 BDF 读取库
try:
    from neuracle_lib.readbdfdata import readbdfdata
except ImportError:
    import sys
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _eeg_trigger_path = os.path.join(_project_root, 'EEG_trigger')
    if os.path.exists(_eeg_trigger_path):
        sys.path.insert(0, _eeg_trigger_path)
    try:
        from neuracle_lib.readbdfdata import readbdfdata
    except ImportError:
        readbdfdata = None


class DataCollector(ABC):
    """数据采集基类"""
    @abstractmethod
    def initialize(self) -> bool: pass
    @abstractmethod
    def get_latest_data(self) -> Dict[str, np.ndarray]: pass


class OriginalDataCollector(DataCollector):
    """原始数据采集器：读取 BDF 文件并回放"""
    
    def __init__(self, data_root: str, subject: str, map_id: str, 
                 imu_sampling_rate: int = 100, eeg_sampling_rate: int = 250, ecg_sampling_rate: int = 250,
                 bdf_path: Optional[str] = None):
        self.data_root = Path(data_root)
        self.subject = subject
        self.map_id = map_id
        self.srates = {'imu': imu_sampling_rate, 'eeg': eeg_sampling_rate, 'ecg': ecg_sampling_rate}
        self.bdf_path = Path(bdf_path).expanduser() if bdf_path else None
        
        self.initialized = False
        self.data = {'imu': None, 'eeg': None, 'ecg': None}
        self.pointers = {'imu': 0, 'eeg': 0, 'ecg': 0}
        self.start_time = None
        
    def _split_modalities(self, ch_names: List[str]) -> Dict[str, List[int]]:
        """
        拆分通道。强制 IMU 顺序 [GyroX,Y,Z, AccX,Y,Z]，强制 EEG 59 维。
        """
        ch_names_up = [n.upper() for n in ch_names]
        eog_set = {'HEOR', 'HEOL', 'VEOU', 'VEOL'}
        ecg_set = {'ECG'}
        ignore_set = {'MAG-X', 'MAG-Y', 'MAG-Z', 'MAGX', 'MAGY', 'MAGZ', 'TRIG', 'STATUS', 'TRG'}
        imu_order = ['GYR-X', 'GYR-Y', 'GYR-Z', 'ACC-X', 'ACC-Y', 'ACC-Z']
        
        idxs = {'eeg': [], 'eog': [], 'ecg': [], 'imu': []}
        
        # 1. 寻找 IMU (按顺序)
        for target in imu_order:
            found = False
            for i, name in enumerate(ch_names_up):
                if target == name or target.replace('-', '') == name:
                    idxs['imu'].append(i)
                    found = True
                    break
            if not found: print(f"⚠️ 警告: 未找到 IMU 通道 {target}")
            
        # 2. 寻找 ECG, EOG
        for i, name in enumerate(ch_names_up):
            if name in ecg_set: idxs['ecg'].append(i)
            elif name in eog_set: idxs['eog'].append(i)
            
        # 3. 其余归为 EEG (排除已识别和忽略的)
        identified = set(idxs['imu']) | set(idxs['ecg']) | set(idxs['eog'])
        for i, name in enumerate(ch_names_up):
            if i in identified or name in ignore_set: continue
            idxs['eeg'].append(i)
            
        # 4. 强制裁剪 EEG 到 59 维
        if len(idxs['eeg']) > 59:
            print(f"⚠️ 警告: EEG 通道数 ({len(idxs['eeg'])}) > 59，执行裁剪")
            idxs['eeg'] = idxs['eeg'][:59]
            
        return idxs

    def initialize(self) -> bool:
        if readbdfdata is None: return False
        
        # 1) 如果前端/调用方直接指定了 bdf_path，优先使用
        if self.bdf_path is not None:
            if not self.bdf_path.exists():
                print(f"❌ 指定的 BDF 文件不存在: {self.bdf_path}")
                return False
            bdf_path = self.bdf_path
            session_dir = bdf_path.parent
        else:
            # 2) 否则按 subject/map 自动定位 session 目录
            session_dir = None
            for d in self.data_root.iterdir():
                if d.is_dir() and f"_{self.subject}_" in d.name and f"_{self.map_id}_" in d.name:
                    session_dir = d
                    break
            if not session_dir:
                print("❌ 未找到匹配的 session 目录（请检查 subject/map）")
                return False

            # 3) 在 session 内选择 BDF：优先 data.bdf；其次任意非 evt 的 .bdf；最后才是 evt.bdf
            candidates = list(session_dir.rglob("*.bdf"))
            if not candidates:
                print("❌ session 目录下未找到任何 .bdf 文件")
                return False

            def _rank(p: Path) -> tuple:
                name = p.name.lower()
                if name == "data.bdf":
                    return (0, name)
                if name.endswith("evt.bdf") or name.startswith("evt"):
                    return (2, name)
                return (1, name)

            candidates.sort(key=_rank)
            bdf_path = candidates[0]
        
        if bdf_path.name.lower().startswith("evt") or bdf_path.name.lower().endswith("evt.bdf"):
            print(f"⚠️ 你选择/匹配到的是 {bdf_path.name}，这通常是事件文件，可能没有完整信号数据；建议优先选 data.bdf")
        
        # 读取
        res = readbdfdata([bdf_path.name], [str(bdf_path.parent)])
        raw_data, events, ch_names, srate = res['data'], res['events'], res['ch_names'], res['srate']
        
        # 寻找开始点 (Event 100)
        start_idx = 0
        if events is not None and events.size > 0:
            starts = events[events[:, 2] == 100]
            if starts.size > 0: start_idx = int(starts[-1, 0])
        
        trimmed = raw_data[:, start_idx:]
        idxs = self._split_modalities(ch_names)
        
        # 提取并重采样 (IMU 从 1000Hz 降到 100Hz)
        if idxs['imu']:
            imu = trimmed[idxs['imu'], :]
            self.data['imu'] = imu[:, ::10] if srate == 1000 else imu
        if idxs['eeg']:
            self.data['eeg'] = trimmed[idxs['eeg'], :]
        if idxs['ecg']:
            self.data['ecg'] = trimmed[idxs['ecg'], :]
            
        self.initialized = True
        self.start_time = time.time()
        print(f"✓ 原始数据回放初始化成功: EEG={self.data['eeg'].shape[0]}ch, IMU={self.data['imu'].shape[0]}ch")
        return True
    
    def get_latest_data(self) -> Dict[str, np.ndarray]:
        if not self.initialized: return {}
        elapsed = time.time() - self.start_time
        res = {}
        for mod in ['imu', 'eeg', 'ecg']:
            if self.data[mod] is not None:
                # ⚠️ 极其关键：必须使用物理采样率来对齐回放速度
                # EEG/ECG 原始是 1000Hz，IMU 降采样后是 100Hz
                sr = 100 if mod == 'imu' else 1000
                target = int(elapsed * sr)
                start = self.pointers[mod]
                if target > start:
                    end = min(target, self.data[mod].shape[1])
                    res[mod] = self.data[mod][:, start:end]
                    self.pointers[mod] = end
        return res


class SimulatedDataCollector(DataCollector):
    def __init__(self, **kwargs):
        self.srates = {k: kwargs.get(f"{k}_sampling_rate", 250) for k in ['imu', 'eeg', 'ecg']}
        self.channels = {'imu': 6, 'eeg': 59, 'ecg': 1}
        self.start_time = time.time()
        self.last_time = {'imu': 0.0, 'eeg': 0.0, 'ecg': 0.0}

    def initialize(self): return True
    
    def get_latest_data(self) -> Dict[str, np.ndarray]:
        elapsed = time.time() - self.start_time
        res = {}
        for mod in ['imu', 'eeg', 'ecg']:
            dt = 1.0 / self.srates[mod]
            ts = np.arange(self.last_time[mod] + dt, elapsed + 1e-9, dt)
            if len(ts) > 0:
                res[mod] = np.random.randn(self.channels[mod], len(ts))
                self.last_time[mod] = ts[-1]
        return res


class HardwareDataCollector(DataCollector):
    def initialize(self): return True
    def get_latest_data(self): return {}
