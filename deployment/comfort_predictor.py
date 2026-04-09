"""
舒适度预测服务主模块
整合所有组件，提供统一的服务接口
使用多线程架构解决耗时处理导致的实时性漂移问题
"""

import os
import time
import pickle
import numpy as np
import threading
from typing import Dict, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from .config import DeploymentConfig
from .data_collector import DataCollector, SimulatedDataCollector, HardwareDataCollector, OriginalDataCollector
from .data_buffer import SlidingWindowBuffer
from .data_processor import DataProcessor
from .model_manager import ModelManager


class ComfortPredictor:
    """舒适度预测服务主类"""
    
    def __init__(self, config: DeploymentConfig):
        """
        初始化舒适度预测服务
        """
        # ⚠️ 极其关键：针对原始 BDF 回放模式，强制覆盖采样率
        if config.use_original_data:
            print("  [ComfortPredictor] ⚠️ 原始数据模式：强制设置频率 EEG/ECG=1000Hz, IMU=100Hz")
            config.eeg_sampling_rate = 1000
            config.ecg_sampling_rate = 1000
            config.imu_sampling_rate = 100
            
        self.config = config
        self.running = False
        
        # 1. 加载归一化统计量
        normalization_stats = None
        if config.normalization_stats_path and os.path.exists(config.normalization_stats_path):
            try:
                with open(config.normalization_stats_path, 'rb') as f:
                    normalization_stats = pickle.load(f)
                print(f"✓ 归一化统计量已加载")
            except Exception as e:
                print(f"⚠️ 警告: 加载统计量失败 ({e})，将使用样本内归一化")
        
        # 2. 初始化组件
        if config.use_original_data:
            self.collector = OriginalDataCollector(
                data_root=config.data_root,
                subject=config.subject_filter,
                map_id=config.map_filter,
                imu_sampling_rate=100,
                eeg_sampling_rate=1000,
                ecg_sampling_rate=1000,
                bdf_path=getattr(config, "bdf_path", None)
            )
        elif config.use_simulated_data:
            self.collector = SimulatedDataCollector(
                imu_sampling_rate=config.imu_sampling_rate,
                eeg_sampling_rate=config.eeg_sampling_rate,
                ecg_sampling_rate=config.ecg_sampling_rate
            )
        else:
            self.collector = HardwareDataCollector()
        
        # 缓冲区现在会正确计算 10s 所需的点数：
        # EEG: 1000Hz * 10s = 10000 points
        # IMU: 100Hz * 10s = 1000 points
        self.buffer = SlidingWindowBuffer(
            window_length=config.window_length,
            target_sampling_rate=config.target_sampling_rate,
            imu_channels=6,
            eeg_channels=config.eeg_channels,
            ecg_channels=config.ecg_channels,
            imu_sampling_rate=config.imu_sampling_rate,
            eeg_sampling_rate=config.eeg_sampling_rate,
            ecg_sampling_rate=config.ecg_sampling_rate
        )
        
        self.processor = DataProcessor(config, normalization_stats)
        self.model_manager = ModelManager(
            checkpoint_path=config.checkpoint_path,
            device=config.device,
            mode=config.mode
        )
        
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.is_processing = False
        self.on_prediction_callback: Optional[Callable] = None
        self.prediction_history = []
        # 供车端对齐：预测服务“已初始化(BDF已读完、模型已加载)”信号
        self.ready_event = threading.Event()

    def set_prediction_callback(self, callback: Callable):
        self.on_prediction_callback = callback

    def _async_predict_task(self, window_data: Dict[str, np.ndarray], wall_time: float):
        try:
            self.is_processing = True
            proc_start = time.time()
            
            # DataProcessor 内部会处理从 1000Hz/100Hz 到 250Hz 的重采样
            patches = self.processor.process_window(window_data)
            result = self.model_manager.predict(patches)
            result["event"] = "prediction"
            
            result['timestamp'] = wall_time
            result['proc_time_ms'] = (time.time() - proc_start) * 1000
            result['datetime'] = datetime.fromtimestamp(wall_time).isoformat()

            # 回放进度（0~1），用于前端进度条
            if self.config.use_original_data and hasattr(self.collector, "pointers") and hasattr(self.collector, "data"):
                try:
                    # 优先用 eeg 作为进度参考
                    if self.collector.data.get("eeg") is not None and self.collector.data["eeg"].shape[1] > 0:
                        result["progress"] = float(self.collector.pointers["eeg"]) / float(self.collector.data["eeg"].shape[1])
                    elif self.collector.data.get("imu") is not None and self.collector.data["imu"].shape[1] > 0:
                        result["progress"] = float(self.collector.pointers["imu"]) / float(self.collector.data["imu"].shape[1])
                except Exception:
                    pass
            
            self.prediction_history.append(result)
            if getattr(self.config, "verbose", True):
                # 降噪：仅输出每次预测的核心信息，不输出 emoji / 大段统计
                prog = result.get("progress", None)
                prog_str = f", progress={prog*100:.1f}%" if isinstance(prog, float) else ""
                print(
                    f"预测结果: score={result.get('score')} "
                    f"(conf={result.get('confidence', 0.0):.3f}), "
                    f"proc={result.get('proc_time_ms', 0.0):.1f}ms{prog_str}"
                )
            if self.on_prediction_callback:
                self.on_prediction_callback(result)
        except Exception as e:
            print(f"异步预测失败: {e}")
        finally:
            self.is_processing = False

    def start(self, collect_interval: float = 0.05):
        if not self.collector.initialize():
            print("❌ 数据源初始化失败")
            return
        
        # 初始化完成：此时 BDF 已读取、缓冲区/处理器/模型均已就绪
        self.ready_event.set()
        
        self.running = True
        predict_interval = 10.0 # 强制 10s 预测一次
        
        print(f"\n{'='*60}")
        print("舒适度预测服务已启动 (频率与时间轴对齐模式)")
        print(f"原始频率: EEG/ECG={self.config.eeg_sampling_rate}Hz, IMU={self.config.imu_sampling_rate}Hz")
        print(f"目标频率: 250Hz (窗口 {self.config.window_length} 点)")
        print(f"{'='*60}\n")

        # 若要求与车端对齐，则必须等 start_at 下发后再开跑
        if self.config.use_original_data:
            while self.running and getattr(self.config, "start_at", None) is None:
                time.sleep(0.05)

        # start_at 对齐：第一条预测在 start_at + 10s（避免 config.start_at 为 None 时 float(None) 报错）
        _start_at = getattr(self.config, "start_at", None)
        start_wall_time = float(_start_at) if _start_at is not None else time.time()
        next_predict_time = start_wall_time + predict_interval

        # 将回放起点对齐到 start_at（否则会在 start_at 已过去时直接快进）
        if self.config.use_original_data and hasattr(self.collector, "start_time"):
            try:
                self.collector.start_time = start_wall_time
            except Exception:
                pass

        # 进度心跳（用于前端按钮进度条实时变化，而不必等到每 10s 一次预测）
        last_progress_push = 0.0
        progress_push_interval = 0.5  # 秒
        last_busy_warn = 0.0
        
        try:
            while self.running:
                data = self.collector.get_latest_data()
                if data:
                    self.buffer.add_frames(imu=data.get('imu'), eeg=data.get('eeg'), ecg=data.get('ecg'))

                # 高频推送进度（不带 score），不打印日志
                if self.config.use_original_data and self.on_prediction_callback:
                    now = time.time()
                    if now - last_progress_push >= progress_push_interval:
                        last_progress_push = now
                        try:
                            prog = 0.0
                            if hasattr(self.collector, "pointers") and hasattr(self.collector, "data"):
                                if self.collector.data.get("eeg") is not None and self.collector.data["eeg"].shape[1] > 0:
                                    prog = float(self.collector.pointers["eeg"]) / float(self.collector.data["eeg"].shape[1])
                                elif self.collector.data.get("imu") is not None and self.collector.data["imu"].shape[1] > 0:
                                    prog = float(self.collector.pointers["imu"]) / float(self.collector.data["imu"].shape[1])
                            self.on_prediction_callback(
                                {
                                    "event": "progress",
                                    "progress": float(prog),
                                    "timestamp": now,
                                    "datetime": datetime.fromtimestamp(now).isoformat(),
                                    "proc_time_ms": 0.0,
                                }
                            )
                        except Exception:
                            pass
                
                current_time = time.time()
                if current_time >= next_predict_time:
                    if self.buffer.is_ready():
                        if not self.is_processing:
                            try:
                                window_snapshot = self.buffer.get_window()
                                self.executor.submit(self._async_predict_task, window_snapshot, current_time)
                                next_predict_time += predict_interval
                            except RuntimeError as e:
                                # executor 已 shutdown，说明服务正在停止
                                if "shutdown" in str(e).lower():
                                    break
                                raise
                        else:
                            # 忙时不刷屏、不推进 next_predict_time，等待当前预测完成后再触发
                            if current_time - last_busy_warn > 5.0:
                                print("警告: 计算中，暂缓本次预测触发")
                                last_busy_warn = current_time
                    else:
                        # 数据还没填满，可能是刚开始，或者采集太慢
                        pass 
                
                # 检查是否结束
                if self.config.use_original_data:
                    if hasattr(self.collector, 'pointers') and hasattr(self.collector, 'data'):
                        finished = True
                        for mod in ['imu', 'eeg', 'ecg']:
                            if self.collector.data[mod] is not None:
                                if self.collector.pointers[mod] < self.collector.data[mod].shape[1]:
                                    finished = False; break
                        if finished and not self.is_processing:
                            # 通知前端回放结束（用于 toast + 仪表盘归零）
                            if self.on_prediction_callback:
                                self.on_prediction_callback({
                                    "event": "finished",
                                    "score": 0,
                                    "confidence": 0.0,
                                    "timestamp": time.time(),
                                    "datetime": datetime.fromtimestamp(time.time()).isoformat(),
                                    "proc_time_ms": 0.0,
                                    "progress": 1.0,
                                })
                            print("\n✓ 原始数据回放完毕")
                            break

                time.sleep(collect_interval)
        except KeyboardInterrupt:
            print("\n接收到停止信号")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        # 等待当前任务完成后再 shutdown，避免 RuntimeError
        try:
            self.executor.shutdown(wait=True, timeout=5.0)
        except Exception:
            pass
        print("服务已停止")
