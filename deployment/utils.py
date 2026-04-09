"""
部署工具函数模块
"""

import os
import pickle
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any

def setup_logging(log_dir: str, name: str = "deployment") -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        log_dir: 日志输出目录
        name: 记录器名称
    
    Returns:
        logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 防止重复添加handler
    if logger.handlers:
        return logger
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # 文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}_{timestamp}.log"))
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def save_prediction_result(result: Dict[str, Any], output_dir: str):
    """
    保存预测结果到文件
    
    Args:
        result: 预测结果字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用追加模式保存到csv或json
    # 这里简单保存为逐行JSON
    filename = os.path.join(output_dir, "predictions.jsonl")
    
    import json
    # 处理numpy数组以便序列化
    serializable_result = result.copy()
    if 'probability' in serializable_result and isinstance(serializable_result['probability'], np.ndarray):
        serializable_result['probability'] = serializable_result['probability'].tolist()
        
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(serializable_result) + "\n")

def load_pkl(path: str) -> Any:
    """加载pkl文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def format_prediction_output(result: Dict[str, Any]) -> str:
    """格式化预测结果输出字符串"""
    score = result.get('score', 0)
    confidence = result.get('confidence', 0.0)
    dt = result.get('datetime', '')
    
    # 简单的可视化进度条
    bar_len = 20
    filled_len = int(round(bar_len * confidence))
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    
    return f"[{dt}] 舒适度评分: {score} | 置信度: {confidence:.3f} [{bar}]"
