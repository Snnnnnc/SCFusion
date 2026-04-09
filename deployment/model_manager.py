"""
模型管理器模块
模型加载、管理和推理
"""

import os
import pickle
import torch
import numpy as np
import sys
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.comfort_model import (
    ComfortClassificationModel, IMUClassificationModel, MixClassificationModel,
    SimpleMixClassificationModel, NewMixClassificationModel, AllMixClassificationModel
)


def infer_mode_from_folder(folder_path: str) -> str:
    """从文件夹路径推断mode"""
    folder_name = os.path.basename(folder_path)
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


def load_checkpoint(checkpoint_path: str):
    """加载checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


class ModelManager:
    """模型管理器"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0', mode: Optional[str] = None):
        """
        初始化模型管理器
        
        Args:
            checkpoint_path: checkpoint文件路径
            device: 设备 ('cuda:0', 'cpu'等)
            mode: 模型模式（如果None则自动推断）
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
        self.mode = mode
        self.model = None
        self.args_dict = {}
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """从checkpoint加载模型"""
        print(f"加载checkpoint: {self.checkpoint_path}")
        checkpoint = load_checkpoint(self.checkpoint_path)
        
        # 从checkpoint获取模型参数
        trainer_state = checkpoint.get('trainer_state', {})
        model_state_dict = trainer_state.get('model_state_dict', None)
        
        if model_state_dict is None:
            raise ValueError("checkpoint中未找到模型状态")
        
        # 从checkpoint获取参数
        checkpoint_args = checkpoint.get('args', {})
        if not isinstance(checkpoint_args, dict):
            checkpoint_args = {}
        
        # 如果mode未指定，尝试从文件夹名推断
        if self.mode is None:
            self.mode = checkpoint_args.get('mode', None)
            if self.mode is None:
                self.mode = infer_mode_from_folder(os.path.dirname(self.checkpoint_path))
                print(f"从文件夹名推断mode: {self.mode}")
        
        # 默认参数值
        self.args_dict = {
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
        if isinstance(self.args_dict['hidden_dims'], str):
            self.args_dict['hidden_dims'] = [int(x.strip()) for x in self.args_dict['hidden_dims'].split(',')]
        
        # 根据mode初始化模型
        num_classes = self.args_dict['num_classes']
        
        if self.mode == 'allmix':
            self.model = AllMixClassificationModel(
                imu_channels=self.args_dict['imu_channels'],
                eeg_channels=self.args_dict['eeg_channels'],
                ecg_channels=self.args_dict['ecg_channels'],
                patch_length=self.args_dict['patch_length'],
                num_patches=self.args_dict['num_patches'],
                encoding_dim=self.args_dict['encoding_dim'],
                num_heads=self.args_dict['num_heads'],
                num_classes=num_classes,
                attention_output_mode=self.args_dict['attention_output_mode'],
                hidden_dims=self.args_dict['hidden_dims'],
                dropout=self.args_dict['dropout'],
            )
        elif self.mode == 'newmix':
            self.model = NewMixClassificationModel(
                imu_channels=self.args_dict['imu_channels'],
                eeg_channels=self.args_dict['eeg_channels'],
                ecg_channels=self.args_dict['ecg_channels'],
                patch_length=self.args_dict['patch_length'],
                num_patches=self.args_dict['num_patches'],
                encoding_dim=self.args_dict['encoding_dim'],
                num_heads=self.args_dict['num_heads'],
                num_classes=num_classes,
                attention_output_mode=self.args_dict['attention_output_mode'],
                hidden_dims=self.args_dict['hidden_dims'],
                dropout=self.args_dict['dropout'],
            )
        elif self.mode == 'simplemix':
            self.model = SimpleMixClassificationModel(
                imu_channels=self.args_dict['imu_channels'],
                eeg_channels=self.args_dict['eeg_channels'],
                ecg_channels=self.args_dict['ecg_channels'],
                patch_length=self.args_dict['patch_length'],
                num_patches=self.args_dict['num_patches'],
                encoding_dim=self.args_dict['encoding_dim'],
                num_heads=self.args_dict['num_heads'],
                num_classes=num_classes,
                attention_output_mode=self.args_dict['attention_output_mode'],
                hidden_dims=self.args_dict['hidden_dims'],
                dropout=self.args_dict['dropout'],
            )
        elif self.mode == 'mix':
            self.model = MixClassificationModel(
                imu_channels=self.args_dict['imu_channels'],
                eeg_channels=self.args_dict['eeg_channels'],
                ecg_channels=self.args_dict['ecg_channels'],
                patch_length=self.args_dict['patch_length'],
                num_patches=self.args_dict['num_patches'],
                encoding_dim=self.args_dict['encoding_dim'],
                num_heads=self.args_dict['num_heads'],
                num_classes=num_classes,
                attention_output_mode=self.args_dict['attention_output_mode'],
                hidden_dims=self.args_dict['hidden_dims'],
                dropout=self.args_dict['dropout'],
            )
        elif self.mode == 'imu':
            self.model = IMUClassificationModel(
                imu_channels=self.args_dict['imu_channels'],
                patch_length=self.args_dict['patch_length'],
                num_patches=self.args_dict['num_patches'],
                encoding_dim=self.args_dict['encoding_dim'],
                num_heads=self.args_dict['num_heads'],
                num_classes=num_classes,
                attention_output_mode=self.args_dict['attention_output_mode'],
                hidden_dims=self.args_dict['hidden_dims'],
                dropout=self.args_dict['dropout'],
            )
        elif self.mode == 'rawimu':
            self.model = IMUClassificationModel(
                imu_channels=6,  # Raw IMU只使用6维
                patch_length=self.args_dict['patch_length'],
                num_patches=self.args_dict['num_patches'],
                encoding_dim=self.args_dict['encoding_dim'],
                num_heads=self.args_dict['num_heads'],
                num_classes=num_classes,
                attention_output_mode=self.args_dict['attention_output_mode'],
                hidden_dims=self.args_dict['hidden_dims'],
                dropout=self.args_dict['dropout'],
            )
        else:  # physio
            self.model = ComfortClassificationModel(
                eeg_channels=self.args_dict['eeg_channels'],
                ecg_channels=self.args_dict['ecg_channels'],
                patch_length=self.args_dict['patch_length'],
                num_patches=self.args_dict['num_patches'],
                encoding_dim=self.args_dict['encoding_dim'],
                num_heads=self.args_dict['num_heads'],
                num_classes=num_classes,
                attention_output_mode=self.args_dict['attention_output_mode'],
                hidden_dims=self.args_dict['hidden_dims'],
                dropout=self.args_dict['dropout'],
            )
        
        # 加载模型权重
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ 模型已加载 (mode={self.mode}, num_classes={num_classes}, device={self.device})")
    
    def predict(self, patches_dict: Dict[str, np.ndarray]) -> Dict:
        """
        执行推理，返回舒适度评分
        
        Args:
            patches_dict: patches格式数据 {'imu': (10, channels, 250), ...}
        
        Returns:
            预测结果字典 {'score': int, 'probability': np.array, 'confidence': float}
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        with torch.no_grad():
            # 确保 numpy 数组是连续的且步长为正（解决 ValueError: At least one stride in the given numpy array is negative）
            processed_patches = {}
            for key, val in patches_dict.items():
                if isinstance(val, np.ndarray):
                    # 使用 ascontiguousarray 确保步长为正且内存连续
                    processed_patches[key] = np.ascontiguousarray(val)
                else:
                    processed_patches[key] = val

            # 转换为torch tensor并添加batch维度
            if self.mode == 'allmix':
                imu_patches = torch.from_numpy(processed_patches['imu']).float().unsqueeze(0).to(self.device)
                eeg_patches = torch.from_numpy(processed_patches['eeg']).float().unsqueeze(0).to(self.device)
                ecg_patches = torch.from_numpy(processed_patches['ecg']).float().unsqueeze(0).to(self.device)
                outputs = self.model(imu_patches, eeg_patches, ecg_patches)
            elif self.mode == 'newmix' or self.mode == 'simplemix' or self.mode == 'mix':
                imu_patches = torch.from_numpy(processed_patches['imu']).float().unsqueeze(0).to(self.device)
                eeg_patches = torch.from_numpy(processed_patches['eeg']).float().unsqueeze(0).to(self.device)
                ecg_patches = torch.from_numpy(processed_patches['ecg']).float().unsqueeze(0).to(self.device)
                outputs = self.model(imu_patches, eeg_patches, ecg_patches)
            elif self.mode == 'imu' or self.mode == 'rawimu':
                imu_patches = torch.from_numpy(processed_patches['imu']).float().unsqueeze(0).to(self.device)
                outputs = self.model(imu_patches)
            else:  # physio
                eeg_patches = torch.from_numpy(processed_patches['eeg']).float().unsqueeze(0).to(self.device)
                ecg_patches = torch.from_numpy(processed_patches['ecg']).float().unsqueeze(0).to(self.device)
                outputs = self.model(eeg_patches, ecg_patches)
            
            # 计算概率和预测类别
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            confidence = float(np.max(probs))
            
            return {
                'score': pred,
                'probability': probs,
                'confidence': confidence
            }
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'mode': self.mode,
            'device': str(self.device),
            'args': self.args_dict,
            'checkpoint_path': self.checkpoint_path
        }
