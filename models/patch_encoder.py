"""
1D CNN Patch编码器：对每个patch独立编码
"""

import torch
import torch.nn as nn


class PatchEncoder1D(nn.Module):
    """
    1D CNN编码器：将 (C, patch_length) 编码为 (N,) 向量
    
    输入：单个patch (C, patch_length)
    输出：编码向量 (N,)
    """
    
    def __init__(
        self,
        input_channels: int,
        patch_length: int,
        hidden_dims: list[int] = [64, 128, 256],
        output_dim: int = 256,
        kernel_sizes: list[int] = [7, 5, 3],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.patch_length = patch_length
        self.output_dim = output_dim
        
        layers = []
        in_ch = input_channels
        current_length = patch_length
        
        # 卷积层
        for i, (hidden_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            layers.append(
                nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=kernel_size//2)
            )
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            if i < len(hidden_dims) - 1:
                # 中间层可以加池化
                layers.append(nn.MaxPool1d(2))
                current_length = current_length // 2
            in_ch = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 全局池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：x (batch, C, patch_length) 或 (C, patch_length)
        输出：(batch, N) 或 (N,)
        """
        # x: (batch, C, patch_length)
        x = self.conv_layers(x)  # (batch, hidden_dim, L')
        x = self.global_pool(x)  # (batch, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch, hidden_dim)
        x = self.fc(x)  # (batch, output_dim)
        return x


def create_encoders(
    eeg_channels: int,
    ecg_channels: int,
    patch_length: int,
    encoding_dim: int = 256,
) -> tuple[PatchEncoder1D, PatchEncoder1D]:
    """
    创建EEG和ECG的编码器
    """
    eeg_encoder = PatchEncoder1D(
        input_channels=eeg_channels,
        patch_length=patch_length,
        output_dim=encoding_dim,
    )
    ecg_encoder = PatchEncoder1D(
        input_channels=ecg_channels,
        patch_length=patch_length,
        output_dim=encoding_dim,
    )
    return eeg_encoder, ecg_encoder

