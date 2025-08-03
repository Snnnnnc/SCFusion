import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EEGBackbone(nn.Module):
    """
    EEG Feature Extraction Backbone Network
    Designed specifically for EEG signal processing
    """
    
    def __init__(self, input_channels=64, output_dim=512, use_pretrained=True):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.use_pretrained = use_pretrained
        
        # EEG-specific 1D CNN architecture
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(input_channels, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Third conv block
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Final projection layer
        self.final_projection = nn.Linear(512, output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input EEG signal [batch_size, time_points, channels]
            
        Returns:
            torch.Tensor: EEG features [batch_size, output_dim]
        """
        # Transpose to [batch_size, channels, time_points] for 1D conv
        if len(x.shape) == 3 and x.shape[-1] == self.input_channels:
            x = x.transpose(1, 2)
        
        # Extract features
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # Remove time dimension
        
        # Final projection
        x = self.final_projection(x)
        
        return x


class ECGBackbone(nn.Module):
    """
    ECG Feature Extraction Backbone Network
    Designed specifically for ECG signal processing
    """
    
    def __init__(self, input_channels=3, output_dim=256, use_pretrained=True):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.use_pretrained = use_pretrained
        
        # ECG-specific 1D CNN architecture
        self.conv_layers = nn.Sequential(
            # First conv block - capture local patterns
            nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block - capture R-wave patterns
            nn.Conv1d(64, 128, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Third conv block - capture global patterns
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Final projection layer
        self.final_projection = nn.Linear(256, output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input ECG signal [batch_size, time_points, channels]
            
        Returns:
            torch.Tensor: ECG features [batch_size, output_dim]
        """
        # Transpose to [batch_size, channels, time_points] for 1D conv
        if len(x.shape) == 3 and x.shape[-1] == self.input_channels:
            x = x.transpose(1, 2)
        
        # Extract features
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # Remove time dimension
        
        # Final projection
        x = self.final_projection(x)
        
        return x 