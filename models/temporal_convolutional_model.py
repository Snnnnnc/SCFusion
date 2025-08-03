import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalBlock(nn.Module):
    """
    Temporal Block for 1D temporal convolutions
    Adapted for physiological signal processing
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.dropout1,
            self.conv2, self.bn2, nn.ReLU(), self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            
    def forward(self, x):
        """Forward pass"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for physiological signals
    """
    
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, 
                 max_length=3000, attention=0):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        self.attention = attention
        
        if attention:
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=num_channels[-1],
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(num_channels[-1])
            
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, channels, time_points]
            
        Returns:
            torch.Tensor: Output features [batch_size, channels, time_points]
        """
        # Apply temporal convolutions
        x = self.network(x)
        
        # Apply attention if enabled
        if self.attention:
            # Transpose for attention: [batch, time, channels]
            x = x.transpose(1, 2)
            attended, _ = self.attention_layer(x, x, x)
            x = self.attention_norm(x + attended)
            # Transpose back: [batch, channels, time]
            x = x.transpose(1, 2)
            
        return x 