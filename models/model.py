import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.nn import Linear, BatchNorm1d, Dropout, Sequential, Module

from .backbone import EEGBackbone, ECGBackbone
from .temporal_convolutional_model import TemporalConvNet
from .multimodal_fusion import MultimodalFusionModule
from .dense_coattention import DCNLayer


class PhysioFusionNet(nn.Module):
    """
    Physiological Fusion Network for Motion Sickness Classification
    Combines EEG and ECG signals using multimodal fusion techniques
    """
    
    def __init__(self, backbone_settings, modality=['eeg', 'ecg'], 
                 kernel_size=5, example_length=3000, tcn_attention=0,
                 tcn_channel={'eeg': [512, 256, 256, 128], 'ecg': [256, 128, 128, 64]},
                 embedding_dim={'eeg': 512, 'ecg': 256},
                 encoder_dim={'eeg': 128, 'ecg': 64},
                 modal_dim=64, num_heads=4, num_classes=11,
                 root_dir='', device='cuda'):
        super().__init__()
        
        self.backbone_settings = backbone_settings
        self.root_dir = root_dir
        self.device = device
        self.modality = modality
        self.kernel_size = kernel_size
        self.example_length = example_length
        self.tcn_channel = tcn_channel
        self.tcn_attention = tcn_attention
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.num_heads = num_heads
        self.modal_dim = modal_dim
        self.num_classes = num_classes
        
        # Initialize components
        self.spatial = nn.ModuleDict()
        self.temporal = nn.ModuleDict()
        self.bn = nn.ModuleDict()
        self.outputs = {}
        
        # Calculate final dimensions
        self.final_dim = sum([self.embedding_dim[modal] for modal in self.modality])
        
    def load_eeg_backbone(self, backbone_settings):
        """Load EEG feature extraction backbone"""
        eeg_backbone = EEGBackbone(
            input_channels=64,  # Default EEG channels
            output_dim=self.embedding_dim['eeg'],
            use_pretrained=True
        )
        
        if 'eeg_state_dict' in backbone_settings:
            state_dict_path = os.path.join(self.root_dir, backbone_settings['eeg_state_dict'] + ".pth")
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location='cpu')
                eeg_backbone.load_state_dict(state_dict)
                
        # Freeze backbone parameters initially
        for param in eeg_backbone.parameters():
            param.requires_grad = False
            
        return eeg_backbone
    
    def load_ecg_backbone(self, backbone_settings):
        """Load ECG feature extraction backbone"""
        ecg_backbone = ECGBackbone(
            input_channels=3,  # Default ECG channels
            output_dim=self.embedding_dim['ecg'],
            use_pretrained=True
        )
        
        if 'ecg_state_dict' in backbone_settings:
            state_dict_path = os.path.join(self.root_dir, backbone_settings['ecg_state_dict'] + ".pth")
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location='cpu')
                ecg_backbone.load_state_dict(state_dict)
                
        # Freeze backbone parameters initially
        for param in ecg_backbone.parameters():
            param.requires_grad = False
            
        return ecg_backbone
    
    def init(self):
        """Initialize model components"""
        self.output_dim = self.num_classes
        
        # Initialize spatial feature extractors
        if 'eeg' in self.modality:
            self.spatial["eeg"] = self.load_eeg_backbone(backbone_settings=self.backbone_settings)
            
        if 'ecg' in self.modality:
            self.spatial["ecg"] = self.load_ecg_backbone(backbone_settings=self.backbone_settings)
        
        # Initialize temporal feature extractors
        for modal in self.modality:
            self.temporal[modal] = TemporalConvNet(
                num_inputs=self.embedding_dim[modal],
                max_length=self.example_length,
                num_channels=self.tcn_channel[modal],
                attention=self.tcn_attention,
                kernel_size=self.kernel_size,
                dropout=0.1
            ).to(self.device)
            
            self.bn[modal] = BatchNorm1d(self.tcn_channel[modal][-1])
        
        # Initialize multimodal fusion module
        self.fusion = MultimodalFusionModule(
            modalities=self.modality,
            input_dims={modal: self.tcn_channel[modal][-1] for modal in self.modality},
            modal_dim=self.modal_dim,
            num_heads=self.num_heads,
            dropout=0.1
        )
        
        # Alternative: Use DCNLayer for fusion (similar to original RCMA)
        # self.fusion = DCNLayer(
        #     dim1=self.tcn_channel['eeg'][-1],
        #     dim2=self.tcn_channel['ecg'][-1], 
        #     dim3=0,  # No third modality
        #     num_seq=2,
        #     dropout=0.6
        # )
        
        # Initialize classifier
        fusion_output_dim = self.modal_dim * len(self.modality)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
    def forward(self, X):
        """
        Forward pass through the network
        
        Args:
            X (dict): Dictionary containing modality data
                - 'eeg': [batch_size, channels, time_points]
                - 'ecg': [batch_size, channels, time_points]
                
        Returns:
            torch.Tensor: Classification logits [batch_size, num_classes]
        """
        batch_size = None
        
        # Process EEG data
        if 'eeg' in X:
            if len(X['eeg'].shape) == 3:
                batch_size, channels, time_points = X['eeg'].shape
                # Reshape for spatial feature extraction
                X['eeg'] = X['eeg'].transpose(1, 2)  # [batch, time, channels]
                X['eeg'] = self.spatial.eeg(X['eeg'])  # Extract spatial features
                X['eeg'] = X['eeg'].transpose(1, 2)  # [batch, features, time]
            else:
                batch_size = X['eeg'].shape[0]
        
        # Process ECG data
        if 'ecg' in X:
            if len(X['ecg'].shape) == 3:
                if batch_size is None:
                    batch_size, channels, time_points = X['ecg'].shape
                # Reshape for spatial feature extraction
                X['ecg'] = X['ecg'].transpose(1, 2)  # [batch, time, channels]
                X['ecg'] = self.spatial.ecg(X['ecg'])  # Extract spatial features
                X['ecg'] = X['ecg'].transpose(1, 2)  # [batch, features, time]
            else:
                if batch_size is None:
                    batch_size = X['ecg'].shape[0]
        
        # Extract temporal features
        for modal in X:
            if modal in self.temporal:
                X[modal] = self.temporal[modal](X[modal])  # Temporal convolution
                X[modal] = self.bn[modal](X[modal])  # Batch normalization
                X[modal] = X[modal].transpose(1, 2)  # [batch, time, features]
        
        # Multimodal fusion
        if len(X) > 1:
            fused_features = self.fusion(X)
        else:
            # Single modality case
            modal = list(X.keys())[0]
            fused_features = X[modal]
        
        # Global average pooling over time dimension
        if len(fused_features.shape) == 3:
            fused_features = torch.mean(fused_features, dim=1)  # [batch, features]
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_attention_maps(self, X):
        """Get attention maps for visualization"""
        # This method can be implemented to extract attention weights
        # for interpretability analysis
        pass


class CAN(nn.Module):
    """
    Context-Aware Network (alternative model)
    """
    
    def __init__(self, modalities, tcn_settings, backbone_settings, output_dim, root_dir, device):
        super().__init__()
        # Implementation similar to original CAN but adapted for physiological signals
        pass
    
    def forward(self, X):
        pass 