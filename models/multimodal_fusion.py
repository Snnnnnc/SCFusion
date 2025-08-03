import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultimodalFusionModule(nn.Module):
    """
    Multimodal Fusion Module for EEG and ECG signals
    Implements cross-attention mechanism for feature fusion
    """
    
    def __init__(self, modalities, input_dims, modal_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.modalities = modalities
        self.input_dims = input_dims
        self.modal_dim = modal_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Projection layers for each modality
        self.modal_projections = nn.ModuleDict()
        for modal in modalities:
            self.modal_projections[modal] = nn.Linear(input_dims[modal], modal_dim)
        
        # Cross-attention layers
        self.cross_attention = CrossAttention(
            modal_dim=modal_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output projection
        total_dim = modal_dim * len(modalities)
        self.output_projection = nn.Linear(total_dim, total_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(total_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (dict): Dictionary containing modality features
                - 'eeg': [batch_size, time_points, features]
                - 'ecg': [batch_size, time_points, features]
                
        Returns:
            torch.Tensor: Fused features [batch_size, time_points, total_features]
        """
        # Project each modality to common space
        projected_features = {}
        for modal in self.modalities:
            if modal in x:
                projected_features[modal] = self.modal_projections[modal](x[modal])
        
        # Apply cross-attention if multiple modalities
        if len(projected_features) > 1:
            fused_features = self.cross_attention(projected_features)
        else:
            # Single modality case
            modal = list(projected_features.keys())[0]
            fused_features = projected_features[modal]
        
        # Output projection and normalization
        fused_features = self.output_projection(fused_features)
        fused_features = self.layer_norm(fused_features)
        
        return fused_features


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for multimodal fusion
    """
    
    def __init__(self, modal_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.modal_dim = modal_dim
        self.num_heads = num_heads
        self.head_dim = modal_dim // num_heads
        
        assert modal_dim % num_heads == 0, "Modal dimension must be divisible by number of heads"
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=modal_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(modal_dim, modal_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(modal_dim * 4, modal_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(modal_dim)
        self.norm2 = nn.LayerNorm(modal_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, modal_features):
        """
        Forward pass
        
        Args:
            modal_features (dict): Dictionary of modality features
                Each value: [batch_size, time_points, modal_dim]
                
        Returns:
            torch.Tensor: Fused features [batch_size, time_points, total_dim]
        """
        modalities = list(modal_features.keys())
        
        if len(modalities) == 1:
            return modal_features[modalities[0]]
        
        # For multiple modalities, apply cross-attention
        fused_features = []
        
        for i, modal in enumerate(modalities):
            # Use current modality as query
            query = modal_features[modal]
            
            # Use all other modalities as key and value
            other_modals = [m for m in modalities if m != modal]
            key_value = torch.cat([modal_features[m] for m in other_modals], dim=1)
            
            # Apply cross-attention
            attn_output, _ = self.multihead_attn(query, key_value, key_value)
            
            # Residual connection and normalization
            attn_output = self.norm1(query + self.dropout(attn_output))
            
            # Feed-forward network
            ff_output = self.feed_forward(attn_output)
            ff_output = self.norm2(attn_output + self.dropout(ff_output))
            
            fused_features.append(ff_output)
        
        # Concatenate all fused features
        total_fused = torch.cat(fused_features, dim=-1)
        
        return total_fused


class SimpleFusion(nn.Module):
    """
    Simple fusion methods for comparison
    """
    
    def __init__(self, modalities, input_dims, fusion_type='concat'):
        super().__init__()
        
        self.modalities = modalities
        self.input_dims = input_dims
        self.fusion_type = fusion_type
        
        total_dim = sum(input_dims.values())
        
        if fusion_type == 'concat':
            self.projection = nn.Linear(total_dim, total_dim)
        elif fusion_type == 'attention':
            self.attention_weights = nn.Parameter(torch.ones(len(modalities)))
            self.projection = nn.Linear(total_dim, total_dim)
        elif fusion_type == 'weighted_sum':
            self.weights = nn.Parameter(torch.ones(len(modalities)))
            # Project to common dimension
            common_dim = max(input_dims.values())
            self.projections = nn.ModuleDict()
            for modal in modalities:
                self.projections[modal] = nn.Linear(input_dims[modal], common_dim)
            self.projection = nn.Linear(common_dim, common_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (dict): Dictionary of modality features
            
        Returns:
            torch.Tensor: Fused features
        """
        if self.fusion_type == 'concat':
            # Simple concatenation
            features = []
            for modal in self.modalities:
                if modal in x:
                    features.append(x[modal])
            fused = torch.cat(features, dim=-1)
            return self.projection(fused)
            
        elif self.fusion_type == 'attention':
            # Weighted concatenation with learned attention
            features = []
            weights = F.softmax(self.attention_weights, dim=0)
            for i, modal in enumerate(self.modalities):
                if modal in x:
                    features.append(x[modal] * weights[i])
            fused = torch.cat(features, dim=-1)
            return self.projection(fused)
            
        elif self.fusion_type == 'weighted_sum':
            # Weighted sum after projection to common space
            features = []
            weights = F.softmax(self.weights, dim=0)
            for i, modal in enumerate(self.modalities):
                if modal in x:
                    projected = self.projections[modal](x[modal])
                    features.append(projected * weights[i])
            fused = torch.stack(features, dim=0).sum(dim=0)
            return self.projection(fused)
        
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")


class TemporalFusion(nn.Module):
    """
    Temporal fusion for sequential multimodal data
    """
    
    def __init__(self, modalities, input_dims, hidden_dim=128, num_layers=2):
        super().__init__()
        
        self.modalities = modalities
        self.input_dims = input_dims
        
        # Project each modality to common dimension
        self.projections = nn.ModuleDict()
        for modal in modalities:
            self.projections[modal] = nn.Linear(input_dims[modal], hidden_dim)
        
        # LSTM for temporal fusion
        self.lstm = nn.LSTM(
            input_size=hidden_dim * len(modalities),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (dict): Dictionary of modality features
                Each value: [batch_size, time_points, features]
                
        Returns:
            torch.Tensor: Temporally fused features [batch_size, time_points, hidden_dim]
        """
        # Project each modality
        projected_features = []
        for modal in self.modalities:
            if modal in x:
                projected = self.projections[modal](x[modal])
                projected_features.append(projected)
        
        # Concatenate along feature dimension
        concatenated = torch.cat(projected_features, dim=-1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(concatenated)
        
        # Project to output dimension
        output = self.output_projection(lstm_out)
        
        return output 