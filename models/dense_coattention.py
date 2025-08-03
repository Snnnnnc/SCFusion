import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseCoAttn(nn.Module):
    """
    Dense Co-Attention Network for physiological signals
    Adapted from the original implementation for EEG and ECG fusion
    """
    
    def __init__(self, dim1, dim2, dropout=0.1):
        super(DenseCoAttn, self).__init__()
        
        self.dim1 = dim1  # EEG dimension
        self.dim2 = dim2  # ECG dimension
        self.dropout = dropout
        
        # Query, Key, Value projections for each modality
        self.query_proj = nn.Linear(dim1 + dim2, dim1 + dim2)
        self.key1_proj = nn.Linear(dim1, dim1)
        self.key2_proj = nn.Linear(dim2, dim2)
        self.value1_proj = nn.Linear(dim1, dim1)
        self.value2_proj = nn.Linear(dim2, dim2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, value1, value2):
        """
        Forward pass
        
        Args:
            value1: EEG features [batch_size, time_points, dim1]
            value2: ECG features [batch_size, time_points, dim2]
            
        Returns:
            tuple: (weighted_eeg, weighted_ecg)
        """
        # Concatenate features for joint representation
        joint = torch.cat((value1, value2), dim=-1)
        
        # Generate query from joint representation
        query = self.query_proj(joint)
        
        # Generate keys and values for each modality
        key1 = self.key1_proj(value1)
        key2 = self.key2_proj(value2)
        value1_proj = self.value1_proj(value1)
        value2_proj = self.value2_proj(value2)
        
        # Apply attention mechanism
        weighted1, attn1 = self.qkv_attention(query, key1, value1_proj, self.dropout1)
        weighted2, attn2 = self.qkv_attention(query, key2, value2_proj, self.dropout2)
        
        return weighted1, weighted2
    
    def qkv_attention(self, query, key, value, dropout=None):
        """
        QKV attention mechanism
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            dropout: Dropout layer
            
        Returns:
            tuple: (weighted_value, attention_scores)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.bmm(key, query.transpose(1, 2)) / math.sqrt(d_k)
        scores = self.tanh(scores)
        
        if dropout:
            scores = dropout(scores)
        
        # Apply attention to values
        weighted = torch.bmm(value, scores)
        weighted = self.tanh(weighted)
        
        return self.relu(weighted), scores


class NormalSubLayer(nn.Module):
    """
    Normal Sub-Layer with residual connections
    """
    
    def __init__(self, dim1, dim2, dropout):
        super(NormalSubLayer, self).__init__()
        
        self.dense_coattn = DenseCoAttn(dim1, dim2, dropout)
        
        # Linear transformations for each modality
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim1 + dim2, dim1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ),
            nn.Sequential(
                nn.Linear(dim1 + dim2, dim2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        ])
        
    def forward(self, data1, data2):
        """
        Forward pass with residual connections
        
        Args:
            data1: EEG data [batch_size, time_points, dim1]
            data2: ECG data [batch_size, time_points, dim2]
            
        Returns:
            tuple: (updated_eeg, updated_ecg)
        """
        # Apply dense co-attention
        weighted1, weighted2 = self.dense_coattn(data1, data2)
        
        # Concatenate for linear transformation
        joint1 = torch.cat((weighted1, weighted2), dim=-1)
        joint2 = torch.cat((weighted1, weighted2), dim=-1)
        
        # Apply linear transformations with residual connections
        data1 = data1 + self.linears[0](joint1)
        data2 = data2 + self.linears[1](joint2)
        
        return data1, data2


class DCNLayer(nn.Module):
    """
    Dense Co-Attention Network Layer
    Multiple stacked co-attention layers
    """
    
    def __init__(self, dim1, dim2, num_seq=2, dropout=0.1):
        super(DCNLayer, self).__init__()
        
        self.dcn_layers = nn.ModuleList([
            NormalSubLayer(dim1, dim2, dropout) 
            for _ in range(num_seq)
        ])
        
    def forward(self, data1, data2):
        """
        Forward pass through multiple co-attention layers
        
        Args:
            data1: EEG data [batch_size, time_points, dim1]
            data2: ECG data [batch_size, time_points, dim2]
            
        Returns:
            tuple: (final_eeg, final_ecg)
        """
        for dense_coattn in self.dcn_layers:
            data1, data2 = dense_coattn(data1, data2)
        
        return data1, data2


class PhysiologicalCoAttention(nn.Module):
    """
    Physiological-specific Co-Attention Network
    Enhanced version with physiological signal considerations
    """
    
    def __init__(self, eeg_dim, ecg_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super(PhysiologicalCoAttention, self).__init__()
        
        self.eeg_dim = eeg_dim
        self.ecg_dim = ecg_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Projection layers
        self.eeg_projection = nn.Linear(eeg_dim, hidden_dim)
        self.ecg_projection = nn.Linear(ecg_dim, hidden_dim)
        
        # Co-attention layers
        self.co_attention_layers = nn.ModuleList([
            CoAttentionLayer(hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output projections
        self.eeg_output = nn.Linear(hidden_dim, eeg_dim)
        self.ecg_output = nn.Linear(hidden_dim, ecg_dim)
        
    def forward(self, eeg_data, ecg_data):
        """
        Forward pass
        
        Args:
            eeg_data: EEG features [batch_size, time_points, eeg_dim]
            ecg_data: ECG features [batch_size, time_points, ecg_dim]
            
        Returns:
            tuple: (enhanced_eeg, enhanced_ecg)
        """
        # Project to common space
        eeg_proj = self.eeg_projection(eeg_data)
        ecg_proj = self.ecg_projection(ecg_data)
        
        # Apply co-attention layers
        for layer in self.co_attention_layers:
            eeg_proj, ecg_proj = layer(eeg_proj, ecg_proj)
        
        # Project back to original dimensions
        enhanced_eeg = self.eeg_output(eeg_proj)
        enhanced_ecg = self.ecg_output(ecg_proj)
        
        return enhanced_eeg, enhanced_ecg


class CoAttentionLayer(nn.Module):
    """
    Single Co-Attention Layer
    """
    
    def __init__(self, hidden_dim, dropout=0.1):
        super(CoAttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Attention mechanisms
        self.eeg_to_ecg_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.ecg_to_eeg_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward networks
        self.eeg_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.ecg_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.eeg_norm1 = nn.LayerNorm(hidden_dim)
        self.eeg_norm2 = nn.LayerNorm(hidden_dim)
        self.ecg_norm1 = nn.LayerNorm(hidden_dim)
        self.ecg_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, eeg_data, ecg_data):
        """
        Forward pass
        
        Args:
            eeg_data: EEG features [batch_size, time_points, hidden_dim]
            ecg_data: ECG features [batch_size, time_points, hidden_dim]
            
        Returns:
            tuple: (enhanced_eeg, enhanced_ecg)
        """
        # Cross-attention: EEG attends to ECG
        eeg_attended, _ = self.eeg_to_ecg_attn(eeg_data, ecg_data, ecg_data)
        eeg_attended = self.eeg_norm1(eeg_data + eeg_attended)
        
        # Cross-attention: ECG attends to EEG
        ecg_attended, _ = self.ecg_to_eeg_attn(ecg_data, eeg_data, eeg_data)
        ecg_attended = self.ecg_norm1(ecg_data + ecg_attended)
        
        # Feed-forward networks
        eeg_ff = self.eeg_ff(eeg_attended)
        eeg_output = self.eeg_norm2(eeg_attended + eeg_ff)
        
        ecg_ff = self.ecg_ff(ecg_attended)
        ecg_output = self.ecg_norm2(ecg_attended + ecg_ff)
        
        return eeg_output, ecg_output 