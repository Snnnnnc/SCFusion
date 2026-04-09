"""
Cross-attention模态融合模块
"""

import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    """
    Cross-attention：EEG和ECG模态融合
    
    输入：
      - eeg_features: (batch, num_patches, N)
      - ecg_features: (batch, num_patches, N)
    输出：
      - fused: (batch, num_patches, N) 或 (batch, N)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_mode: str = "patch_wise",  # "patch_wise" 或 "global"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_mode = output_mode
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        self.head_dim = embed_dim // num_heads
        
        # Multi-head attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 如果输出global，使用attention pooling
        if output_mode == "global":
            # Attention pooling: 使用可学习的query vector对所有patches做attention
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=1,  # 可以设置为1，或者使用num_heads
                dropout=dropout,
                batch_first=True
            )
            # 可学习的query vector (1, 1, embed_dim)
            self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(
        self,
        eeg_features: torch.Tensor,
        ecg_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入：
          - eeg_features: (batch, num_patches, embed_dim)
          - ecg_features: (batch, num_patches, embed_dim)
        输出：
          - fused: (batch, num_patches, embed_dim) 或 (batch, embed_dim)
        """
        batch, num_patches, embed_dim = eeg_features.shape
        
        # EEG作为Query，ECG作为Key和Value
        q = self.q_proj(eeg_features)  # (batch, num_patches, embed_dim)
        k = self.k_proj(ecg_features)  # (batch, num_patches, embed_dim)
        v = self.v_proj(ecg_features)  # (batch, num_patches, embed_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch, num_patches, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, num_patches, head_dim)
        k = k.view(batch, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, num_heads, num_patches, num_patches)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, num_patches, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, num_patches, num_heads, head_dim)
        attn_output = attn_output.view(batch, num_patches, embed_dim)  # (batch, num_patches, embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.norm1(output + eeg_features)  # Residual connection
        
        # 如果输出global，使用attention pooling
        if self.output_mode == "global":
            # Attention pooling: 使用可学习的query对所有patches做attention
            # output: (batch, num_patches, embed_dim)
            # pool_query: (1, 1, embed_dim) -> 扩展到 (batch, 1, embed_dim)
            query = self.pool_query.expand(batch, -1, -1)  # (batch, 1, embed_dim)
            
            # MultiheadAttention: query作为query, output作为key和value
            pooled_output, _ = self.attention_pool(
                query, output, output  # query, key, value
            )  # pooled_output: (batch, 1, embed_dim)
            
            output = pooled_output.squeeze(1)  # (batch, embed_dim)
            output = self.norm2(output)
        else:
            output = self.norm2(output)
        
        return output


class BidirectionalCrossAttention(nn.Module):
    """
    双向Cross-attention：EEG->ECG 和 ECG->EEG 双向融合
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_mode: str = "patch_wise",
    ):
        super().__init__()
        self.eeg_to_ecg = CrossAttention(embed_dim, num_heads, dropout, output_mode)
        self.ecg_to_eeg = CrossAttention(embed_dim, num_heads, dropout, output_mode)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_mode = output_mode
    
    def forward(
        self,
        eeg_features: torch.Tensor,
        ecg_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入：
          - eeg_features: (batch, num_patches, embed_dim)
          - ecg_features: (batch, num_patches, embed_dim)
        输出：
          - fused: (batch, num_patches, embed_dim) 或 (batch, embed_dim)
        """
        # EEG attends to ECG
        eeg_enhanced = self.eeg_to_ecg(eeg_features, ecg_features)
        # ECG attends to EEG
        ecg_enhanced = self.ecg_to_eeg(ecg_features, eeg_features)
        
        # Concatenate and fuse
        if self.output_mode == "global":
            # (batch, embed_dim) each
            fused = torch.cat([eeg_enhanced, ecg_enhanced], dim=-1)  # (batch, embed_dim*2)
        else:
            # (batch, num_patches, embed_dim) each
            fused = torch.cat([eeg_enhanced, ecg_enhanced], dim=-1)  # (batch, num_patches, embed_dim*2)
        
        fused = self.fusion(fused)  # (batch, num_patches, embed_dim) 或 (batch, embed_dim)
        return fused


class SelfAttention(nn.Module):
    """
    Self-attention：用于单模态（如IMU）的时序特征学习
    学习运动和冲突的时序特征
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_mode: str = "patch_wise",  # "patch_wise" 或 "global"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_mode = output_mode
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        self.head_dim = embed_dim // num_heads
        
        # Multi-head self-attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        
        # 如果输出global，使用attention pooling
        if output_mode == "global":
            # Attention pooling: 使用可学习的query vector对所有patches做attention
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=1,  # 可以设置为1，或者使用num_heads
                dropout=dropout,
                batch_first=True
            )
            # 可学习的query vector (1, 1, embed_dim)
            self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入：
          - features: (batch, num_patches, embed_dim)
        输出：
          - output: (batch, num_patches, embed_dim) 或 (batch, embed_dim)
        """
        batch, num_patches, embed_dim = features.shape
        
        # Self-attention: Query, Key, Value都来自同一个输入
        q = self.q_proj(features)  # (batch, num_patches, embed_dim)
        k = self.k_proj(features)  # (batch, num_patches, embed_dim)
        v = self.v_proj(features)  # (batch, num_patches, embed_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch, num_patches, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, num_patches, head_dim)
        k = k.view(batch, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, num_heads, num_patches, num_patches)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, num_patches, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, num_patches, num_heads, head_dim)
        attn_output = attn_output.view(batch, num_patches, embed_dim)  # (batch, num_patches, embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.norm1(output + features)  # Residual connection
        
        # Feed-forward network
        ffn_output = self.ffn(output)
        output = self.norm2(output + ffn_output)  # Residual connection
        
        # 如果输出global，使用attention pooling
        if self.output_mode == "global":
            # Attention pooling: 使用可学习的query对所有patches做attention
            # output: (batch, num_patches, embed_dim)
            # pool_query: (1, 1, embed_dim) -> 扩展到 (batch, 1, embed_dim)
            query = self.pool_query.expand(batch, -1, -1)  # (batch, 1, embed_dim)
            
            # MultiheadAttention: query作为query, output作为key和value
            pooled_output, _ = self.attention_pool(
                query, output, output  # query, key, value
            )  # pooled_output: (batch, 1, embed_dim)
            
            output = pooled_output.squeeze(1)  # (batch, embed_dim)
        
        return output

