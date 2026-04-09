"""
舒适度分类模型：整合Patch编码、Cross-attention和分类网络
"""

import torch
import torch.nn as nn
from typing import Optional

from .patch_encoder import PatchEncoder1D
from .cross_attention import BidirectionalCrossAttention, SelfAttention
from .kalman_fusion import KalmanFusion


class ComfortClassificationModel(nn.Module):
    """
    完整模型流程：
    1. Patch编码（EEG和ECG分别编码）
    2. Cross-attention融合
    3. 分类网络输出
    """
    
    def __init__(
        self,
        eeg_channels: int,
        ecg_channels: int,
        patch_length: int,
        num_patches: int = 10,
        encoding_dim: int = 256,
        num_heads: int = 8,
        num_classes: int = 5,  # 0-4分
        attention_output_mode: str = "global",  # "global" 或 "patch_wise"
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.encoding_dim = encoding_dim
        self.attention_output_mode = attention_output_mode
        
        # 1. Patch编码器
        self.eeg_encoder = PatchEncoder1D(
            input_channels=eeg_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        self.ecg_encoder = PatchEncoder1D(
            input_channels=ecg_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        
        # 2. Cross-attention融合
        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode=attention_output_mode,
        )
        
        # 3. 分类网络
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        classifier_layers = []
        input_dim = encoding_dim
        if attention_output_mode == "patch_wise":
            # 如果是patch_wise，先池化
            classifier_layers.append(nn.AdaptiveAvgPool1d(1))
            # 输入维度需要调整
            input_dim = encoding_dim * num_patches
        
        for hidden_dim in hidden_dims:
            classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(
        self,
        eeg_patches: torch.Tensor,
        ecg_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入：
          - eeg_patches: (batch, num_patches, eeg_channels, patch_length)
          - ecg_patches: (batch, num_patches, ecg_channels, patch_length)
        输出：
          - logits: (batch, num_classes)
        """
        batch, num_patches, eeg_ch, patch_len = eeg_patches.shape
        _, _, ecg_ch, _ = ecg_patches.shape
        
        # 1. Patch编码：对每个patch独立编码
        eeg_encoded = []
        ecg_encoded = []
        
        for p in range(num_patches):
            eeg_patch = eeg_patches[:, p, :, :]  # (batch, eeg_channels, patch_length)
            ecg_patch = ecg_patches[:, p, :, :]  # (batch, ecg_channels, patch_length)
            
            eeg_enc = self.eeg_encoder(eeg_patch)  # (batch, encoding_dim)
            ecg_enc = self.ecg_encoder(ecg_patch)  # (batch, encoding_dim)
            
            eeg_encoded.append(eeg_enc)
            ecg_encoded.append(ecg_enc)
        
        eeg_features = torch.stack(eeg_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        ecg_features = torch.stack(ecg_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        
        # 2. Cross-attention融合
        fused = self.cross_attention(eeg_features, ecg_features)  # (batch, num_patches, encoding_dim) 或 (batch, encoding_dim)
        
        # 3. 分类
        if self.attention_output_mode == "patch_wise":
            # fused: (batch, num_patches, encoding_dim)
            fused = fused.transpose(1, 2)  # (batch, encoding_dim, num_patches)
            # 通过classifier的第一层（池化）
            fused = self.classifier[0](fused)  # (batch, encoding_dim, 1)
            fused = fused.squeeze(-1)  # (batch, encoding_dim)
            # 继续后续层
            for layer in self.classifier[1:]:
                fused = layer(fused)
            logits = fused
        else:
            # fused: (batch, encoding_dim)
            logits = self.classifier(fused)  # (batch, num_classes)
        
        return logits


class IMUClassificationModel(nn.Module):
    """
    IMU数据分类模型：
    1. Patch编码（IMU数据编码）
    2. Self-attention融合（学习运动和冲突的时序特征）
    3. 分类网络输出
    """
    
    def __init__(
        self,
        imu_channels: int,
        patch_length: int,
        num_patches: int = 10,
        encoding_dim: int = 256,
        num_heads: int = 8,
        num_classes: int = 5,  # 0-4分
        attention_output_mode: str = "global",  # "global" 或 "patch_wise"
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.encoding_dim = encoding_dim
        self.attention_output_mode = attention_output_mode
        
        # 1. Patch编码器
        self.imu_encoder = PatchEncoder1D(
            input_channels=imu_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        
        # 2. Self-attention融合
        self.self_attention = SelfAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode=attention_output_mode,
        )
        
        # 3. 分类网络
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        classifier_layers = []
        input_dim = encoding_dim
        if attention_output_mode == "patch_wise":
            # 如果是patch_wise，先池化
            classifier_layers.append(nn.AdaptiveAvgPool1d(1))
            # 输入维度需要调整
            input_dim = encoding_dim * num_patches
        
        for hidden_dim in hidden_dims:
            classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(
        self,
        imu_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入：
          - imu_patches: (batch, num_patches, imu_channels, patch_length)
        输出：
          - logits: (batch, num_classes)
        """
        batch, num_patches, imu_ch, patch_len = imu_patches.shape
        
        # 1. Patch编码：对每个patch独立编码
        imu_encoded = []
        
        for p in range(num_patches):
            imu_patch = imu_patches[:, p, :, :]  # (batch, imu_channels, patch_length)
            
            imu_enc = self.imu_encoder(imu_patch)  # (batch, encoding_dim)
            
            imu_encoded.append(imu_enc)
        
        imu_features = torch.stack(imu_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        
        # 2. Self-attention融合
        fused = self.self_attention(imu_features)  # (batch, num_patches, encoding_dim) 或 (batch, encoding_dim)
        
        # 3. 分类
        if self.attention_output_mode == "patch_wise":
            # fused: (batch, num_patches, encoding_dim)
            fused = fused.transpose(1, 2)  # (batch, encoding_dim, num_patches)
            # 通过classifier的第一层（池化）
            fused = self.classifier[0](fused)  # (batch, encoding_dim, 1)
            fused = fused.squeeze(-1)  # (batch, encoding_dim)
            # 继续后续层
            for layer in self.classifier[1:]:
                fused = layer(fused)
            logits = fused
        else:
            # fused: (batch, encoding_dim)
            logits = self.classifier(fused)  # (batch, num_classes)
        
        return logits


class SingleModalPhysioModel(nn.Module):
    """
    单模态生理信号分类模型（仅 EEG 或仅 ECG）：
    Patch 编码 + Self-attention + 分类器。用于 mode='eeg' 或 mode='ecg'。
    """
    def __init__(
        self,
        channels: int,
        patch_length: int,
        num_patches: int = 10,
        encoding_dim: int = 256,
        num_heads: int = 8,
        num_classes: int = 5,
        attention_output_mode: str = "global",
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
        modal: str = "ecg",  # "ecg" 或 "eeg"，用于 trainer 从 batch 取对应 key
    ):
        super().__init__()
        self.num_patches = num_patches
        self.encoding_dim = encoding_dim
        self.attention_output_mode = attention_output_mode
        self.modal = modal  # 标识输入来自 batch['ecg'] 还是 batch['eeg']
        self.physio_encoder = PatchEncoder1D(
            input_channels=channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        self.self_attention = SelfAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode=attention_output_mode,
        )
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        classifier_layers = []
        input_dim = encoding_dim
        if attention_output_mode == "patch_wise":
            classifier_layers.append(nn.AdaptiveAvgPool1d(1))
            input_dim = encoding_dim * num_patches
        for hidden_dim in hidden_dims:
            classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        输入: patches (batch, num_patches, channels, patch_length)
        输出: logits (batch, num_classes)
        """
        batch, num_patches, ch, patch_len = patches.shape
        encoded = []
        for p in range(num_patches):
            patch = patches[:, p, :, :]
            enc = self.physio_encoder(patch)
            encoded.append(enc)
        features = torch.stack(encoded, dim=1)
        fused = self.self_attention(features)
        if self.attention_output_mode == "patch_wise":
            fused = fused.transpose(1, 2)
            fused = self.classifier[0](fused)
            fused = fused.squeeze(-1)
            for layer in self.classifier[1:]:
                fused = layer(fused)
            return fused
        return self.classifier(fused)


def create_model(
    eeg_channels: int,
    ecg_channels: int,
    patch_length: int = 250,  # 1s @ 250Hz
    num_patches: int = 10,
    encoding_dim: int = 256,
    num_classes: int = 5,
    **kwargs,
) -> ComfortClassificationModel:
    """
    创建模型实例
    """
    return ComfortClassificationModel(
        eeg_channels=eeg_channels,
        ecg_channels=ecg_channels,
        patch_length=patch_length,
        num_patches=num_patches,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        **kwargs,
    )


def create_imu_model(
    imu_channels: int,
    patch_length: int = 250,  # 1s @ 250Hz
    num_patches: int = 10,
    encoding_dim: int = 256,
    num_classes: int = 5,
    **kwargs,
) -> IMUClassificationModel:
    """
    创建IMU模型实例
    """
    return IMUClassificationModel(
        imu_channels=imu_channels,
        patch_length=patch_length,
        num_patches=num_patches,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        **kwargs,
    )


class MixClassificationModel(nn.Module):
    """
    IMU+Physio混合分类模型：
    1. IMU和Physio分别进行Patch编码和Attention融合（保持原有架构）
    2. 对每个patch进行分类预测，得到patch-level预测（各5维向量）
    3. 使用KalmanNet思路进行决策级融合
    4. 通过attention pooling得到窗口级预测
    """
    
    def __init__(
        self,
        imu_channels: int,
        eeg_channels: int,
        ecg_channels: int,
        patch_length: int,
        num_patches: int = 10,
        encoding_dim: int = 256,
        num_heads: int = 8,
        num_classes: int = 5,  # 0-4分
        attention_output_mode: str = "patch_wise",  # 必须使用patch_wise以保持patch级别特征
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
        # KalmanFusion参数
        state_predictor_hidden: int = 64,
        use_gru: bool = True,
        gain_net_hidden_dims: Optional[list[int]] = None,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
        self.attention_output_mode = attention_output_mode
        
        # 确保使用patch_wise模式以保持patch级别特征
        assert attention_output_mode == "patch_wise", "MixClassificationModel必须使用patch_wise模式"
        
        # 1. IMU分支：Patch编码器
        self.imu_encoder = PatchEncoder1D(
            input_channels=imu_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        
        # 2. IMU分支：Self-attention融合（输出patch_wise特征）
        self.imu_self_attention = SelfAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode="patch_wise",  # 保持patch级别特征
        )
        
        # 3. Physio分支：Patch编码器
        self.eeg_encoder = PatchEncoder1D(
            input_channels=eeg_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        self.ecg_encoder = PatchEncoder1D(
            input_channels=ecg_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        
        # 4. Physio分支：Cross-attention融合（输出patch_wise特征）
        self.physio_cross_attention = BidirectionalCrossAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode="patch_wise",  # 保持patch级别特征
        )
        
        # 5. Patch-level分类器：对每个patch进行分类预测
        if hidden_dims is None:
            hidden_dims = [128, 64]  # 较小的网络，因为输入是patch级别特征
        
        # IMU分支的patch分类器
        imu_patch_classifier_layers = []
        input_dim = encoding_dim
        for hidden_dim in hidden_dims:
            imu_patch_classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            imu_patch_classifier_layers.append(nn.ReLU())
            imu_patch_classifier_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        imu_patch_classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.imu_patch_classifier = nn.Sequential(*imu_patch_classifier_layers)
        
        # Physio分支的patch分类器
        physio_patch_classifier_layers = []
        input_dim = encoding_dim
        for hidden_dim in hidden_dims:
            physio_patch_classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            physio_patch_classifier_layers.append(nn.ReLU())
            physio_patch_classifier_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        physio_patch_classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.physio_patch_classifier = nn.Sequential(*physio_patch_classifier_layers)
        
        # 6. KalmanNet融合模块
        if gain_net_hidden_dims is None:
            gain_net_hidden_dims = [64, 128]
        
        self.kalman_fusion = KalmanFusion(
            state_dim=num_classes,  # 状态维度 = 类别数 (5)
            measurement_dim=num_classes * 2,  # 测量维度 = 2 * 类别数 (10: imu预测+physio预测)
            num_patches=num_patches,
            state_predictor_hidden=state_predictor_hidden,
            use_gru=use_gru,
            gain_net_hidden_dims=gain_net_hidden_dims,
            dropout=dropout,
        )
        
        # 7. Attention pooling：对融合后的状态序列进行池化
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=num_classes,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )
        # 可学习的query vector (1, 1, num_classes)
        self.pool_query = nn.Parameter(torch.randn(1, 1, num_classes))
    
    def forward(
        self,
        imu_patches: torch.Tensor,
        eeg_patches: torch.Tensor,
        ecg_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入：
          - imu_patches: (batch, num_patches, imu_channels, patch_length)
          - eeg_patches: (batch, num_patches, eeg_channels, patch_length)
          - ecg_patches: (batch, num_patches, ecg_channels, patch_length)
        输出：
          - logits: (batch, num_classes)
        """
        batch, num_patches, imu_ch, patch_len = imu_patches.shape
        _, _, eeg_ch, _ = eeg_patches.shape
        _, _, ecg_ch, _ = ecg_patches.shape
        
        # ========== IMU分支：Patch编码 + Attention ==========
        imu_encoded = []
        for p in range(num_patches):
            imu_patch = imu_patches[:, p, :, :]  # (batch, imu_channels, patch_length)
            imu_enc = self.imu_encoder(imu_patch)  # (batch, encoding_dim)
            imu_encoded.append(imu_enc)
        imu_features = torch.stack(imu_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        
        # Self-attention融合（输出patch_wise特征）
        imu_fused = self.imu_self_attention(imu_features)  # (batch, num_patches, encoding_dim)
        
        # ========== Physio分支：Patch编码 + Attention ==========
        eeg_encoded = []
        ecg_encoded = []
        for p in range(num_patches):
            eeg_patch = eeg_patches[:, p, :, :]  # (batch, eeg_channels, patch_length)
            ecg_patch = ecg_patches[:, p, :, :]  # (batch, ecg_channels, patch_length)
            eeg_enc = self.eeg_encoder(eeg_patch)  # (batch, encoding_dim)
            ecg_enc = self.ecg_encoder(ecg_patch)  # (batch, encoding_dim)
            eeg_encoded.append(eeg_enc)
            ecg_encoded.append(ecg_enc)
        eeg_features = torch.stack(eeg_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        ecg_features = torch.stack(ecg_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        
        # Cross-attention融合（输出patch_wise特征）
        physio_fused = self.physio_cross_attention(eeg_features, ecg_features)  # (batch, num_patches, encoding_dim)
        
        # ========== Patch-level分类预测 ==========
        # 对每个patch进行分类预测
        imu_patch_logits = []  # 存储每个patch的预测
        physio_patch_logits = []  # 存储每个patch的预测
        
        for p in range(num_patches):
            imu_patch_feat = imu_fused[:, p, :]  # (batch, encoding_dim)
            physio_patch_feat = physio_fused[:, p, :]  # (batch, encoding_dim)
            
            # 分类预测（输出logits，后面会应用softmax得到概率）
            imu_logits = self.imu_patch_classifier(imu_patch_feat)  # (batch, num_classes)
            physio_logits = self.physio_patch_classifier(physio_patch_feat)  # (batch, num_classes)
            
            # 转换为概率（softmax）
            imu_probs = torch.softmax(imu_logits, dim=-1)  # (batch, num_classes)
            physio_probs = torch.softmax(physio_logits, dim=-1)  # (batch, num_classes)
            
            imu_patch_logits.append(imu_probs)
            physio_patch_logits.append(physio_probs)
        
        # 堆叠为序列: (batch, num_patches, num_classes)
        imu_patch_probs = torch.stack(imu_patch_logits, dim=1)  # (batch, num_patches, 5)
        physio_patch_probs = torch.stack(physio_patch_logits, dim=1)  # (batch, num_patches, 5)
        
        # ========== KalmanNet决策级融合 ==========
        # 拼接每个patch的imu和physio预测: z_k = concat(p_imu[k], p_phy[k])
        z_sequence = torch.cat([imu_patch_probs, physio_patch_probs], dim=-1)  # (batch, num_patches, 10)
        
        # KalmanNet融合: 得到融合后的状态序列
        x_hat_sequence = self.kalman_fusion(z_sequence)  # (batch, num_patches, 5)
        
        # ========== Attention Pooling ==========
        # 使用attention pooling对融合后的状态序列进行池化
        # x_hat_sequence: (batch, num_patches, num_classes)
        query = self.pool_query.expand(batch, -1, -1)  # (batch, 1, num_classes)
        
        pooled_output, _ = self.attention_pool(
            query, x_hat_sequence, x_hat_sequence  # query, key, value
        )  # pooled_output: (batch, 1, num_classes)
        
        logits = pooled_output.squeeze(1)  # (batch, num_classes)
        
        return logits


def create_mix_model(
    imu_channels: int,
    eeg_channels: int,
    ecg_channels: int,
    patch_length: int = 250,  # 1s @ 250Hz
    num_patches: int = 10,
    encoding_dim: int = 256,
    num_classes: int = 5,
    **kwargs,
) -> MixClassificationModel:
    """
    创建Mix模型实例
    """
    return MixClassificationModel(
        imu_channels=imu_channels,
        eeg_channels=eeg_channels,
        ecg_channels=ecg_channels,
        patch_length=patch_length,
        num_patches=num_patches,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        **kwargs,
    )


class SimpleMixClassificationModel(nn.Module):
    """
    IMU+ECG混合分类模型（SimpleMix模式）：
    1. IMU分支：Patch编码 + Self-attention融合（保持原状）
    2. ECG分支：Patch编码 + Self-attention融合（替换cross-attention为self-attention）
    3. 对每个patch进行分类预测，得到patch-level预测（各5维向量）
    4. 使用KalmanNet思路进行决策级融合（只使用ECG和IMU，不包含EEG）
    5. 通过attention pooling得到窗口级预测
    """
    
    def __init__(
        self,
        imu_channels: int,
        ecg_channels: int,
        patch_length: int,
        num_patches: int = 10,
        encoding_dim: int = 256,
        num_heads: int = 8,
        num_classes: int = 5,  # 0-4分
        attention_output_mode: str = "patch_wise",  # 必须使用patch_wise以保持patch级别特征
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
        # KalmanFusion参数
        state_predictor_hidden: int = 64,
        use_gru: bool = True,
        gain_net_hidden_dims: Optional[list[int]] = None,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
        self.attention_output_mode = attention_output_mode
        
        # 确保使用patch_wise模式以保持patch级别特征
        assert attention_output_mode == "patch_wise", "SimpleMixClassificationModel必须使用patch_wise模式"
        
        # 1. IMU分支：Patch编码器（保持原状）
        self.imu_encoder = PatchEncoder1D(
            input_channels=imu_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        
        # 2. IMU分支：Self-attention融合（输出patch_wise特征，保持原状）
        self.imu_self_attention = SelfAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode="patch_wise",  # 保持patch级别特征
        )
        
        # 3. ECG分支：Patch编码器（和physio一致）
        self.ecg_encoder = PatchEncoder1D(
            input_channels=ecg_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        
        # 4. ECG分支：Self-attention融合（替换cross-attention为self-attention，和imu模型一致）
        self.ecg_self_attention = SelfAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode="patch_wise",  # 保持patch级别特征
        )
        
        # 5. Patch-level分类器：对每个patch进行分类预测
        if hidden_dims is None:
            hidden_dims = [128, 64]  # 较小的网络，因为输入是patch级别特征
        
        # IMU分支的patch分类器
        imu_patch_classifier_layers = []
        input_dim = encoding_dim
        for hidden_dim in hidden_dims:
            imu_patch_classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            imu_patch_classifier_layers.append(nn.ReLU())
            imu_patch_classifier_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        imu_patch_classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.imu_patch_classifier = nn.Sequential(*imu_patch_classifier_layers)
        
        # ECG分支的patch分类器
        ecg_patch_classifier_layers = []
        input_dim = encoding_dim
        for hidden_dim in hidden_dims:
            ecg_patch_classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            ecg_patch_classifier_layers.append(nn.ReLU())
            ecg_patch_classifier_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        ecg_patch_classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.ecg_patch_classifier = nn.Sequential(*ecg_patch_classifier_layers)
        
        # 6. KalmanNet融合模块（和mix模型完全一致，只使用ECG和IMU）
        if gain_net_hidden_dims is None:
            gain_net_hidden_dims = [64, 128]
        
        self.kalman_fusion = KalmanFusion(
            state_dim=num_classes,  # 状态维度 = 类别数 (5)
            measurement_dim=num_classes * 2,  # 测量维度 = 2 * 类别数 (10: imu预测+ecg预测)
            num_patches=num_patches,
            state_predictor_hidden=state_predictor_hidden,
            use_gru=use_gru,
            gain_net_hidden_dims=gain_net_hidden_dims,
            dropout=dropout,
        )
        
        # 7. Attention pooling：对融合后的状态序列进行池化
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=num_classes,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )
        # 可学习的query vector (1, 1, num_classes)
        self.pool_query = nn.Parameter(torch.randn(1, 1, num_classes))
    
    def forward(
        self,
        imu_patches: torch.Tensor,
        ecg_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入：
          - imu_patches: (batch, num_patches, imu_channels, patch_length)
          - ecg_patches: (batch, num_patches, ecg_channels, patch_length)
        输出：
          - logits: (batch, num_classes)
        """
        batch, num_patches, imu_ch, patch_len = imu_patches.shape
        _, _, ecg_ch, _ = ecg_patches.shape
        
        # ========== IMU分支：Patch编码 + Attention ==========
        imu_encoded = []
        for p in range(num_patches):
            imu_patch = imu_patches[:, p, :, :]  # (batch, imu_channels, patch_length)
            imu_enc = self.imu_encoder(imu_patch)  # (batch, encoding_dim)
            imu_encoded.append(imu_enc)
        imu_features = torch.stack(imu_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        
        # Self-attention融合（输出patch_wise特征）
        imu_fused = self.imu_self_attention(imu_features)  # (batch, num_patches, encoding_dim)
        
        # ========== ECG分支：Patch编码 + Self-attention ==========
        ecg_encoded = []
        for p in range(num_patches):
            ecg_patch = ecg_patches[:, p, :, :]  # (batch, ecg_channels, patch_length)
            ecg_enc = self.ecg_encoder(ecg_patch)  # (batch, encoding_dim)
            ecg_encoded.append(ecg_enc)
        ecg_features = torch.stack(ecg_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        
        # Self-attention融合（输出patch_wise特征，替换了cross-attention）
        ecg_fused = self.ecg_self_attention(ecg_features)  # (batch, num_patches, encoding_dim)
        
        # ========== Patch-level分类预测 ==========
        # 对每个patch进行分类预测
        imu_patch_logits = []  # 存储每个patch的预测
        ecg_patch_logits = []  # 存储每个patch的预测
        
        for p in range(num_patches):
            imu_patch_feat = imu_fused[:, p, :]  # (batch, encoding_dim)
            ecg_patch_feat = ecg_fused[:, p, :]  # (batch, encoding_dim)
            
            # 分类预测（输出logits，后面会应用softmax得到概率）
            imu_logits = self.imu_patch_classifier(imu_patch_feat)  # (batch, num_classes)
            ecg_logits = self.ecg_patch_classifier(ecg_patch_feat)  # (batch, num_classes)
            
            # 转换为概率（softmax）
            imu_probs = torch.softmax(imu_logits, dim=-1)  # (batch, num_classes)
            ecg_probs = torch.softmax(ecg_logits, dim=-1)  # (batch, num_classes)
            
            imu_patch_logits.append(imu_probs)
            ecg_patch_logits.append(ecg_probs)
        
        # 堆叠为序列: (batch, num_patches, num_classes)
        imu_patch_probs = torch.stack(imu_patch_logits, dim=1)  # (batch, num_patches, 5)
        ecg_patch_probs = torch.stack(ecg_patch_logits, dim=1)  # (batch, num_patches, 5)
        
        # ========== KalmanNet决策级融合 ==========
        # 拼接每个patch的imu和ecg预测: z_k = concat(p_imu[k], p_ecg[k])
        z_sequence = torch.cat([imu_patch_probs, ecg_patch_probs], dim=-1)  # (batch, num_patches, 10)
        
        # KalmanNet融合: 得到融合后的状态序列
        x_hat_sequence = self.kalman_fusion(z_sequence)  # (batch, num_patches, 5)
        
        # ========== Attention Pooling ==========
        # 使用attention pooling对融合后的状态序列进行池化
        # x_hat_sequence: (batch, num_patches, num_classes)
        query = self.pool_query.expand(batch, -1, -1)  # (batch, 1, num_classes)
        
        pooled_output, _ = self.attention_pool(
            query, x_hat_sequence, x_hat_sequence  # query, key, value
        )  # pooled_output: (batch, 1, num_classes)
        
        logits = pooled_output.squeeze(1)  # (batch, num_classes)
        
        return logits


def create_simple_mix_model(
    imu_channels: int,
    ecg_channels: int,
    patch_length: int = 250,  # 1s @ 250Hz
    num_patches: int = 10,
    encoding_dim: int = 256,
    num_classes: int = 5,
    **kwargs,
) -> SimpleMixClassificationModel:
    """
    创建SimpleMix模型实例
    """
    return SimpleMixClassificationModel(
        imu_channels=imu_channels,
        ecg_channels=ecg_channels,
        patch_length=patch_length,
        num_patches=num_patches,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        **kwargs,
    )


class NewMixClassificationModel(nn.Module):
    """
    IMU+ECG特征级融合分类模型（NewMix模式）：
    整体结构和physio完全一致，但输入换成imu和ecg
    1. Patch编码（IMU和ECG分别编码）
    2. Cross-attention特征级融合
    3. 分类网络输出
    """
    
    def __init__(
        self,
        imu_channels: int,
        ecg_channels: int,
        patch_length: int,
        num_patches: int = 10,
        encoding_dim: int = 256,
        num_heads: int = 8,
        num_classes: int = 5,  # 0-4分
        attention_output_mode: str = "global",  # "global" 或 "patch_wise"
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.encoding_dim = encoding_dim
        self.attention_output_mode = attention_output_mode
        
        # 1. Patch编码器
        self.imu_encoder = PatchEncoder1D(
            input_channels=imu_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        self.ecg_encoder = PatchEncoder1D(
            input_channels=ecg_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        
        # 2. Cross-attention特征级融合
        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode=attention_output_mode,
        )
        
        # 3. 分类网络
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        classifier_layers = []
        input_dim = encoding_dim
        if attention_output_mode == "patch_wise":
            # 如果是patch_wise，先池化
            classifier_layers.append(nn.AdaptiveAvgPool1d(1))
            # 输入维度需要调整
            input_dim = encoding_dim * num_patches
        
        for hidden_dim in hidden_dims:
            classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(
        self,
        imu_patches: torch.Tensor,
        ecg_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入：
          - imu_patches: (batch, num_patches, imu_channels, patch_length)
          - ecg_patches: (batch, num_patches, ecg_channels, patch_length)
        输出：
          - logits: (batch, num_classes)
        """
        batch, num_patches, imu_ch, patch_len = imu_patches.shape
        _, _, ecg_ch, _ = ecg_patches.shape
        
        # 1. Patch编码：对每个patch独立编码
        imu_encoded = []
        ecg_encoded = []
        
        for p in range(num_patches):
            imu_patch = imu_patches[:, p, :, :]  # (batch, imu_channels, patch_length)
            ecg_patch = ecg_patches[:, p, :, :]  # (batch, ecg_channels, patch_length)
            
            imu_enc = self.imu_encoder(imu_patch)  # (batch, encoding_dim)
            ecg_enc = self.ecg_encoder(ecg_patch)  # (batch, encoding_dim)
            
            imu_encoded.append(imu_enc)
            ecg_encoded.append(ecg_enc)
        
        imu_features = torch.stack(imu_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        ecg_features = torch.stack(ecg_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        
        # 2. Cross-attention特征级融合
        fused = self.cross_attention(imu_features, ecg_features)  # (batch, num_patches, encoding_dim) 或 (batch, encoding_dim)
        
        # 3. 分类
        if self.attention_output_mode == "patch_wise":
            # fused: (batch, num_patches, encoding_dim)
            fused = fused.transpose(1, 2)  # (batch, encoding_dim, num_patches)
            # 通过classifier的第一层（池化）
            fused = self.classifier[0](fused)  # (batch, encoding_dim, 1)
            fused = fused.squeeze(-1)  # (batch, encoding_dim)
            # 继续后续层
            for layer in self.classifier[1:]:
                fused = layer(fused)
            logits = fused
        else:
            # fused: (batch, encoding_dim)
            logits = self.classifier(fused)  # (batch, num_classes)
        
        return logits


def create_new_mix_model(
    imu_channels: int,
    ecg_channels: int,
    patch_length: int = 250,  # 1s @ 250Hz
    num_patches: int = 10,
    encoding_dim: int = 256,
    num_classes: int = 5,
    **kwargs,
) -> NewMixClassificationModel:
    """
    创建NewMix模型实例
    """
    return NewMixClassificationModel(
        imu_channels=imu_channels,
        ecg_channels=ecg_channels,
        patch_length=patch_length,
        num_patches=num_patches,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        **kwargs,
    )


class AllMixClassificationModel(nn.Module):
    """
    IMU+EEG+ECG三模态特征级融合分类模型（AllMix模式）：
    1. 三个模态分别做patch encoder
    2. IMU分别和ECG、EEG做Cross-attention
    3. IMU+EEG和IMU+ECG再做Cross-attention
    4. 通过MLP分类器输出
    """
    
    def __init__(
        self,
        imu_channels: int,
        eeg_channels: int,
        ecg_channels: int,
        patch_length: int,
        num_patches: int = 10,
        encoding_dim: int = 256,
        num_heads: int = 8,
        num_classes: int = 5,  # 0-4分
        attention_output_mode: str = "global",  # "global" 或 "patch_wise"
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.encoding_dim = encoding_dim
        self.attention_output_mode = attention_output_mode
        
        # 1. Patch编码器（三个模态分别编码）
        self.imu_encoder = PatchEncoder1D(
            input_channels=imu_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        self.eeg_encoder = PatchEncoder1D(
            input_channels=eeg_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        self.ecg_encoder = PatchEncoder1D(
            input_channels=ecg_channels,
            patch_length=patch_length,
            output_dim=encoding_dim,
            dropout=dropout,
        )
        
        # 2. 第一层Cross-attention：IMU分别和ECG、EEG做Cross-attention
        # IMU <-> ECG Cross-attention
        self.imu_ecg_cross_attention = BidirectionalCrossAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode=attention_output_mode,
        )
        # IMU <-> EEG Cross-attention
        self.imu_eeg_cross_attention = BidirectionalCrossAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode=attention_output_mode,
        )
        
        # 3. 第二层Cross-attention：IMU+EEG和IMU+ECG再做Cross-attention
        # 注意：这里需要融合后的特征，所以需要先处理第一层的输出
        # 如果output_mode是global，第一层输出是(batch, encoding_dim)
        # 如果output_mode是patch_wise，第一层输出是(batch, num_patches, encoding_dim)
        # 为了统一处理，我们使用一个融合层来合并IMU+EEG和IMU+ECG的特征
        self.final_cross_attention = BidirectionalCrossAttention(
            embed_dim=encoding_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_mode=attention_output_mode,
        )
        
        # 4. 分类网络
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        classifier_layers = []
        input_dim = encoding_dim
        if attention_output_mode == "patch_wise":
            # 如果是patch_wise，先池化
            classifier_layers.append(nn.AdaptiveAvgPool1d(1))
            # 输入维度需要调整
            input_dim = encoding_dim * num_patches
        
        for hidden_dim in hidden_dims:
            classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(
        self,
        imu_patches: torch.Tensor,
        eeg_patches: torch.Tensor,
        ecg_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        输入：
          - imu_patches: (batch, num_patches, imu_channels, patch_length)
          - eeg_patches: (batch, num_patches, eeg_channels, patch_length)
          - ecg_patches: (batch, num_patches, ecg_channels, patch_length)
        输出：
          - logits: (batch, num_classes)
        """
        batch, num_patches, imu_ch, patch_len = imu_patches.shape
        _, _, eeg_ch, _ = eeg_patches.shape
        _, _, ecg_ch, _ = ecg_patches.shape
        
        # 1. Patch编码：对每个patch独立编码
        imu_encoded = []
        eeg_encoded = []
        ecg_encoded = []
        
        for p in range(num_patches):
            imu_patch = imu_patches[:, p, :, :]  # (batch, imu_channels, patch_length)
            eeg_patch = eeg_patches[:, p, :, :]  # (batch, eeg_channels, patch_length)
            ecg_patch = ecg_patches[:, p, :, :]  # (batch, ecg_channels, patch_length)
            
            imu_enc = self.imu_encoder(imu_patch)  # (batch, encoding_dim)
            eeg_enc = self.eeg_encoder(eeg_patch)  # (batch, encoding_dim)
            ecg_enc = self.ecg_encoder(ecg_patch)  # (batch, encoding_dim)
            
            imu_encoded.append(imu_enc)
            eeg_encoded.append(eeg_enc)
            ecg_encoded.append(ecg_enc)
        
        imu_features = torch.stack(imu_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        eeg_features = torch.stack(eeg_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        ecg_features = torch.stack(ecg_encoded, dim=1)  # (batch, num_patches, encoding_dim)
        
        # 2. 第一层Cross-attention：IMU分别和ECG、EEG做Cross-attention
        # IMU <-> ECG Cross-attention
        imu_ecg_fused = self.imu_ecg_cross_attention(imu_features, ecg_features)  # (batch, num_patches, encoding_dim) 或 (batch, encoding_dim)
        # IMU <-> EEG Cross-attention
        imu_eeg_fused = self.imu_eeg_cross_attention(imu_features, eeg_features)  # (batch, num_patches, encoding_dim) 或 (batch, encoding_dim)
        
        # 3. 第二层Cross-attention：IMU+EEG和IMU+ECG再做Cross-attention
        # 注意：BidirectionalCrossAttention的forward方法期望输入是(batch, num_patches, embed_dim)
        # 如果第一层输出是global模式(batch, encoding_dim)，需要扩展回patch_wise模式
        if self.attention_output_mode == "global":
            # 如果是global模式，第一层输出是(batch, encoding_dim)
            # 需要扩展为(batch, num_patches, encoding_dim)以便第二层Cross-attention处理
            if len(imu_ecg_fused.shape) == 2:
                # (batch, encoding_dim) -> (batch, num_patches, encoding_dim)
                imu_ecg_fused = imu_ecg_fused.unsqueeze(1).expand(-1, num_patches, -1)
            if len(imu_eeg_fused.shape) == 2:
                # (batch, encoding_dim) -> (batch, num_patches, encoding_dim)
                imu_eeg_fused = imu_eeg_fused.unsqueeze(1).expand(-1, num_patches, -1)
        
        # 现在两个特征都是(batch, num_patches, encoding_dim)，可以传入第二层Cross-attention
        # 注意：第二层Cross-attention也使用相同的output_mode，所以输出形状会与第一层一致
        final_fused = self.final_cross_attention(imu_eeg_fused, imu_ecg_fused)  # (batch, num_patches, encoding_dim) 或 (batch, encoding_dim)
        
        # 4. 分类
        if self.attention_output_mode == "patch_wise":
            # final_fused: (batch, num_patches, encoding_dim)
            final_fused = final_fused.transpose(1, 2)  # (batch, encoding_dim, num_patches)
            # 通过classifier的第一层（池化）
            final_fused = self.classifier[0](final_fused)  # (batch, encoding_dim, 1)
            final_fused = final_fused.squeeze(-1)  # (batch, encoding_dim)
            # 继续后续层
            for layer in self.classifier[1:]:
                final_fused = layer(final_fused)
            logits = final_fused
        else:
            # final_fused: (batch, encoding_dim)
            logits = self.classifier(final_fused)  # (batch, num_classes)
        
        return logits


def create_all_mix_model(
    imu_channels: int,
    eeg_channels: int,
    ecg_channels: int,
    patch_length: int = 250,  # 1s @ 250Hz
    num_patches: int = 10,
    encoding_dim: int = 256,
    num_classes: int = 5,
    **kwargs,
) -> AllMixClassificationModel:
    """
    创建AllMix模型实例
    """
    return AllMixClassificationModel(
        imu_channels=imu_channels,
        eeg_channels=eeg_channels,
        ecg_channels=ecg_channels,
        patch_length=patch_length,
        num_patches=num_patches,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        **kwargs,
    )

