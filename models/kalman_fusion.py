"""
KalmanNet决策级融合模块
基于Kalman滤波器的思路，使用神经网络实现状态更新和增益计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GATLayer(nn.Module):
    """
    GATLayer: 基于图注意力网络的层

    输入：(N, in_features)
    输出：(N, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, negative_slope: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        # attention层输入是h_i和h_j的拼接，所以是2*out_features
        self.att = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = dropout
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.W(x)  # (N, out_features)
        
        # pairwise attention logits
        h_i = h.unsqueeze(1).expand(N, N, -1)  # (N, N, out_features)
        h_j = h.unsqueeze(0).expand(N, N, -1)  # (N, N, out_features)
        a_input = torch.cat([h_i, h_j], dim=-1)  # (N, N, 2*out_features)

        # 需要reshape为(N*N, 2*out_features)才能通过Linear层
        a_input_flat = a_input.view(-1, a_input.size(-1))  # (N*N, 2*out_features) --a(whi,whj)
        e_flat = self.att(a_input_flat).squeeze(-1)  # (N*N,)
        e = e_flat.view(N, N)  # (N, N)
        e = F.leaky_relu(e, self.negative_slope)

        # attention weights
        alpha = F.softmax(e, dim=1)
        alpha = F.dropout(alpha, p=self.dropout)
        
        # weighted sum
        out = alpha @ h

        return out, alpha

class MultiHeadGATLayer(nn.Module):
    """
    MultiHeadGATLayer: 多头图注意力网络层
    """
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.0, concat: bool = True):
        super().__init__()
        self.heads = nn.ModuleList([GATLayer(in_features, out_features, dropout) for _ in range(num_heads)])
        self.concat = concat
        self.num_heads = num_heads
        self.out_features = out_features if concat else out_features * num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs, alphas = [], []
        for head in self.heads:
            out, alpha = head(x)
            outs.append(out)
            alphas.append(alpha)
        
        if self.concat:
            out = torch.cat(outs, dim=-1)
        else:
            out = torch.stack(outs, dim=0).mean(dim=0)
        alpha = torch.stack(alphas, dim=0).mean(dim=0)
        return out, alpha

class GainNet(nn.Module):
    """
    GAT-GainNet: 基于图注意力网络生成Kalman增益矩阵K_k
    """
    
    def __init__(
        self, 
        innovation_dim: int = 10, 
        state_dim: int = 5, 
        node_dim: int = 16,
        gat_hidden: int = 16,
        num_heads: int = 4, 
        mlp_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.innovation_dim = innovation_dim
        self.state_dim = state_dim
        self.node_dim = node_dim
        self.gat_hidden = gat_hidden
        self.mlp_hidden = mlp_hidden    

        self.node_embed = nn.Linear(1, node_dim)
        self.gat_layer = MultiHeadGATLayer(node_dim, gat_hidden, num_heads, dropout)
        gat_out_dim = gat_hidden * num_heads
        
        self.mlp = nn.Sequential(
            nn.Linear(innovation_dim * gat_out_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, state_dim * innovation_dim),
        )

    def forward(self, innovation: torch.Tensor) -> torch.Tensor:
        batch = innovation.shape[0]
        node_embed = self.node_embed(innovation.unsqueeze(-1))

        K = []
        attns = []
        for b in range(batch):     
            node_embed_b = node_embed[b]
            out, alpha = self.gat_layer(node_embed_b)
            K_b = self.mlp(out.reshape(-1))
            K.append(K_b.reshape(self.state_dim, self.innovation_dim))
            attns.append(alpha)
        
        K = torch.stack(K, dim=0)
        attns = torch.stack(attns, dim=0)
        return K

# class GainNet(nn.Module):
#     """
#     GAT-GainNet: 基于图注意力网络生成Kalman增益矩阵K_k
    
#     输入: innovation (batch, 10) - 残差向量
#     输出: K_k (batch, 5, 10) - Kalman增益矩阵
#     """
    
#     def __init__(
#         self,
#         innovation_dim: int = 10,  # z_k的维度 (5+5)
#         state_dim: int = 5,  # x的维度
#         hidden_dims: list[int] = [64, 128],
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.innovation_dim = innovation_dim
#         self.state_dim = state_dim
        
#         layers = []
#         input_dim = innovation_dim
        
#         # 构建MLP网络
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(input_dim, hidden_dim))
#             layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout))
#             input_dim = hidden_dim
        
#         self.feature_extractor = nn.Sequential(*layers)
        
#         # 输出层：生成K_k矩阵 (state_dim × innovation_dim)
#         # 将输出reshape为(batch, state_dim, innovation_dim)
#         output_size = state_dim * innovation_dim
#         self.output_proj = nn.Linear(input_dim, output_size)
        
#     def forward(self, innovation: torch.Tensor) -> torch.Tensor:
#         """
#         输入:
#           - innovation: (batch, 10) - 创新残差向量
#         输出:
#           - K_k: (batch, 5, 10) - Kalman增益矩阵
#         """
#         batch = innovation.shape[0]
        
#         # 特征提取
#         features = self.feature_extractor(innovation)  # (batch, hidden_dim)
        
#         # 生成K_k矩阵
#         K_flat = self.output_proj(features)  # (batch, state_dim * innovation_dim)
#         K_k = K_flat.view(batch, self.state_dim, self.innovation_dim)  # (batch, 5, 10)
        
#         return K_k


class StatePredictor(nn.Module):
    """
    状态预测网络: x_pred[k] = f(x_hat[k-1])
    
    使用GRU或MLP实现状态转移函数
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        hidden_dim: int = 64,
        use_gru: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.use_gru = use_gru
        
        if use_gru:
            # 使用GRU进行时序状态预测
            self.gru = nn.GRU(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=dropout if 1 > 1 else 0,
            )
            self.output_proj = nn.Linear(hidden_dim, state_dim)
        else:
            # 使用MLP进行状态预测
            self.mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, state_dim),
            )
    
    def forward(self, x_prev: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
          - x_prev: (batch, state_dim) - 前一个状态
          - hidden: GRU的隐状态（可选）
        输出:
          - x_pred: (batch, state_dim) - 预测状态
          - hidden: GRU的隐状态
        """
        if self.use_gru:
            # GRU期望输入为 (batch, seq_len, input_size)
            x_input = x_prev.unsqueeze(1)  # (batch, 1, state_dim)
            gru_out, hidden = self.gru(x_input, hidden)  # gru_out: (batch, 1, hidden_dim)
            x_pred = self.output_proj(gru_out.squeeze(1))  # (batch, state_dim)
            return x_pred, hidden
        else:
            x_pred = self.mlp(x_prev)  # (batch, state_dim)
            return x_pred, None


class MeasurementPredictor(nn.Module):
    """
    测量预测网络: z_pred[k] = H(x_pred[k])
    
    将状态空间映射到测量空间
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        measurement_dim: int = 10,  # z_k的维度 (5+5)
        hidden_dims: list[int] = [64, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, measurement_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x_pred: torch.Tensor) -> torch.Tensor:
        """
        输入:
          - x_pred: (batch, state_dim) - 预测状态
        输出:
          - z_pred: (batch, measurement_dim) - 预测测量
        """
        return self.network(x_pred)  # (batch, 10)


class KalmanFusion(nn.Module):
    """
    KalmanNet决策级融合模块
    
    实现完整的Kalman滤波器流程：
    1. 状态预测: x_pred[k] = f(x_hat[k-1])
    2. 测量预测: z_pred[k] = H(x_pred[k])
    3. 计算创新残差: innovation[k] = z_k - z_pred[k]
    4. 生成增益矩阵: K_k = GainNet(innovation[k])
    5. 状态更新: x_hat[k] = x_pred[k] + K_k @ innovation[k]
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        measurement_dim: int = 10,  # z_k的维度 (5+5)
        num_patches: int = 10,
        state_predictor_hidden: int = 64,
        use_gru: bool = True,
        gain_net_hidden_dims: list[int] = [64, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.num_patches = num_patches
        
        # 状态预测网络
        self.state_predictor = StatePredictor(
            state_dim=state_dim,
            hidden_dim=state_predictor_hidden,
            use_gru=use_gru,
            dropout=dropout,
        )
        
        # 测量预测网络
        self.measurement_predictor = MeasurementPredictor(
            state_dim=state_dim,
            measurement_dim=measurement_dim,
            hidden_dims=gain_net_hidden_dims,
            dropout=dropout,
        )
        
        # GainNet: 生成Kalman增益矩阵
        # self.gain_net = GainNet(
        #     innovation_dim=measurement_dim,
        #     state_dim=state_dim,
        #     hidden_dims=gain_net_hidden_dims,
        #     dropout=dropout,
        # )
        self.gain_net = GainNet(
            innovation_dim=measurement_dim,
            state_dim=state_dim,
            dropout=dropout,
        )
    
    def forward(self, z_sequence: torch.Tensor) -> torch.Tensor:
        """
        输入:
          - z_sequence: (batch, num_patches, measurement_dim) - 测量序列
                         每个patch的测量是concat(p_imu[k], p_phy[k])
        输出:
          - x_hat_sequence: (batch, num_patches, state_dim) - 融合后的状态序列
        """
        batch, num_patches, measurement_dim = z_sequence.shape
        assert measurement_dim == self.measurement_dim, f"测量维度不匹配: {measurement_dim} != {self.measurement_dim}"
        assert num_patches == self.num_patches, f"patch数量不匹配: {num_patches} != {self.num_patches}"
        
        # 初始化状态（可以使用零初始化或可学习的初始状态）
        x_hat = torch.zeros(batch, self.state_dim, device=z_sequence.device, dtype=z_sequence.dtype)
        hidden = None  # GRU的隐状态
        
        x_hat_sequence = []
        
        # 对每个patch进行Kalman滤波更新
        for k in range(num_patches):
            z_k = z_sequence[:, k, :]  # (batch, measurement_dim) - 当前测量
            
            # 1. 状态预测: x_pred[k] = f(x_hat[k-1])
            x_pred, hidden = self.state_predictor(x_hat, hidden)  # (batch, state_dim)
            
            # 2. 测量预测: z_pred[k] = H(x_pred[k])
            z_pred = self.measurement_predictor(x_pred)  # (batch, measurement_dim)
            
            # 3. 计算创新残差: innovation[k] = z_k - z_pred[k]
            innovation = z_k - z_pred  # (batch, measurement_dim)
            
            # 4. 生成增益矩阵: K_k = GainNet(innovation[k])
            K_k = self.gain_net(innovation)  # (batch, state_dim, measurement_dim)
            
            # 5. 状态更新: x_hat[k] = x_pred[k] + K_k @ innovation[k]
            # K_k: (batch, state_dim, measurement_dim)
            # innovation: (batch, measurement_dim) -> (batch, 1, measurement_dim)
            innovation_expanded = innovation.unsqueeze(1)  # (batch, 1, measurement_dim)
            correction = torch.bmm(K_k, innovation_expanded.transpose(1, 2))  # (batch, state_dim, 1)
            correction = correction.squeeze(-1)  # (batch, state_dim)
            
            x_hat = x_pred + correction  # (batch, state_dim)
            
            x_hat_sequence.append(x_hat)
        
        # 将序列堆叠: (batch, num_patches, state_dim)
        x_hat_sequence = torch.stack(x_hat_sequence, dim=1)
        
        return x_hat_sequence

