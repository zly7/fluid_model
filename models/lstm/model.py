"""
LSTM model for fluid dynamics time series prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
import logging

from ..base import BaseModel
from .config import LSTMConfig

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for LSTM outputs."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor [B, T, hidden_dim]
            mask: Optional attention mask [B, T, T]
            
        Returns:
            Output tensor [B, T, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, T, T]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(out)


class FluidLSTM(BaseModel):
    """
    LSTM模型，用于天然气管网流体动力学预测。
    
    架构：
    1. 分别处理boundary和equipment变量
    2. 双向LSTM进行时间序列建模
    3. 可选的注意力机制
    4. 输出投影回原始维度
    """
    
    def __init__(self, config: Optional[LSTMConfig] = None, **kwargs):
        """
        初始化FluidLSTM模型。
        
        Args:
            config: LSTMConfig实例
            **kwargs: 额外参数（会覆盖config）
        """
        if config is None:
            config = LSTMConfig()
        
        # 更新config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 从config中删除input_dim和output_dim，避免重复传递
        config_dict = config.to_dict()
        config_dict.pop('input_dim', None)
        config_dict.pop('output_dim', None)
        super().__init__(input_dim=config.input_dim, output_dim=config.output_dim, **config_dict)
        
        self.config = config
        
        # 构建模型
        self._build_model()
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info(f"FluidLSTM initialized: {self.get_model_info()}")
    
    def _build_model(self):
        """构建LSTM模型架构。"""
        config = self.config
        
        # Boundary变量处理 (前538维)
        self.boundary_projection = nn.Sequential(
            nn.Linear(config.boundary_dims, config.boundary_hidden_dim),
            self._get_activation(config.activation),
            nn.BatchNorm1d(config.boundary_hidden_dim) if config.batch_norm else nn.Identity(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Equipment变量处理 (后6174维) 
        self.equipment_projection = nn.Sequential(
            nn.Linear(config.equipment_dims, config.equipment_hidden_dim),
            self._get_activation(config.activation),
            nn.BatchNorm1d(config.equipment_hidden_dim) if config.batch_norm else nn.Identity(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 计算LSTM输入维度
        lstm_input_dim = config.boundary_hidden_dim + config.equipment_hidden_dim
        
        # LSTM层 - 处理时间序列
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.recurrent_dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # 可选的注意力机制
        if config.use_attention:
            self.attention = MultiHeadAttention(
                hidden_dim=lstm_output_dim,
                num_heads=config.attention_heads,
                dropout=config.dropout_rate
            )
        else:
            self.attention = None
        
        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_dim)
        else:
            self.layer_norm = None
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, config.projection_hidden_dim),
            self._get_activation(config.activation),
            nn.BatchNorm1d(config.projection_hidden_dim) if config.batch_norm else nn.Identity(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.projection_hidden_dim, config.output_dim)
        )
        
        # Dropout层
        self.dropout = nn.Dropout(config.weight_dropout)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """根据字符串获取激活函数。"""
        if activation == "tanh":
            return nn.Tanh()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            return nn.Tanh()
    
    def _initialize_weights(self):
        """初始化模型权重。"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights - Xavier initialization
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights - Orthogonal initialization
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases - zeros, but set forget gate bias to 1
                param.data.fill_(0)
                # For LSTM, set forget gate bias to 1
                if 'bias_ih' in name:
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids=None, labels=None, prediction_mask=None, **kwargs):
        """
        FluidLSTM前向传播，兼容transformers格式。
        
        Args:
            input_ids: 输入张量 [B, T, V=6712] 或包含'input'键的字典
            labels: 目标张量 [B, T, V=6712] 用于损失计算 (可选)
            prediction_mask: 预测mask [B, V] (可选)
            **kwargs: 额外参数
        
        Returns:
            如果提供labels: {'loss': tensor, 'logits': tensor}
            否则: {'logits': tensor}
        """
        # 处理输入格式兼容性
        if isinstance(input_ids, dict):
            # 兼容原有的batch格式
            batch = input_ids
            x = batch['input']  # [B, T, V]
            if labels is None and 'target' in batch:
                labels = batch['target']
            if prediction_mask is None and 'prediction_mask' in batch:
                prediction_mask = batch['prediction_mask']
        else:
            x = input_ids  # [B, T, V]
        
        batch_size, time_steps, num_variables = x.shape
        
        # 分离boundary和equipment变量
        boundary_data = x[:, :, :self.boundary_dims]  # [B, T, 538]
        equipment_data = x[:, :, self.boundary_dims:]  # [B, T, 6174]
        
        # 处理boundary变量
        # Reshape for linear projection: [B, T, 538] -> [B*T, 538]
        boundary_reshaped = boundary_data.reshape(-1, self.boundary_dims)
        boundary_features = self.boundary_projection(boundary_reshaped)  # [B*T, boundary_hidden_dim]
        boundary_features = boundary_features.reshape(batch_size, time_steps, -1)  # [B, T, boundary_hidden_dim]
        
        # 处理equipment变量
        # Reshape for linear projection: [B, T, 6174] -> [B*T, 6174]
        equipment_reshaped = equipment_data.reshape(-1, self.equipment_dims)
        equipment_features = self.equipment_projection(equipment_reshaped)  # [B*T, equipment_hidden_dim]
        equipment_features = equipment_features.reshape(batch_size, time_steps, -1)  # [B, T, equipment_hidden_dim]
        
        # 融合特征
        combined_features = torch.cat([boundary_features, equipment_features], dim=-1)  # [B, T, lstm_input_dim]
        
        # 应用dropout
        combined_features = self.dropout(combined_features)
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(combined_features)  # [B, T, lstm_output_dim]
        
        # 可选的注意力机制
        if self.attention is not None:
            # 创建因果mask（可选）
            seq_len = lstm_out.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=lstm_out.device))
            causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, T]
            
            lstm_out = self.attention(lstm_out, mask=causal_mask)
        
        # Layer normalization
        if self.layer_norm is not None:
            lstm_out = self.layer_norm(lstm_out)
        
        # 输出投影
        # Reshape for linear layer: [B, T, C] -> [B*T, C]
        output_reshaped = lstm_out.reshape(-1, lstm_out.size(-1))
        predictions = self.output_projection(output_reshaped)  # [B*T, output_dim]
        predictions = predictions.reshape(batch_size, time_steps, self.output_dim)  # [B, T, V]
        
        # 返回格式兼容transformers
        if labels is not None:
            # 计算loss（使用基类的统一损失函数）
            loss = self.compute_loss(predictions, labels, prediction_mask)
            return {'loss': loss, 'logits': predictions}
        else:
            # 只返回预测结果
            return {'logits': predictions}
    
    def get_model_info(self) -> Dict:
        """获取详细的模型信息。"""
        base_info = super().get_model_info()
        
        lstm_info = {
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'bidirectional': self.config.bidirectional,
            'boundary_hidden_dim': self.config.boundary_hidden_dim,
            'equipment_hidden_dim': self.config.equipment_hidden_dim,
            'use_attention': self.config.use_attention,
            'attention_heads': self.config.attention_heads if self.config.use_attention else None,
            'activation': self.config.activation,
            'use_layer_norm': self.config.use_layer_norm
        }
        
        base_info.update(lstm_info)
        return base_info