"""
TCN (Temporal Convolutional Network) model for fluid dynamics time series prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

from ..base import BaseModel
from .config import TCNConfig

logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """
    简化的TCN构建块，适用于短序列。
    """
    
    def __init__(self, 
                 n_inputs: int, 
                 n_outputs: int, 
                 kernel_size: int, 
                 stride: int, 
                 dilation: int, 
                 padding: int, 
                 dropout: float = 0.2,
                 use_norm: bool = True,
                 activation: str = "relu",
                 use_residual: bool = True):
        super().__init__()
        
        self.use_residual = use_residual
        
        # 确保kernel_size适合短序列，强制最大为3
        effective_kernel_size = min(kernel_size, 3)
        
        # 简单的1D卷积，使用same padding保持序列长度
        padding = (effective_kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, effective_kernel_size,
                              stride=stride, padding=padding, dilation=1)
        
        # 归一化层
        if use_norm:
            self.norm1 = nn.BatchNorm1d(n_outputs)
        else:
            self.norm1 = nn.Identity()
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接的投影层（如果输入输出维度不同）
        if use_residual:
            if n_inputs != n_outputs:
                self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
            else:
                self.downsample = None
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        
        if hasattr(self, 'downsample') and self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量 [B, C, T]
            
        Returns:
            输出张量 [B, C_out, T]
        """
        # 卷积块
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.use_residual:
            if self.downsample is not None:
                res = self.downsample(x)
            else:
                res = x
            
            out = out + res
        
        return self.activation(out)


class TemporalConvNet(nn.Module):
    """
    完整的TCN网络，由多个TemporalBlock组成。
    """
    
    def __init__(self, 
                 num_inputs: int, 
                 num_channels: List[int], 
                 kernel_size: int = 2, 
                 dropout: float = 0.2,
                 use_norm: bool = True,
                 activation: str = "relu",
                 use_residual: bool = True):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # padding参数传递给TemporalBlock，将在其内部处理
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size, 
                                   padding=0,  # TemporalBlock内部处理padding
                                   dropout=dropout,
                                   use_norm=use_norm, activation=activation,
                                   use_residual=use_residual)]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量 [B, C, T]
            
        Returns:
            输出张量 [B, num_channels[-1], T]
        """
        return self.network(x)


class FluidTCN(BaseModel):
    """
    TCN模型，用于天然气管网流体动力学预测。
    
    架构：
    1. 分别处理boundary和equipment变量
    2. 使用TCN网络进行时间序列建模
    3. 特征融合和输出投影
    """
    
    def __init__(self, config: Optional[TCNConfig] = None, **kwargs):
        """
        初始化FluidTCN模型。
        
        Args:
            config: TCNConfig实例
            **kwargs: 额外参数（会覆盖config）
        """
        if config is None:
            config = TCNConfig()
        
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
        
        logger.info(f"FluidTCN initialized: {self.get_model_info()}")
    
    def _build_model(self):
        """构建TCN模型架构。"""
        config = self.config
        
        # 分别处理boundary和equipment变量
        # Boundary变量处理 (前538维)
        self.boundary_projection = nn.Sequential(
            nn.Linear(config.boundary_dims, config.boundary_hidden_dim),
            nn.ReLU() if config.activation == "relu" else (
                nn.GELU() if config.activation == "gelu" else nn.Tanh()),
            nn.BatchNorm1d(config.boundary_hidden_dim) if config.use_norm else nn.Identity(),
            nn.Dropout(config.dropout)
        )
        
        # Equipment变量处理 (后6174维)
        self.equipment_projection = nn.Sequential(
            nn.Linear(config.equipment_dims, config.equipment_hidden_dim),
            nn.ReLU() if config.activation == "relu" else (
                nn.GELU() if config.activation == "gelu" else nn.Tanh()),
            nn.BatchNorm1d(config.equipment_hidden_dim) if config.use_norm else nn.Identity(),
            nn.Dropout(config.dropout)
        )
        
        # TCN网络 - 处理boundary特征
        self.boundary_tcn = TemporalConvNet(
            num_inputs=config.boundary_hidden_dim,
            num_channels=config.num_channels.copy(),
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            use_norm=config.use_norm,
            activation=config.activation,
            use_residual=config.use_residual
        )
        
        # TCN网络 - 处理equipment特征
        self.equipment_tcn = TemporalConvNet(
            num_inputs=config.equipment_hidden_dim,
            num_channels=config.num_channels.copy(),
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            use_norm=config.use_norm,
            activation=config.activation,
            use_residual=config.use_residual
        )
        
        # 特征融合层
        total_hidden_dim = config.num_channels[-1] * 2  # boundary和equipment特征的组合
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_hidden_dim, config.projection_hidden_dim),
            nn.ReLU() if config.activation == "relu" else (
                nn.GELU() if config.activation == "gelu" else nn.Tanh()),
            nn.BatchNorm1d(config.projection_hidden_dim) if config.use_norm else nn.Identity(),
            nn.Dropout(config.dropout)
        )
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(config.projection_hidden_dim, config.projection_hidden_dim // 2),
            nn.ReLU() if config.activation == "relu" else (
                nn.GELU() if config.activation == "gelu" else nn.Tanh()),
            nn.BatchNorm1d(config.projection_hidden_dim // 2) if config.use_norm else nn.Identity(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projection_hidden_dim // 2, config.output_dim)
        )
    
    def _initialize_weights(self):
        """初始化模型权重。"""
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
        FluidTCN前向传播，兼容transformers格式。
        
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
        
        # Transpose for TCN: [B, T, C] -> [B, C, T]
        boundary_features = boundary_features.transpose(1, 2)  # [B, boundary_hidden_dim, T]
        
        # 应用TCN到boundary特征
        boundary_tcn_out = self.boundary_tcn(boundary_features)  # [B, num_channels[-1], T]
        
        # 处理equipment变量
        # Reshape for linear projection: [B, T, 6174] -> [B*T, 6174]
        equipment_reshaped = equipment_data.reshape(-1, self.equipment_dims)
        equipment_features = self.equipment_projection(equipment_reshaped)  # [B*T, equipment_hidden_dim]
        equipment_features = equipment_features.reshape(batch_size, time_steps, -1)  # [B, T, equipment_hidden_dim]
        
        # Transpose for TCN: [B, T, C] -> [B, C, T]
        equipment_features = equipment_features.transpose(1, 2)  # [B, equipment_hidden_dim, T]
        
        # 应用TCN到equipment特征
        equipment_tcn_out = self.equipment_tcn(equipment_features)  # [B, num_channels[-1], T]
        
        # 融合特征
        # Transpose back: [B, C, T] -> [B, T, C]
        boundary_tcn_out = boundary_tcn_out.transpose(1, 2)  # [B, T, num_channels[-1]]
        equipment_tcn_out = equipment_tcn_out.transpose(1, 2)  # [B, T, num_channels[-1]]
        
        # 连接特征
        combined_features = torch.cat([boundary_tcn_out, equipment_tcn_out], dim=-1)  # [B, T, 2*num_channels[-1]]
        
        # 特征融合
        # Reshape for linear layer: [B, T, C] -> [B*T, C]
        combined_reshaped = combined_features.reshape(-1, combined_features.size(-1))
        fused_features = self.feature_fusion(combined_reshaped)  # [B*T, projection_hidden_dim]
        fused_features = fused_features.reshape(batch_size, time_steps, -1)  # [B, T, projection_hidden_dim]
        
        # 输出投影
        # Reshape for linear layer: [B, T, C] -> [B*T, C]
        output_reshaped = fused_features.reshape(-1, fused_features.size(-1))
        predictions = self.output_projection(output_reshaped)  # [B*T, output_dim]
        predictions = predictions.reshape(batch_size, time_steps, self.output_dim)  # [B, T, V]
        
        # 返回格式兼容transformers
        if labels is not None:
            # 计算loss
            loss = self.compute_loss(predictions, labels, prediction_mask)
            return {'loss': loss, 'logits': predictions}
        else:
            # 只返回预测结果
            return {'logits': predictions}
    
    def get_model_info(self) -> Dict:
        """获取详细的模型信息。"""
        base_info = super().get_model_info()
        
        tcn_info = {
            'num_channels': self.config.num_channels,
            'kernel_size': self.config.kernel_size,
            'boundary_hidden_dim': self.config.boundary_hidden_dim,
            'equipment_hidden_dim': self.config.equipment_hidden_dim,
            'projection_hidden_dim': self.config.projection_hidden_dim,
            'use_residual': self.config.use_residual,
            'use_norm': self.config.use_norm,
            'activation': self.config.activation,
            'dropout': self.config.dropout
        }
        
        base_info.update(tcn_info)
        return base_info