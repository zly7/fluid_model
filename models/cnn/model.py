"""
CNN model for fluid dynamics time series prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from ..base import BaseModel
from .config import CNNConfig

logger = logging.getLogger(__name__)


class MultiScaleConv1D(nn.Module):
    """多尺度1D卷积模块，捕获不同时间尺度的特征。"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_sizes: tuple = (3, 5, 7), 
                 dilation_rates: tuple = (1, 2, 4),
                 activation: str = "relu",
                 batch_norm: bool = True):
        super().__init__()
        
        self.conv_branches = nn.ModuleList()
        branch_out_channels = out_channels // len(kernel_sizes)
        
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilation_rates)):
            padding = (kernel_size - 1) * dilation // 2  # Same padding
            
            branch = nn.Sequential()
            branch.add_module(f'conv_{i}', 
                nn.Conv1d(in_channels, branch_out_channels, 
                         kernel_size=kernel_size, 
                         dilation=dilation,
                         padding=padding))
            
            if batch_norm:
                branch.add_module(f'bn_{i}', nn.BatchNorm1d(branch_out_channels))
            
            if activation == "relu":
                branch.add_module(f'act_{i}', nn.ReLU())
            elif activation == "gelu":
                branch.add_module(f'act_{i}', nn.GELU())
            elif activation == "leaky_relu":
                branch.add_module(f'act_{i}', nn.LeakyReLU())
            
            self.conv_branches.append(branch)
        
        # 调整最后一个分支的输出通道数以匹配目标维度
        remaining_channels = out_channels - (len(kernel_sizes) - 1) * branch_out_channels
        if remaining_channels != branch_out_channels and len(self.conv_branches) > 0:
            # 重新创建最后一个分支
            last_kernel, last_dilation = kernel_sizes[-1], dilation_rates[-1]
            padding = (last_kernel - 1) * last_dilation // 2
            
            last_branch = nn.Sequential()
            last_branch.add_module('conv_last', 
                nn.Conv1d(in_channels, remaining_channels, 
                         kernel_size=last_kernel, 
                         dilation=last_dilation,
                         padding=padding))
            
            if batch_norm:
                last_branch.add_module('bn_last', nn.BatchNorm1d(remaining_channels))
            
            if activation == "relu":
                last_branch.add_module('act_last', nn.ReLU())
            elif activation == "gelu":
                last_branch.add_module('act_last', nn.GELU())
            elif activation == "leaky_relu":
                last_branch.add_module('act_last', nn.LeakyReLU())
            
            self.conv_branches[-1] = last_branch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量 [B, C, T]
            
        Returns:
            输出张量 [B, out_channels, T]
        """
        branch_outputs = []
        for branch in self.conv_branches:
            branch_outputs.append(branch(x))
        
        # 连接所有分支的输出
        output = torch.cat(branch_outputs, dim=1)  # [B, out_channels, T]
        return output


class TemporalCNNBlock(nn.Module):
    """时间CNN块，包含残差连接。"""
    
    def __init__(self, channels: int, kernel_sizes: tuple = (3, 5, 7),
                 dilation_rates: tuple = (1, 2, 4),
                 activation: str = "relu",
                 batch_norm: bool = True,
                 use_residual: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.use_residual = use_residual
        
        # 多尺度卷积
        self.multi_scale_conv = MultiScaleConv1D(
            in_channels=channels,
            out_channels=channels,
            kernel_sizes=kernel_sizes,
            dilation_rates=dilation_rates,
            activation=activation,
            batch_norm=batch_norm
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 可选的残差投影（如果需要匹配维度）
        self.residual_proj = None
        if use_residual:
            # 这里假设输入输出维度相同，如果不同需要投影
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量 [B, C, T]
            
        Returns:
            输出张量 [B, C, T]
        """
        identity = x
        
        # 多尺度卷积
        out = self.multi_scale_conv(x)
        out = self.dropout(out)
        
        # 残差连接
        if self.use_residual and self.residual_proj is not None:
            identity = self.residual_proj(identity)
            out = out + identity
        
        return out


class FluidCNN(BaseModel):
    """
    CNN模型，用于天然气管网流体动力学预测。
    
    架构：
    1. 分别处理boundary和equipment变量
    2. 多尺度时间卷积捕获时空模式
    3. 残差连接和批归一化
    4. 输出投影回原始维度
    """
    
    def __init__(self, config: Optional[CNNConfig] = None, **kwargs):
        """
        初始化FluidCNN模型。
        
        Args:
            config: CNNConfig实例
            **kwargs: 额外参数（会覆盖config）
        """
        if config is None:
            config = CNNConfig()
        
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
        
        logger.info(f"FluidCNN initialized: {self.get_model_info()}")
    
    def _build_model(self):
        """构建CNN模型架构。"""
        config = self.config
        
        # 分别处理boundary和equipment变量
        # Boundary变量处理 (前538维)
        self.boundary_projection = nn.Sequential(
            nn.Linear(config.boundary_dims, config.boundary_hidden_dim),
            nn.ReLU() if config.activation == "relu" else nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Equipment变量处理 (后6174维)
        self.equipment_projection = nn.Sequential(
            nn.Linear(config.equipment_dims, config.equipment_hidden_dim),
            nn.ReLU() if config.activation == "relu" else nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 时间CNN层 - 处理boundary特征
        self.boundary_temporal_layers = nn.ModuleList([
            TemporalCNNBlock(
                channels=config.boundary_hidden_dim,
                kernel_sizes=config.kernel_sizes,
                dilation_rates=config.dilation_rates,
                activation=config.activation,
                batch_norm=False, 
                use_residual=config.use_residual,
                dropout=config.dropout_rate
            ) for _ in range(config.num_conv_layers)
        ])
        
        # 时间CNN层 - 处理equipment特征
        self.equipment_temporal_layers = nn.ModuleList([
            TemporalCNNBlock(
                channels=config.equipment_hidden_dim,
                kernel_sizes=config.kernel_sizes,
                dilation_rates=config.dilation_rates,
                activation=config.activation,
                batch_norm=False, 
                use_residual=config.use_residual,
                dropout=config.dropout_rate
            ) for _ in range(config.num_conv_layers)
        ])
        
        # 特征融合层
        total_hidden_dim = config.boundary_hidden_dim + config.equipment_hidden_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_hidden_dim, config.hidden_channels),
            nn.ReLU() if config.activation == "relu" else nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 全局时间卷积层
        self.global_temporal_layers = nn.ModuleList([
            TemporalCNNBlock(
                channels=config.hidden_channels,
                kernel_sizes=config.kernel_sizes,
                dilation_rates=config.dilation_rates,
                activation=config.activation,
                batch_norm=False,  
                use_residual=config.use_residual,
                dropout=config.dropout_rate
            ) for _ in range(config.num_conv_layers // 2) 
        ])
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_channels, config.projection_hidden_dim),
            nn.ReLU() if config.activation == "relu" else nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.projection_hidden_dim, config.output_dim)
        )
    
    def _initialize_weights(self):
        """初始化模型权重。"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids=None, labels=None, prediction_mask=None, **kwargs):
        """
        FluidCNN前向传播，兼容transformers格式。
        
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
        
        # Transpose for conv1d: [B, T, C] -> [B, C, T]
        boundary_features = boundary_features.transpose(1, 2)  # [B, boundary_hidden_dim, T]
        
        # 应用时间CNN层到boundary特征
        for layer in self.boundary_temporal_layers:
            boundary_features = layer(boundary_features)  # [B, boundary_hidden_dim, T]
        
        # 处理equipment变量
        # Reshape for linear projection: [B, T, 6174] -> [B*T, 6174]
        equipment_reshaped = equipment_data.reshape(-1, self.equipment_dims)
        equipment_features = self.equipment_projection(equipment_reshaped)  # [B*T, equipment_hidden_dim]
        equipment_features = equipment_features.reshape(batch_size, time_steps, -1)  # [B, T, equipment_hidden_dim]
        
        # Transpose for conv1d: [B, T, C] -> [B, C, T]
        equipment_features = equipment_features.transpose(1, 2)  # [B, equipment_hidden_dim, T]
        
        # 应用时间CNN层到equipment特征
        for layer in self.equipment_temporal_layers:
            equipment_features = layer(equipment_features)  # [B, equipment_hidden_dim, T]
        
        # 融合特征
        # Transpose back: [B, C, T] -> [B, T, C]
        boundary_features = boundary_features.transpose(1, 2)  # [B, T, boundary_hidden_dim]
        equipment_features = equipment_features.transpose(1, 2)  # [B, T, equipment_hidden_dim]
        
        # 连接特征
        combined_features = torch.cat([boundary_features, equipment_features], dim=-1)  # [B, T, total_hidden_dim]
        
        # 特征融合
        # Reshape for linear layer: [B, T, C] -> [B*T, C]
        combined_reshaped = combined_features.reshape(-1, combined_features.size(-1))
        fused_features = self.feature_fusion(combined_reshaped)  # [B*T, hidden_channels]
        fused_features = fused_features.reshape(batch_size, time_steps, -1)  # [B, T, hidden_channels]
        
        # Transpose for global temporal layers: [B, T, C] -> [B, C, T]
        fused_features = fused_features.transpose(1, 2)  # [B, hidden_channels, T]
        
        # 全局时间建模
        for layer in self.global_temporal_layers:
            fused_features = layer(fused_features)  # [B, hidden_channels, T]
        
        # Transpose back for output projection: [B, C, T] -> [B, T, C]
        fused_features = fused_features.transpose(1, 2)  # [B, T, hidden_channels]
        
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
        
        cnn_info = {
            'hidden_channels': self.config.hidden_channels,
            'kernel_sizes': self.config.kernel_sizes,
            'num_conv_layers': self.config.num_conv_layers,
            'dilation_rates': self.config.dilation_rates,
            'boundary_hidden_dim': self.config.boundary_hidden_dim,
            'equipment_hidden_dim': self.config.equipment_hidden_dim,
            'use_residual': self.config.use_residual,
            'activation': self.config.activation
        }
        
        base_info.update(cnn_info)
        return base_info