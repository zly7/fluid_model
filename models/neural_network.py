import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class FluidNet(nn.Module):
    """天然气管网数值模拟神经网络模型"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dims: Dict[str, int],
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.1,
                 activation: str = 'relu'):
        """
        Args:
            input_dim: 输入维度
            output_dims: 输出维度字典，键为输出文件名，值为对应维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: dropout比例
            activation: 激活函数类型
        """
        super(FluidNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.output_files = list(output_dims.keys())
        
        # 共享的编码器层
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 为每个输出文件创建专门的解码器头
        self.decoders = nn.ModuleDict()
        for output_file, output_dim in output_dims.items():
            decoder = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                self._get_activation(activation),
                nn.BatchNorm1d(prev_dim // 2),
                nn.Dropout(dropout_rate),
                nn.Linear(prev_dim // 2, output_dim)
            )
            self.decoders[output_file] = decoder
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            return nn.ELU(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim] 或 [batch_size, input_dim]
            
        Returns:
            输出字典，键为输出文件名，值为对应的输出张量
        """
        # 处理输入维度
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.view(-1, self.input_dim)  # [batch_size * seq_len, input_dim]
            is_sequential = True
        else:
            batch_size, seq_len = x.shape[0], 1
            is_sequential = False
        
        # 编码
        encoded = self.encoder(x)  # [batch_size * seq_len, hidden_dim]
        
        # 解码
        outputs = {}
        for output_file in self.output_files:
            decoder_output = self.decoders[output_file](encoded)
            
            if is_sequential:
                # 恢复序列维度
                decoder_output = decoder_output.view(batch_size, seq_len, -1)
            
            outputs[output_file] = decoder_output
        
        return outputs


class PhysicsInformedNet(FluidNet):
    """物理信息神经网络(PINN)"""
    
    def __init__(self,
                 input_dim: int,
                 output_dims: Dict[str, int],
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.1,
                 activation: str = 'tanh',
                 physics_weight: float = 0.1):
        """
        Args:
            physics_weight: 物理约束损失权重
        """
        super(PhysicsInformedNet, self).__init__(
            input_dim, output_dims, hidden_dims, dropout_rate, activation
        )
        
        self.physics_weight = physics_weight
    
    def compute_physics_loss(self, 
                           inputs: torch.Tensor, 
                           outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算物理约束损失
        
        这里实现简单的质量守恒和能量守恒约束
        在实际应用中需要根据具体的物理方程来实现
        """
        physics_loss = 0.0
        
        # 示例：简单的连续性方程约束
        # 实际实现需要根据管网的拓扑结构和物理方程
        for output_name, output_tensor in outputs.items():
            if 'q_in' in str(output_name) or 'q_out' in str(output_name):
                # 流量连续性约束
                flow_continuity_loss = torch.mean(torch.abs(
                    torch.gradient(output_tensor, dim=-1)[0]
                ))
                physics_loss += flow_continuity_loss
            
            if 'p_in' in str(output_name) or 'p_out' in str(output_name):
                # 压力梯度约束
                pressure_gradient_loss = torch.mean(torch.abs(
                    torch.gradient(output_tensor, dim=-1)[0]
                ))
                physics_loss += pressure_gradient_loss * 0.1
        
        return physics_loss
    
    def forward_with_physics(self, 
                           x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """带物理约束的前向传播"""
        outputs = self.forward(x)
        physics_loss = self.compute_physics_loss(x, outputs)
        
        return outputs, physics_loss


class AttentionFluidNet(nn.Module):
    """带注意力机制的流体网络"""
    
    def __init__(self,
                 input_dim: int,
                 output_dims: Dict[str, int],
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout_rate: float = 0.1):
        """
        Args:
            num_heads: 注意力头数量
            num_layers: transformer层数量
        """
        super(AttentionFluidNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.output_files = list(output_dims.keys())
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出头
        self.output_heads = nn.ModuleDict()
        for output_file, output_dim in output_dims.items():
            self.output_heads[output_file] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Transformer编码
        encoded = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        
        # 全局平均池化或使用最后一个时间步
        if encoded.dim() == 3:
            encoded = encoded.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 输出预测
        outputs = {}
        for output_file in self.output_files:
            outputs[output_file] = self.output_heads[output_file](encoded)
        
        return outputs