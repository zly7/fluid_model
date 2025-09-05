"""
TCN (Temporal Convolutional Network) configuration.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import json

from ..config import ModelConfig


@dataclass
class TCNConfig(ModelConfig):
    """
    TCN模型配置类。
    
    TCN核心参数：
        num_channels: TCN层的通道数列表，每一层都会对应一个通道数
        kernel_size: 卷积核大小
        dropout: Dropout概率
        use_residual: 是否使用残差连接
        use_norm: 是否使用批归一化
        activation: 激活函数类型 ('relu', 'gelu', 'tanh')
        
    Architecture参数：
        boundary_hidden_dim: Boundary变量的隐藏层维度
        equipment_hidden_dim: Equipment变量的隐藏层维度
        projection_hidden_dim: 输出投影层的隐藏层维度
    """
    
    # TCN核心参数
    num_channels: List[int] = field(default_factory=lambda: [64, 128, 128, 64])
    kernel_size: int = 3
    dropout: float = 0.1
    use_residual: bool = True
    use_norm: bool = True
    activation: str = "relu"  # 'relu', 'gelu', 'tanh'
    
    # 架构参数
    boundary_hidden_dim: int = 64
    equipment_hidden_dim: int = 128
    projection_hidden_dim: int = 128
    
    # 数据维度（这些通常由数据决定）
    boundary_dims: int = 538
    equipment_dims: int = 6174
    
    def __post_init__(self):
        
        # 验证参数
        if not self.num_channels:
            raise ValueError("num_channels cannot be empty")
        
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        
        if not (0 <= self.dropout <= 1):
            raise ValueError("dropout must be between 0 and 1")
        
        if self.activation not in ['relu', 'gelu', 'tanh']:
            raise ValueError("activation must be one of: 'relu', 'gelu', 'tanh'")
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TCNConfig':
        """从JSON文件加载配置。"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TCNConfig':
        """从字典创建配置。"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'sequence_length': self.sequence_length,
            'boundary_dims': self.boundary_dims,
            'equipment_dims': self.equipment_dims,
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'use_residual': self.use_residual,
            'use_norm': self.use_norm,
            'activation': self.activation,
            'boundary_hidden_dim': self.boundary_hidden_dim,
            'equipment_hidden_dim': self.equipment_hidden_dim,
            'projection_hidden_dim': self.projection_hidden_dim,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }
    
    def save_to_json(self, json_path: str):
        """保存配置到JSON文件。"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)