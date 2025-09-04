"""
Decoder-only model for fluid dynamics time series prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
import logging

from ..base import BaseModel
from .config import DecoderConfig

logger = logging.getLogger(__name__)


class CombinedPositionalEncoding(nn.Module):
    """组合位置编码：时间维度 + 变量维度。"""
    
    def __init__(self, d_model: int, max_time_positions: int = 10, 
                 max_variable_positions: int = 6712, 
                 time_encoding_type: str = "sinusoidal",
                 variable_encoding_type: str = "sinusoidal"):
        super().__init__()
        self.d_model = d_model
        self.max_time_positions = max_time_positions
        self.max_variable_positions = max_variable_positions
        
        # 时间位置编码
        if time_encoding_type == "sinusoidal":
            self.time_pe = self._create_sinusoidal_encoding(max_time_positions, d_model)
            self.register_buffer('time_pe_buffer', self.time_pe)
        else:
            self.time_pe = nn.Embedding(max_time_positions, d_model)
        
        # 变量位置编码
        if variable_encoding_type == "sinusoidal":
            self.variable_pe = self._create_sinusoidal_encoding(max_variable_positions, d_model)
            self.register_buffer('variable_pe_buffer', self.variable_pe)
        else:
            self.variable_pe = nn.Embedding(max_variable_positions, d_model)
        
        self.time_encoding_type = time_encoding_type
        self.variable_encoding_type = variable_encoding_type
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建正弦-余弦位置编码。"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: torch.Tensor, time_steps: int, num_variables: int) -> torch.Tensor:
        """
        添加组合位置编码到输入。
        
        Args:
            x: 输入张量 [B, T*V, d_model]
            time_steps: 时间步数 T
            num_variables: 变量数 V
            
        Returns:
            位置编码后的张量 [B, T*V, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 创建时间和变量索引
        time_indices = torch.arange(time_steps, device=x.device).repeat_interleave(num_variables)  # [T*V]
        variable_indices = torch.arange(num_variables, device=x.device).repeat(time_steps)  # [T*V]
        
        # 获取位置编码
        if self.time_encoding_type == "sinusoidal":
            time_encoding = self.time_pe_buffer[time_indices]  # [T*V, d_model]
        else:
            time_encoding = self.time_pe(time_indices)  # [T*V, d_model]
        
        if self.variable_encoding_type == "sinusoidal":
            variable_encoding = self.variable_pe_buffer[variable_indices]  # [T*V, d_model]
        else:
            variable_encoding = self.variable_pe(variable_indices)  # [T*V, d_model]
        
        # 组合编码并添加到输入
        combined_encoding = time_encoding + variable_encoding  # [T*V, d_model]
        combined_encoding = combined_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T*V, d_model]
        
        return x + combined_encoding


class DecoderAttentionMask:
    """生成Decoder的attention mask。"""
    
    @staticmethod
    def create_decoder_mask(batch_size: int, time_steps: int, num_variables: int, 
                          boundary_dims: int = 538, device: torch.device = None) -> torch.Tensor:
        """
        创建decoder的attention mask。
        
        Args:
            batch_size: 批次大小 B
            time_steps: 时间步数 T
            num_variables: 变量数 V
            boundary_dims: boundary变量维度数
            device: 设备
            
        Returns:
            attention_mask: [B, T*V, T*V] 
        """
        if device is None:
            device = torch.device('cpu')
        
        # 确保boundary_dims不超过num_variables
        boundary_dims = min(boundary_dims, num_variables)
        
        seq_len = time_steps * num_variables
        mask = torch.zeros(batch_size, seq_len, seq_len, device=device)
        
        for t in range(time_steps):
            t_start = t * num_variables
            t_end = (t + 1) * num_variables
            
            # Boundary变量：只能看到同一时间步的自己
            boundary_end = t_start + boundary_dims
            for i in range(t_start, min(boundary_end, t_end)):  # 确保不超出边界
                mask[:, i, i] = 1.0  # 只能看到自己
            
            # Equipment变量：因果mask，能看到当前及之前时间步的所有equipment变量
            equipment_start = t_start + boundary_dims
            for i in range(equipment_start, t_end):
                # 能看到当前及之前时间步的所有equipment变量
                for prev_t in range(t + 1):
                    prev_eq_start = prev_t * num_variables + boundary_dims
                    prev_eq_end = (prev_t + 1) * num_variables
                    mask[:, i, prev_eq_start:prev_eq_end] = 1.0
        
        return mask


class SimpleMultiHeadAttention(nn.Module):
    """简化的多头注意力机制。"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        多头自注意力前向传播。
        
        Args:
            x: 输入张量 [B, T*V, d_model]
            attention_mask: 注意力mask [B, T*V, T*V]
            
        Returns:
            输出张量 [B, T*V, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 线性投影和重塑
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T*V, d_k]
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T*V, d_k]
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T*V, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, T*V, T*V]
        
        # 应用attention mask
        if attention_mask is not None:
            # 扩展mask到多头维度
            attention_mask = attention_mask.unsqueeze(1)  # [B, 1, T*V, T*V]
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax和dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力到values
        output = torch.matmul(attention_weights, V)  # [B, H, T*V, d_k]
        
        # 重塑和输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # [B, T*V, d_model]
        output = self.w_o(output)
        
        return output


class DecoderBlock(nn.Module):
    """简化的Decoder block。"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        
        self.attention = SimpleMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decoder block前向传播。
        
        Args:
            x: 输入张量 [B, T*V, d_model]
            attention_mask: 注意力mask [B, T*V, T*V]
            
        Returns:
            输出张量 [B, T*V, d_model]
        """
        # Self-attention with residual connection
        attn_out = self.attention(x, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class FluidDecoder(BaseModel):
    """
    纯Decoder模型，用于天然气管网流体动力学预测。
    
    架构：
    1. 输入重塑：[B, T, V] -> [B, T*V, d_model]
    2. 组合位置编码：时间编码 + 变量编码
    3. Decoder层堆叠
    4. 输出投影：[B, T*V, d_model] -> [B, T, V]
    """
    
    def __init__(self, config: Optional[DecoderConfig] = None, **kwargs):
        """
        初始化FluidDecoder模型。
        
        Args:
            config: DecoderConfig实例
            **kwargs: 额外参数（会覆盖config）
        """
        if config is None:
            config = DecoderConfig()
        
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
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        
        # 构建模型
        self._build_model()
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info(f"FluidDecoder initialized: {self.get_model_info()}")
    
    def _build_model(self):
        """构建decoder模型架构。"""
        config = self.config
        
        # 输入投影（简化版本）- 将标量值映射到d_model维度
        self.input_projection = nn.Sequential(
            nn.Linear(1, config.projection_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.projection_hidden_dim),
            nn.Linear(config.projection_hidden_dim, config.d_model)
        )
        
        # 组合位置编码
        self.pos_encoding = CombinedPositionalEncoding(
            d_model=config.d_model,
            max_time_positions=config.max_time_positions,
            max_variable_positions=config.max_variable_positions,
            time_encoding_type=config.time_position_encoding,
            variable_encoding_type=config.variable_position_encoding
        )
        
        self.pos_dropout = nn.Dropout(config.dropout_rate)
        
        # Decoder层
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.attention_dropout,
                activation=config.activation
            ) for _ in range(config.n_layers)
        ])
        
        # 最终归一化
        if config.use_layer_norm:
            self.final_norm = nn.LayerNorm(config.d_model)
        else:
            self.final_norm = None
        
        # 输出投影（简化版本）- 将d_model维度映射回标量值
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model, config.projection_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.projection_hidden_dim),
            nn.Linear(config.projection_hidden_dim, 1)
        )
    
    def _initialize_weights(self):
        """初始化模型权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids=None, labels=None, prediction_mask=None, **kwargs):
        """
        FluidDecoder前向传播，兼容transformers格式。
        
        Args:
            input_ids: 输入张量 [B, T, V=6712] 或包含'input'键的字典
            labels: 目标张量 [B, T, V=6712] 用于损失计算 (可选)
            prediction_mask: 预测mask [B, V] (可选)
            **kwargs: 额外参数
        
        Returns:
            如果提供labels: {'loss': tensor, 'logits': tensor}
            否则: {'logits': tensor} 或 tensor
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
        
        # 重塑输入：[B, T, V] -> [B, T*V, 1] 
        # 每个变量值当作一个token
        x_reshaped = x.view(batch_size, -1, 1)  # [B, T*V, 1]
        
        # 输入投影：[B, T*V, 1] -> [B, T*V, d_model]  
        x = self.input_projection(x_reshaped)  # [B, T*V, d_model]
        
        # 添加位置编码
        x = self.pos_encoding(x, time_steps, num_variables)
        x = self.pos_dropout(x)
        
        # 创建attention mask
        attention_mask = DecoderAttentionMask.create_decoder_mask(
            batch_size, time_steps, num_variables, 
            boundary_dims=self.boundary_dims, device=x.device
        )  # [B, T*V, T*V]
        
        # 通过decoder层
        for block in self.decoder_blocks:
            x = block(x, attention_mask)
        
        # 最终归一化
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # 输出投影：[B, T*V, d_model] -> [B, T*V, 1]
        predictions = self.output_projection(x)  # [B, T*V, 1]
        
        # 重塑回原始维度：[B, T*V, 1] -> [B, T, V]
        predictions = predictions.squeeze(-1).view(batch_size, time_steps, num_variables)
        
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
        
        decoder_info = {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.config.d_ff,
            'time_position_encoding': self.config.time_position_encoding,
            'variable_position_encoding': self.config.variable_position_encoding
        }
        
        base_info.update(decoder_info)
        return base_info