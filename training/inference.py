"""
推理接口 for FluidDecoder 模型。

提供训练后模型的推理功能，支持单步预测和批量预测。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
import json

from transformers import Trainer
from .hf_integration import FluidDecoderForTraining, FluidDecoderConfig
from data import DataNormalizer

logger = logging.getLogger(__name__)


class FluidInference:
    """
    FluidDecoder 推理接口。
    
    支持:
    - 单步预测
    - 批量预测
    - 自回归生成
    - 结果反归一化
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        normalizer_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        """
        初始化推理接口。
        
        Args:
            model_path: 训练好的模型路径
            normalizer_path: 数据归一化器路径 (可选)
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
        self.model_path = Path(model_path)
        self.normalizer_path = Path(normalizer_path) if normalizer_path else None
        
        # 设备选择
        if device == "auto" or device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 加载模型
        self.model = self._load_model()
        self.model.eval()
        
        # 加载归一化器 (如果存在)
        self.normalizer = self._load_normalizer()
        
        logger.info(f"FluidInference initialized on {self.device}")
    
    def _load_model(self) -> FluidDecoderForTraining:
        """加载训练好的模型。"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            model = FluidDecoderForTraining.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32
            )
            model.to(self.device)
            
            logger.info(f"Model loaded successfully: {model.num_parameters():,} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise
    
    def _load_normalizer(self) -> Optional[DataNormalizer]:
        """加载数据归一化器。"""
        if self.normalizer_path is None:
            logger.warning("No normalizer path provided, predictions will be in normalized space")
            return None
        
        try:
            normalizer = DataNormalizer.load(str(self.normalizer_path))
            logger.info(f"Normalizer loaded from {self.normalizer_path}")
            return normalizer
            
        except Exception as e:
            logger.warning(f"Failed to load normalizer: {e}")
            return None
    
    def predict_single(
        self,
        input_data: np.ndarray,
        prediction_mask: Optional[np.ndarray] = None,
        denormalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        单样本预测。
        
        Args:
            input_data: 输入数据 [T, V=6712]
            prediction_mask: 预测掩码 [V] (可选)
            denormalize: 是否反归一化结果
            
        Returns:
            预测结果字典，包含:
            - predictions: 预测结果 [T, V]
            - input: 输入数据 [T, V]
            - mask: 预测掩码 [V]
        """
        # 输入验证
        if input_data.ndim != 2 or input_data.shape[1] != 6712:
            raise ValueError(f"Input data should be [T, 6712], got {input_data.shape}")
        
        # 转换为 tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # [1, T, V]
        input_tensor = input_tensor.to(self.device)
        
        # 创建预测掩码
        if prediction_mask is None:
            prediction_mask = np.ones(6712)
            prediction_mask[:538] = 0  # 边界条件不预测
        
        mask_tensor = torch.tensor(prediction_mask, dtype=torch.float32).unsqueeze(0)  # [1, V]
        mask_tensor = mask_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                prediction_mask=mask_tensor
            )
            predictions = outputs.logits  # [1, T, V]
        
        # 转回 numpy
        predictions_np = predictions.cpu().numpy()[0]  # [T, V]
        
        # 反归一化 (如果需要)
        if denormalize and self.normalizer is not None:
            predictions_np = self.normalizer.inverse_transform(predictions_np)
            input_data = self.normalizer.inverse_transform(input_data)
        
        return {
            'predictions': predictions_np,
            'input': input_data,
            'mask': prediction_mask
        }
    
    def predict_batch(
        self,
        input_batch: np.ndarray,
        prediction_masks: Optional[np.ndarray] = None,
        batch_size: int = 32,
        denormalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        批量预测。
        
        Args:
            input_batch: 输入批次 [B, T, V=6712]
            prediction_masks: 预测掩码批次 [B, V] (可选)
            batch_size: 推理批次大小
            denormalize: 是否反归一化结果
            
        Returns:
            批量预测结果字典
        """
        # 输入验证
        if input_batch.ndim != 3 or input_batch.shape[2] != 6712:
            raise ValueError(f"Input batch should be [B, T, 6712], got {input_batch.shape}")
        
        num_samples = input_batch.shape[0]
        time_steps = input_batch.shape[1]
        
        # 创建预测掩码
        if prediction_masks is None:
            prediction_masks = np.ones((num_samples, 6712))
            prediction_masks[:, :538] = 0
        
        # 批量推理
        all_predictions = []
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            
            # 当前批次数据
            batch_input = input_batch[i:end_idx]
            batch_masks = prediction_masks[i:end_idx]
            
            # 转换为 tensor
            input_tensor = torch.tensor(batch_input, dtype=torch.float32).to(self.device)
            mask_tensor = torch.tensor(batch_masks, dtype=torch.float32).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    prediction_mask=mask_tensor
                )
                batch_predictions = outputs.logits.cpu().numpy()
            
            all_predictions.append(batch_predictions)
        
        # 合并结果
        predictions = np.concatenate(all_predictions, axis=0)  # [B, T, V]
        
        # 反归一化 (如果需要)
        if denormalize and self.normalizer is not None:
            original_shape = predictions.shape
            predictions = predictions.reshape(-1, 6712)
            predictions = self.normalizer.inverse_transform(predictions)
            predictions = predictions.reshape(original_shape)
            
            # 同样处理输入数据
            input_original_shape = input_batch.shape
            input_batch = input_batch.reshape(-1, 6712)
            input_batch = self.normalizer.inverse_transform(input_batch)
            input_batch = input_batch.reshape(input_original_shape)
        
        return {
            'predictions': predictions,
            'input': input_batch,
            'masks': prediction_masks
        }
    
    def predict_autoregressive(
        self,
        initial_input: np.ndarray,
        steps: int,
        prediction_mask: Optional[np.ndarray] = None,
        denormalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        自回归预测：使用模型输出作为下一步输入。
        
        Args:
            initial_input: 初始输入 [T, V=6712]
            steps: 预测步数
            prediction_mask: 预测掩码 [V]
            denormalize: 是否反归一化结果
            
        Returns:
            自回归预测结果
        """
        if initial_input.ndim != 2 or initial_input.shape[1] != 6712:
            raise ValueError(f"Initial input should be [T, 6712], got {initial_input.shape}")
        
        time_steps = initial_input.shape[0]
        current_input = initial_input.copy()
        
        # 创建预测掩码
        if prediction_mask is None:
            prediction_mask = np.ones(6712)
            prediction_mask[:538] = 0
        
        predictions_sequence = []
        
        for step in range(steps):
            logger.debug(f"Autoregressive step {step + 1}/{steps}")
            
            # 单步预测
            result = self.predict_single(
                current_input, 
                prediction_mask, 
                denormalize=False  # 暂不反归一化，最后统一处理
            )
            
            prediction = result['predictions']
            predictions_sequence.append(prediction)
            
            # 更新输入：使用预测结果作为下一步输入
            # 只更新需要预测的部分，边界条件保持原样
            mask_expanded = prediction_mask[np.newaxis, :]  # [1, V]
            
            # 移位时间维度：[t1, t2, t3] -> [t2, t3, pred]
            new_input = np.zeros_like(current_input)
            new_input[:-1] = current_input[1:]  # 前移时间步
            new_input[-1] = current_input[-1]   # 先复制最后一步
            
            # 用预测结果更新最后一步的预测部分
            new_input[-1] = np.where(mask_expanded, prediction[-1], new_input[-1])
            
            current_input = new_input
        
        # 合并所有预测
        predictions_array = np.array(predictions_sequence)  # [steps, T, V]
        
        # 反归一化 (如果需要)
        if denormalize and self.normalizer is not None:
            original_shape = predictions_array.shape
            predictions_array = predictions_array.reshape(-1, 6712)
            predictions_array = self.normalizer.inverse_transform(predictions_array)
            predictions_array = predictions_array.reshape(original_shape)
            
            # 同样处理初始输入
            initial_input = self.normalizer.inverse_transform(initial_input)
        
        return {
            'predictions': predictions_array,
            'initial_input': initial_input,
            'steps': steps,
            'mask': prediction_mask
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息。"""
        config = self.model.config
        
        info = {
            'model_type': config.model_type,
            'num_parameters': self.model.num_parameters(),
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'input_dim': config.input_dim,
            'output_dim': config.output_dim,
            'boundary_dims': config.boundary_dims,
            'device': str(self.device),
            'has_normalizer': self.normalizer is not None
        }
        
        return info


def load_inference_model(
    model_path: Union[str, Path],
    normalizer_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None
) -> FluidInference:
    """
    加载推理模型的便捷函数。
    
    Args:
        model_path: 模型路径
        normalizer_path: 归一化器路径
        device: 计算设备
        
    Returns:
        FluidInference 实例
    """
    return FluidInference(model_path, normalizer_path, device)