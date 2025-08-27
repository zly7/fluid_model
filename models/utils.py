import torch
import random
import numpy as np
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42):
    """设置全局随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置PyTorch的随机数生成器状态
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    logger.info(f"Random seed set to {seed}")


def save_model(model: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               epoch: int,
               loss: float,
               save_path: str,
               additional_info: Dict[str, Any] = None):
    """保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        loss: 当前损失
        save_path: 保存路径
        additional_info: 额外信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'class_name': model.__class__.__name__,
            'input_dim': getattr(model, 'input_dim', None),
            'output_dims': getattr(model, 'output_dims', None),
        }
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(model: torch.nn.Module,
               load_path: str,
               optimizer: torch.optim.Optimizer = None,
               device: str = 'cpu') -> Dict[str, Any]:
    """加载模型检查点
    
    Args:
        model: 模型实例
        load_path: 模型路径
        optimizer: 优化器（可选）
        device: 设备
        
    Returns:
        包含epoch、loss等信息的字典
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    
    checkpoint = torch.load(load_path, map_location=device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态（如果提供）
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.to(device)
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'model_config': checkpoint.get('model_config', {})
    }
    
    logger.info(f"Model loaded from {load_path}, epoch: {info['epoch']}, loss: {info['loss']:.6f}")
    
    return info


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return trainable_params


def get_device(prefer_gpu: bool = True) -> str:
    """获取可用设备"""
    if prefer_gpu and torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        logger.info("Using CPU")
    
    return device


def initialize_weights(model: torch.nn.Module):
    """初始化模型权重"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Xavier/Glorot初始化
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, restore_best_weights: bool = True):
        """
        Args:
            patience: 容忍轮数
            min_delta: 最小改善幅度
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """检查是否需要早停
        
        Returns:
            是否需要停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best model weights")
            return True
        
        return False


def get_model_summary(model: torch.nn.Module, input_shape: tuple) -> str:
    """获取模型摘要信息"""
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    summary = []
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append(f"Input shape: {input_shape}")
    summary.append("-" * 50)
    
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            params = count_params(module)
            total_params += params
            summary.append(f"{name}: {module.__class__.__name__} - {params:,} params")
    
    summary.append("-" * 50)
    summary.append(f"Total trainable parameters: {total_params:,}")
    
    return "\n".join(summary)