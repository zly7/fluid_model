"""
Utility functions for model management and operations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Type
import logging
from pathlib import Path
import json
import warnings

from .base import BaseModel
from .decoder import FluidDecoder
from .cnn import FluidCNN
from .lstm import FluidLSTM, LSTMConfig
from .config import ModelConfig, DecoderConfig, CNNConfig, load_config_from_file

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def initialize_weights(model: nn.Module, method: str = 'xavier') -> None:
    """
    Initialize model weights using specified method.
    
    Args:
        model: PyTorch model to initialize
        method: Initialization method ('xavier', 'he', 'normal', 'zero')
    """
    def init_fn(m):
        if isinstance(m, nn.Linear):
            if method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'he':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif method == 'normal':
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif method == 'zero':
                nn.init.zeros_(m.weight)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
                
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1 for LSTM
                    if 'bias_ih' in name and isinstance(m, nn.LSTM):
                        hidden_size = param.size(0) // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
    
    model.apply(init_fn)
    logger.info(f"Model weights initialized using {method} method")


def create_model(model_type: str, config: Optional[Union[ModelConfig, Dict, str]] = None, **kwargs) -> BaseModel:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('decoder', 'cnn', 'lstm')
        config: Model configuration (Config object, dict, or path to JSON file)
        **kwargs: Additional parameters to override config
        
    Returns:
        Initialized model
    """
    # Load config if it's a file path
    if isinstance(config, (str, Path)):
        config = load_config_from_file(str(config))
    elif isinstance(config, dict):
        # Convert dict to appropriate config class
        if model_type.lower() == 'decoder':
            config = DecoderConfig.from_dict(config)
        elif model_type.lower() == 'cnn':
            config = CNNConfig.from_dict(config)
        elif model_type.lower() == 'lstm':
            config = LSTMConfig.from_dict(config)
        else:
            config = ModelConfig.from_dict(config)
    
    # Update config with kwargs
    if config is not None and kwargs:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create model based on type
    model_type = model_type.lower()
    
    if model_type == 'decoder':
        model = FluidDecoder(config)
    elif model_type == 'cnn':
        model = FluidCNN(config)
    elif model_type == 'lstm':
        model = FluidLSTM(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'decoder', 'cnn', 'lstm'")
    
    logger.info(f"Created {model_type} model with {count_parameters(model)} parameters")
    return model


def load_model(checkpoint_path: str, model_type: Optional[str] = None, 
               config: Optional[Union[ModelConfig, Dict]] = None, strict: bool = True) -> BaseModel:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model (if not in checkpoint)
        config: Model configuration (if not in checkpoint)
        strict: Whether to strictly match state dict keys
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model type and config from checkpoint
    saved_model_type = checkpoint.get('model_name', model_type)
    saved_config = checkpoint.get('model_config', config)
    
    if saved_model_type is None:
        raise ValueError("Model type not found in checkpoint and not provided")
    
    # Create model
    model = create_model(saved_model_type, saved_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    logger.info(f"Model loaded from {checkpoint_path}")
    return model


def save_model(model: BaseModel, save_path: str, epoch: int = 0, 
               optimizer_state: Optional[Dict] = None, **kwargs) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
        epoch: Training epoch
        optimizer_state: Optimizer state dict
        **kwargs: Additional information to save
    """
    model.save_checkpoint(save_path, epoch, optimizer_state, **kwargs)


def get_model_summary(model: BaseModel, input_shape: tuple = (1, 3, 6712)) -> Dict[str, Any]:
    """
    Get comprehensive model summary.
    
    Args:
        model: Model to analyze
        input_shape: Input tensor shape (B, T, V)
        
    Returns:
        Dictionary with model summary information
    """
    model.eval()
    
    # Create dummy input
    dummy_batch = {
        'input': torch.randn(*input_shape),
        'prediction_mask': torch.ones(input_shape[0], input_shape[2]),
        'attention_mask': torch.ones(input_shape)
    }
    
    # Forward pass to get output shape
    with torch.no_grad():
        output = model(dummy_batch)
    
    # Collect summary information
    summary = {
        'model_info': model.get_model_info(),
        'input_shape': input_shape,
        'output_shape': tuple(output.shape),
        'total_parameters': count_parameters(model, trainable_only=False),
        'trainable_parameters': count_parameters(model, trainable_only=True),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    }
    
    return summary


def compare_models(models: Dict[str, BaseModel], input_batch: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
    """
    Compare multiple models on the same input.
    
    Args:
        models: Dictionary of model name -> model
        input_batch: Input batch for comparison
        
    Returns:
        Comparison results
    """
    results = {}
    
    for name, model in models.items():
        model.eval()
        
        with torch.no_grad():
            # Time forward pass
            import time
            start_time = time.time()
            output = model(input_batch)
            forward_time = time.time() - start_time
            
            # Compute loss if targets available
            loss_info = {}
            if 'target' in input_batch:
                loss_info = model.compute_loss(input_batch, output)
                loss_info = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_info.items()}
        
        results[name] = {
            'output_shape': tuple(output.shape),
            'forward_time': forward_time,
            'parameters': count_parameters(model),
            'loss_info': loss_info
        }
    
    return results


def model_memory_usage(model: BaseModel) -> Dict[str, float]:
    """
    Estimate model memory usage.
    
    Args:
        model: Model to analyze
        
    Returns:
        Memory usage information in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Estimate gradient memory (same as parameters for most cases)
    grad_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    
    total_size = param_size + buffer_size + grad_size
    
    return {
        'parameters_mb': param_size / (1024 * 1024),
        'buffers_mb': buffer_size / (1024 * 1024),
        'gradients_mb': grad_size / (1024 * 1024),
        'total_mb': total_size / (1024 * 1024)
    }


def freeze_model_parts(model: BaseModel, freeze_patterns: list) -> None:
    """
    Freeze specific parts of the model based on name patterns.
    
    Args:
        model: Model to modify
        freeze_patterns: List of parameter name patterns to freeze
    """
    frozen_count = 0
    
    for name, param in model.named_parameters():
        for pattern in freeze_patterns:
            if pattern in name:
                param.requires_grad = False
                frozen_count += 1
                break
    
    logger.info(f"Frozen {frozen_count} parameters matching patterns: {freeze_patterns}")


def unfreeze_model_parts(model: BaseModel, unfreeze_patterns: list) -> None:
    """
    Unfreeze specific parts of the model based on name patterns.
    
    Args:
        model: Model to modify
        unfreeze_patterns: List of parameter name patterns to unfreeze
    """
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        for pattern in unfreeze_patterns:
            if pattern in name:
                param.requires_grad = True
                unfrozen_count += 1
                break
    
    logger.info(f"Unfrozen {unfrozen_count} parameters matching patterns: {unfreeze_patterns}")


def validate_model_config(config: ModelConfig) -> bool:
    """
    Validate model configuration.
    
    Args:
        config: Model configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Check basic dimensions
    if config.input_dim <= 0 or config.output_dim <= 0:
        raise ValueError("Input and output dimensions must be positive")
    
    if config.sequence_length <= 0:
        raise ValueError("Sequence length must be positive")
    
    # Check dropout rates
    if not 0 <= config.dropout_rate <= 1:
        raise ValueError("Dropout rate must be between 0 and 1")
    
    # Decoder-specific checks
    if isinstance(config, DecoderConfig):
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        if config.n_heads <= 0 or config.n_layers <= 0:
            raise ValueError("Number of heads and layers must be positive")
    
    # CNN-specific checks
    if isinstance(config, CNNConfig):
        if config.hidden_channels <= 0:
            raise ValueError("Hidden channels must be positive")
        
        if config.num_conv_layers <= 0:
            raise ValueError("Number of conv layers must be positive")
        
        if len(config.kernel_sizes) == 0:
            raise ValueError("At least one kernel size must be specified")
        
        if any(k <= 0 or k % 2 == 0 for k in config.kernel_sizes):
            raise ValueError("All kernel sizes must be positive and odd")
    
    return True


def convert_model_precision(model: BaseModel, precision: str = 'fp16') -> BaseModel:
    """
    Convert model to different precision.
    
    Args:
        model: Model to convert
        precision: Target precision ('fp16', 'fp32', 'bf16')
        
    Returns:
        Converted model
    """
    if precision == 'fp16':
        model = model.half()
    elif precision == 'bf16':
        if hasattr(torch, 'bfloat16'):
            model = model.to(torch.bfloat16)
        else:
            warnings.warn("bfloat16 not supported, using fp16 instead")
            model = model.half()
    elif precision == 'fp32':
        model = model.float()
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    logger.info(f"Model converted to {precision} precision")
    return model