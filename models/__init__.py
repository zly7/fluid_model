"""
Models package for fluid dynamics neural network models.

This package provides:
- BaseModel: Abstract base class for all models
- FluidDecoder: Decoder-only model for autoregressive prediction
- FluidCNN: CNN model for time series prediction
- FluidLSTM: LSTM model for time series prediction
- ModelConfig: Configuration classes for models
- Utility functions for model management
"""

from .base import BaseModel, MaskedMSELoss
from .decoder import FluidDecoder
from .cnn import FluidCNN
from .lstm import FluidLSTM
from .config import (
    ModelConfig,
    DecoderConfig,
    CNNConfig,
    create_default_configs,
    load_config_from_file,
    save_config_to_file
)
from .lstm import LSTMConfig
from .utils import (
    count_parameters,
    initialize_weights,
    create_model,
    load_model,
    save_model
)

__all__ = [
    'BaseModel',
    'MaskedMSELoss',
    'FluidDecoder',
    'FluidCNN',
    'FluidLSTM',
    'ModelConfig',
    'DecoderConfig',
    'CNNConfig',
    'LSTMConfig',
    'create_default_configs',
    'load_config_from_file',
    'save_config_to_file',
    'count_parameters',
    'initialize_weights',
    'create_model',
    'load_model',
    'save_model'
]