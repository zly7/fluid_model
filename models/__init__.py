"""
Models package for fluid dynamics neural network models.

This package provides:
- BaseModel: Abstract base class for all models
- FluidDecoder: Decoder-only model for autoregressive prediction
- ModelConfig: Configuration classes for models
- Utility functions for model management
"""

from .base import BaseModel, MaskedMSELoss
from .decoder import FluidDecoder
from .config import (
    ModelConfig,
    DecoderConfig,
    create_default_configs,
    load_config_from_file,
    save_config_to_file
)
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
    'ModelConfig',
    'DecoderConfig',
    'create_default_configs',
    'load_config_from_file',
    'save_config_to_file',
    'count_parameters',
    'initialize_weights',
    'create_model',
    'load_model',
    'save_model'
]