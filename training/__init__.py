"""
Training package for fluid dynamics models using transformers.Trainer.

This package provides:
- HuggingFace integration components
- Training scripts and configurations  
- Metrics computation and evaluation
- Inference interfaces
- Custom callbacks for training monitoring
"""

from .hf_integration import (
    FluidDecoderForTraining,
    FluidDecoderConfig, 
    FluidDataCollator,
    compute_fluid_metrics
)
from .hf_integration.callbacks import create_training_callbacks
from .inference import FluidInference, load_inference_model

__all__ = [
    'FluidDecoderForTraining',
    'FluidDecoderConfig',
    'FluidDataCollator', 
    'compute_fluid_metrics',
    'create_training_callbacks',
    'FluidInference',
    'load_inference_model'
]