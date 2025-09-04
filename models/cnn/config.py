"""
Configuration classes for CNN model.
"""

from dataclasses import dataclass
from ..config import ModelConfig


@dataclass
class CNNConfig(ModelConfig):
    """Configuration for CNN model."""
    
    model_name: str = "FluidCNN"
    
    # CNN Architecture parameters
    hidden_channels: int = 512
    kernel_sizes: tuple = (3, 5, 7)  # Multiple kernel sizes for multi-scale features
    num_conv_layers: int = 4
    dilation_rates: tuple = (1, 2, 4, 8)  # For dilated convolutions
    
    # Feature extraction
    boundary_hidden_dim: int = 256
    equipment_hidden_dim: int = 512
    
    # Temporal modeling
    temporal_kernel_size: int = 3
    temporal_stride: int = 1
    temporal_padding: str = "same"  # "same" or "valid"
    
    # Regularization
    batch_norm: bool = True
    use_residual: bool = True
    activation: str = "relu"  # "relu", "gelu", "leaky_relu"
    
    # Output projection
    projection_hidden_dim: int = 256
    
    def __post_init__(self):
        """Validate configuration."""
        if len(self.kernel_sizes) != len(self.dilation_rates):
            # If lengths don't match, adjust dilation rates
            self.dilation_rates = self.dilation_rates[:len(self.kernel_sizes)]
        
        if self.temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size should be odd for proper padding")