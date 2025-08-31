"""
Configuration classes for Decoder model.
"""

from dataclasses import dataclass
from ..config import ModelConfig


@dataclass
class DecoderConfig(ModelConfig):
    """Configuration for Decoder model."""
    
    model_name: str = "FluidDecoder"
    
    # Architecture parameters
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 6
    d_ff: int = 3072
    
    # Attention parameters
    attention_dropout: float = 0.1
    
    # Positional encoding
    time_position_encoding: str = "sinusoidal"  # "sinusoidal" or "learned"
    variable_position_encoding: str = "sinusoidal"  # "sinusoidal" or "learned"
    max_time_positions: int = 10
    max_variable_positions: int = 6712
    
    # Input/output projection (简化版本)
    projection_hidden_dim: int = 256
    
    # Optimization
    use_layer_norm: bool = True
    activation: str = "gelu"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")