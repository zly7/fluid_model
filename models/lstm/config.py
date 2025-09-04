"""
Configuration class for LSTM model.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


@dataclass
class LSTMConfig:
    """
    Configuration for LSTM-based fluid dynamics model.
    
    Model Architecture:
    - Input dimension: 6712 (boundary + equipment variables)  
    - LSTM layers for temporal modeling
    - Separate processing for boundary (538) and equipment (6174) variables
    - Output projection back to original dimension
    """
    
    # Model metadata
    model_name: str = "FluidLSTM"
    
    # Basic model dimensions
    input_dim: int = 6712
    output_dim: int = 6712
    boundary_dims: int = 538
    equipment_dims: int = 6174
    
    # LSTM architecture
    hidden_dim: int = 256
    num_layers: int = 2
    bidirectional: bool = False
    dropout_rate: float = 0.1
    
    # Separate processing for boundary and equipment
    boundary_hidden_dim: int = 128
    equipment_hidden_dim: int = 256
    
    # Feature processing
    projection_hidden_dim: int = 512
    
    # Training parameters
    batch_norm: bool = True
    activation: str = "tanh"  # tanh, relu, gelu
    
    # Advanced LSTM options
    use_attention: bool = False
    attention_heads: int = 8
    use_layer_norm: bool = True
    
    # Regularization
    recurrent_dropout: float = 0.0
    weight_dropout: float = 0.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.output_dim > 0, "output_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0.0 <= self.dropout_rate < 1.0, "dropout_rate must be in [0, 1)"
        assert self.boundary_dims + self.equipment_dims == self.input_dim, \
            f"boundary_dims ({self.boundary_dims}) + equipment_dims ({self.equipment_dims}) must equal input_dim ({self.input_dim})"
        assert self.activation in ["tanh", "relu", "gelu"], \
            f"activation must be one of ['tanh', 'relu', 'gelu'], got {self.activation}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LSTMConfig':
        """Create config from dictionary."""
        return cls(**config_dict)