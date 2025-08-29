"""
Configuration classes for fluid dynamics models.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json


@dataclass
class ModelConfig:
    """Base configuration for all models."""
    
    # Data dimensions
    input_dim: int = 6712
    output_dim: int = 6712
    sequence_length: int = 3
    boundary_dims: int = 538
    equipment_dims: int = 6174
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1
    
    # Model metadata
    model_name: str = "BaseModel"
    model_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create config from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)




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


    
    
def create_default_configs() -> Dict[str, ModelConfig]:
    """Create default configurations for all model types."""
    return {
        'decoder': DecoderConfig()
    }


def load_config_from_file(filepath: str) -> ModelConfig:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    model_type = config_dict.get('model_name', '').lower()
    
    if 'decoder' in model_type:
        return DecoderConfig.from_dict(config_dict)
    else:
        return ModelConfig.from_dict(config_dict)


def save_config_to_file(config: ModelConfig, filepath: str):
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        f.write(config.to_json())