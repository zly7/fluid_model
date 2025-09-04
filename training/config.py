"""
Training configuration for fluid dynamics models.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import os
from pathlib import Path


@dataclass
class TrainingConfig:
    """Training configuration for fluid dynamics models."""
    
    # Data parameters
    data_dir: str = "./data"
    train_batch_size: int = 32
    eval_batch_size: int = 64
    sequence_length: int = 3
    max_samples: Optional[int] = None
    max_sequences_per_sample: Optional[int] = None
    
    # Training parameters
    num_train_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    
    # Optimization
    optimizer_type: str = "adamw"  # "adamw", "adam", "sgd"
    scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    
    # Model parameters
    model_config_path: str = "configs/models/decoder/medium.json"  # Path to model config file
    
    # Output and logging
    output_dir: str = "./outputs"
    logging_dir: Optional[str] = None
    run_name: Optional[str] = None
    
    # Evaluation and checkpointing
    eval_strategy: str = "steps"  # "steps", "epoch", "no"
    eval_steps: int = 500
    save_strategy: str = "steps"  # "steps", "epoch", "no"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    # Data processing
    normalization_method: str = "standard"  # "standard", "minmax", "none"
    apply_normalization: bool = True
    compute_normalization_stats: bool = False
    
    # Hardware and performance
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    mixed_precision: bool = True  # Use automatic mixed precision
    
    # Reproducibility
    seed: int = 42
    
    # Debugging and development
    debug_mode: bool = False
    max_steps: Optional[int] = None
    dataloader_drop_last: bool = False
    
    # Weights & Biases integration
    use_swanlab: bool = False
    swanlab_project: Optional[str] = "fluid-dynamics"
    swanlab_entity: Optional[str] = None
    swanlab_run_name: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set logging directory if not specified
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.logging_dir, exist_ok=True)
        
        # Validate parameters
        if self.eval_strategy not in ["steps", "epoch", "no"]:
            raise ValueError(f"Invalid eval_strategy: {self.eval_strategy}")
        
        if self.save_strategy not in ["steps", "epoch", "no"]:
            raise ValueError(f"Invalid save_strategy: {self.save_strategy}")
        
        if self.optimizer_type not in ["adamw", "adam", "sgd"]:
            raise ValueError(f"Invalid optimizer_type: {self.optimizer_type}")
        
        if self.scheduler_type not in ["cosine", "linear", "constant"]:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")
        
        # Set run name if not specified
        if self.run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_size = Path(self.model_config_path).stem  # Get model size from filename
            self.run_name = f"FluidDecoder_{model_size}_{timestamp}"
        
        # W&B setup
        if self.swanlab_run_name is None:
            self.swanlab_run_name = self.run_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        config_dict = self.to_dict()
        # Handle non-serializable fields
        for key, value in config_dict.items():
            if value is None:
                continue
            if isinstance(value, Path):
                config_dict[key] = str(value)
        return json.dumps(config_dict, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create config from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    def load_model_config(self):
        """Load model configuration from the specified path."""
        from models.config import load_config_from_file
        return load_config_from_file(self.model_config_path)
    
    def get_transformers_training_args(self) -> Dict[str, Any]:
        """
        Convert to transformers TrainingArguments format.
        
        Returns:
            Dict compatible with transformers.TrainingArguments
        """
        args = {
            'output_dir': self.output_dir,
            'num_train_epochs': self.num_train_epochs,
            'per_device_train_batch_size': self.train_batch_size,
            'per_device_eval_batch_size': self.eval_batch_size,
            'gradient_accumulation_steps': self.accumulation_steps,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'max_grad_norm': self.gradient_clip_norm,
            'logging_dir': self.logging_dir,
            'logging_steps': min(self.eval_steps // 4, 100),
            'eval_strategy': self.eval_strategy,
            'eval_steps': self.eval_steps,
            'save_strategy': self.save_strategy,
            'save_steps': self.save_steps,
            'save_total_limit': self.save_total_limit,
            'load_best_model_at_end': self.load_best_model_at_end,
            'metric_for_best_model': self.metric_for_best_model,
            'greater_is_better': self.greater_is_better,
            'run_name': self.run_name,
            'seed': self.seed,
            'fp16': self.mixed_precision,
            'dataloader_drop_last': self.dataloader_drop_last,
            'dataloader_num_workers': self.num_workers,
            'dataloader_pin_memory': self.pin_memory,
            'report_to': ["swanlab"] if self.use_swanlab else [],  # SwanLab integration
            'max_steps': self.max_steps if self.max_steps else -1,
            'disable_tqdm': False,
            'remove_unused_columns': False,  # Important for our custom data format
        }
        
        # Device configuration is now handled earlier in setup_training_environment()
        # to ensure CUDA_VISIBLE_DEVICES is set before PyTorch initialization
        
        # Only include warmup_steps if it's explicitly set (not None)
        if self.warmup_steps is not None:
            args['warmup_steps'] = self.warmup_steps
            
        return args


def create_default_training_config(**kwargs) -> TrainingConfig:
    """Create a default training configuration with optional overrides."""
    return TrainingConfig(**kwargs)


def create_quick_test_config(**kwargs) -> TrainingConfig:
    """Create a configuration for quick testing/debugging."""
    defaults = {
        'num_train_epochs': 2,
        'train_batch_size': 4,
        'eval_batch_size': 8,
        'eval_steps': 10,
        'save_steps': 10,
        'max_samples': 5,
        'max_sequences_per_sample': 10,
        'debug_mode': True,
        'mixed_precision': False,
    }
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def create_full_training_config(**kwargs) -> TrainingConfig:
    """Create a configuration for full-scale training."""
    defaults = {
        'num_train_epochs': 50,
        'train_batch_size': 32,
        'eval_batch_size': 64,
        'learning_rate': 1e-4,
        'warmup_ratio': 0.1,
        'eval_steps': 500,
        'save_steps': 1000,
        'early_stopping_patience': 10,
        'mixed_precision': True,
    }
    defaults.update(kwargs)
    return TrainingConfig(**defaults)