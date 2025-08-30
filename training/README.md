# Training Package for Fluid Dynamics Models

✅ **COMPLETED** - Complete training infrastructure for fluid dynamics neural networks, built on top of Transformers library.

## Features

- **Complete Training Pipeline**: Setup, training, validation, and evaluation
- **Transformers Integration**: Compatible with HuggingFace Transformers Trainer
- **Custom Metrics**: Fluid-dynamics specific evaluation metrics
- **Data Normalization**: Automatic normalization and denormalization
- **Flexible Configuration**: JSON-based configuration with command-line overrides
- **Monitoring**: Weights & Biases integration, custom callbacks
- **Checkpointing**: Automatic model saving and loading
- **Multi-GPU Support**: Through Transformers infrastructure

## Quick Start

### 1. Basic Training
```bash
# Quick test with limited data
python -m training.train --quick_test --data_dir ./data --output_dir ./outputs

# Full training
python -m training.train --full_training --data_dir ./data --output_dir ./outputs --epochs 50

# Custom training
python -m training.train --data_dir ./data --output_dir ./outputs --epochs 10 --batch_size 32 --learning_rate 1e-4
```

### 2. Using Configuration Files
```bash
# Save a configuration
python -c "
from training.config import create_full_training_config
config = create_full_training_config(data_dir='./data', output_dir='./outputs')
config.save_to_file('my_config.json')
"

# Train with config file
python -m training.train --config my_config.json
```

### 3. Programmatic Usage
```python
from training.utils import setup_training, run_training

# Setup training
setup_result = setup_training(
    data_dir="./data",
    output_dir="./outputs", 
    num_train_epochs=10,
    train_batch_size=32
)

# Get trainer and run training
trainer = setup_result['trainer']
results = run_training(trainer)
```

## Configuration

### Training Configuration (`TrainingConfig`)

Key parameters:

- **Data**: `data_dir`, `train_batch_size`, `eval_batch_size`, `sequence_length`
- **Training**: `num_train_epochs`, `learning_rate`, `weight_decay`, `warmup_ratio`
- **Model**: `model_name`, `model_config`
- **Evaluation**: `eval_strategy`, `eval_steps`, `save_steps`
- **Hardware**: `device`, `mixed_precision`, `num_workers`
- **Monitoring**: `use_wandb`, `wandb_project`

### Example Configuration
```json
{
  "data_dir": "./data",
  "output_dir": "./outputs",
  "num_train_epochs": 50,
  "train_batch_size": 32,
  "eval_batch_size": 64,
  "learning_rate": 1e-4,
  "weight_decay": 1e-5,
  "warmup_ratio": 0.1,
  "eval_steps": 500,
  "save_steps": 1000,
  "model_name": "FluidDecoder",
  "use_wandb": true,
  "wandb_project": "fluid-dynamics"
}
```

## Architecture

### Core Components

1. **`TrainingConfig`**: Configuration management
2. **`FluidTrainer`**: Custom trainer extending HF Trainer
3. **`FluidTrainingCallbacks`**: Custom training callbacks
4. **Training Utils**: Setup and utility functions

### File Structure
```
training/
├── __init__.py                 # Package initialization
├── config.py                  # Training configuration classes
├── trainer.py                 # Custom FluidTrainer class
├── callbacks.py               # Training callbacks and monitoring
├── utils.py                   # Setup and utility functions
├── train.py                   # Main training script
└── CLAUDE.md                  # This documentation
```

### Data Flow

```
Raw Data → Dataset → DataLoader → Normalizer → Model → Loss → Metrics
                                    ↓
                              Trainer → Callbacks → Logging/Saving
```

## Advanced Usage

### Custom Model Configuration
```python
from models.config import DecoderConfig
from training.config import TrainingConfig

# Custom model config
model_config = DecoderConfig(
    d_model=512,
    n_heads=8,
    n_layers=6,
    learning_rate=5e-5
)

# Training config with custom model
training_config = TrainingConfig(
    model_config=model_config.to_dict(),
    num_train_epochs=100
)
```

### Weights & Biases Integration
```bash
python -m training.train \
    --use_wandb \
    --wandb_project "my-fluid-project" \
    --wandb_entity "my-team" \
    --run_name "experiment-1"
```

### Resume Training
```bash
python -m training.train \
    --resume_from_checkpoint ./outputs/checkpoint-1000 \
    --output_dir ./outputs
```

## Model Compatibility

The training system is designed to work with models that:
1. Inherit from `BaseModel`
2. Implement transformers-compatible `forward()` method
3. Return `{'loss': tensor, 'logits': tensor}` format

Currently supported models:
- **FluidDecoder**: Pure decoder architecture for time series prediction

## Metrics and Evaluation

The system computes various metrics:
- **MSE Loss**: Mean squared error
- **Equipment MSE**: MSE for equipment variables only
- **MAE**: Mean absolute error  
- **Boundary Metrics**: Metrics specific to boundary conditions
- **Custom Metrics**: Through `compute_fluid_metrics()`

## Implementation Status

✅ **Completed Components:**

1. **TrainingConfig** - Complete configuration management with JSON support
2. **FluidTrainer** - Custom trainer with fluid-specific metrics and normalization
3. **FluidTrainingCallbacks** - Memory monitoring, progress tracking, sample predictions
4. **Training Utils** - Environment setup, data loading, trainer creation
5. **Main Training Script** - Command-line interface with full argument parsing
6. **Integration** - Full compatibility with existing data and model packages

## Best Practices

1. **Start Small**: Use `--quick_test` for development
2. **Normalization**: Always compute normalization stats first
3. **Monitoring**: Use W&B for experiment tracking
4. **Checkpointing**: Configure appropriate save frequencies
5. **Evaluation**: Monitor both training and validation metrics
6. **Resources**: Tune batch size based on available GPU memory

## Usage Examples

```bash
# Compute normalization stats first
python -m training.train --compute_normalization_stats --data_dir ./data

# Quick development test
python -m training.train --quick_test --debug --max_samples 5

# Full production training
python -m training.train --full_training --use_wandb --mixed_precision --epochs 100
```
