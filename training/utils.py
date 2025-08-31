"""
Training utilities for fluid dynamics models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
import logging
import os
import random
import numpy as np
from pathlib import Path

from transformers import set_seed
import swanlab
from models import FluidDecoder
from models.config import DecoderConfig
from data.dataset import FluidDataset, create_dataloader_with_normalization
from data.normalizer import load_normalizer
from .trainer import create_fluid_trainer, FluidTrainer
from .config import TrainingConfig, create_default_training_config
from .callbacks import create_default_callbacks

logger = logging.getLogger(__name__)


def setup_training_environment(config: TrainingConfig) -> None:
    """
    Setup training environment with proper logging, seeds, and device configuration.
    
    Args:
        config: Training configuration
    """
    # Set up logging
    log_level = logging.DEBUG if config.debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(config.output_dir, 'training.log'))
        ]
    )
    
    logger.info("Setting up training environment...")
    
    # Set random seeds for reproducibility
    set_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        # For deterministic behavior (may reduce performance)
        if config.debug_mode:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seeds set to {config.seed}")
    
    # Device setup
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # Save training config
    config.save_to_file(os.path.join(config.output_dir, "training_config.json"))
    
    # Initialize SwanLab if configured
    if config.use_swanlab:  # Keep config name for backward compatibility
        swanlab.init(
            project=config.wandb_project,
            workspace=config.wandb_entity,
            experiment_name=config.wandb_run_name,
            config=config.to_dict(),
            logdir=config.output_dir
        )
        logger.info(f"Initialized SwanLab run: {config.wandb_run_name}")
    
    logger.info("Training environment setup complete")


def create_model(config: TrainingConfig) -> nn.Module:
    """
    Create and initialize the model.
    
    Args:
        config: Training configuration
        
    Returns:
        Initialized model
    """
    logger.info("Creating model...")
    
    # Load model config from separate file
    model_config = config.load_model_config()
    
    # Override with training config values where appropriate
    model_config.sequence_length = config.sequence_length
    
    # Create model
    if model_config.model_name.lower() == "fluiddecoder":
        model = FluidDecoder(config=model_config)
    else:
        raise ValueError(f"Unknown model name: {model_config.model_name}")
    
    # Log model info
    model_info = model.get_model_info()
    logger.info(f"Created {model_info['model_name']} with {model_info['trainable_parameters']:,} trainable parameters")
    
    return model


def create_datasets(config: TrainingConfig) -> Tuple[FluidDataset, Optional[FluidDataset], Optional[FluidDataset]]:
    """
    Create train, validation, and test datasets.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Creating datasets...")
    
    # Create training dataset
    train_dataset = FluidDataset(
        data_dir=config.data_dir,
        split='train',
        sequence_length=config.sequence_length,
        cache_data=True,
        max_samples=config.max_samples,
        max_sequences_per_sample=config.max_sequences_per_sample
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} sequences")
    
    # Create validation dataset
    val_dataset = FluidDataset(
        data_dir=config.data_dir,
        split='val',
        sequence_length=config.sequence_length,
        cache_data=True,
        max_samples=config.max_samples // 4 if config.max_samples else None,
        max_sequences_per_sample=config.max_sequences_per_sample
    )
    
    logger.info(f"Validation dataset: {len(val_dataset)} sequences")
    
    # Create test dataset if available
    try:
        test_dataset = FluidDataset(
            data_dir=config.data_dir,
            split='test',
            sequence_length=config.sequence_length,
            cache_data=True,
            max_samples=config.max_samples // 10 if config.max_samples else None,
            max_sequences_per_sample=config.max_sequences_per_sample
        )
        logger.info(f"Test dataset: {len(test_dataset)} sequences")
    except Exception as e:
        logger.warning(f"Could not create test dataset: {e}")
        test_dataset = None
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: FluidDataset,
    val_dataset: Optional[FluidDataset],
    config: TrainingConfig
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create data loaders with normalization.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logger.info("Creating data loaders...")
    
    # Create training dataloader
    train_dataloader = create_dataloader_with_normalization(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        normalizer_method=config.normalization_method,
        apply_normalization=config.apply_normalization
    )
    
    logger.info(f"Training dataloader: {len(train_dataloader)} batches")
    
    # Create validation dataloader
    val_dataloader = None
    if val_dataset:
        val_dataloader = create_dataloader_with_normalization(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            normalizer_method=config.normalization_method,
            apply_normalization=config.apply_normalization
        )
        logger.info(f"Validation dataloader: {len(val_dataloader)} batches")
    
    return train_dataloader, val_dataloader


def setup_training(config: Optional[TrainingConfig] = None, **config_kwargs) -> Dict[str, Any]:
    """
    Complete training setup including environment, model, datasets, and trainer.
    
    Args:
        config: Training configuration (if None, creates default)
        **config_kwargs: Override config parameters
        
    Returns:
        Dictionary containing all training components
    """
    # Create config if not provided
    if config is None:
        config = create_default_training_config(**config_kwargs)
    else:
        # Update config with any overrides
        for key, value in config_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Setup environment
    setup_training_environment(config)
    
    # Create model
    model = create_model(config)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, config)
    
    # Load normalizer if needed
    normalizer = None
    if config.apply_normalization:
        normalizer = load_normalizer(config.data_dir, config.normalization_method)
        if normalizer is None:
            logger.warning("Normalizer not found. Training will proceed without normalization.")
    
    # Create trainer
    trainer = create_fluid_trainer(
        model=model,
        training_config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        normalizer=normalizer,
        callbacks=create_default_callbacks(
            log_memory_usage=True,
            save_predictions_every_n_steps=config.save_steps * 2 if config.save_steps > 0 else None
        )
    )
    
    return {
        'config': config,
        'model': model,
        'trainer': trainer,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'normalizer': normalizer
    }


def run_training(trainer: FluidTrainer, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the training process.
    
    Args:
        trainer: Configured FluidTrainer
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Training results dictionary
    """
    logger.info("Starting training...")
    
    try:
        # Run training
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        trainer.save_model()
        
        # Final evaluation
        eval_result = {}
        if trainer.get_eval_dataloader() is not None:
            logger.info("Running final evaluation...")
            eval_result = trainer.evaluate()
        
        # Training summary
        training_summary = {
            'train_result': train_result,
            'eval_result': eval_result,
            'model_path': trainer.args.output_dir,
            'total_steps': trainer.state.global_step,
            'best_metric': trainer.state.best_metric,
        }
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {trainer.args.output_dir}")
        
        if eval_result:
            logger.info(f"Final evaluation loss: {eval_result.get('eval_loss', 'N/A')}")
        
        return training_summary
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        if trainer.training_config.use_swanlab:
            swanlab.finish()


def evaluate_model(
    trainer: FluidTrainer,
    test_dataset: Optional[FluidDataset] = None,
    save_predictions: bool = True
) -> Dict[str, Any]:
    """
    Evaluate trained model on test dataset.
    
    Args:
        trainer: Trained FluidTrainer
        test_dataset: Test dataset (optional)
        save_predictions: Whether to save predictions
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating model...")
    
    # Use provided test dataset or trainer's eval dataset
    if test_dataset is not None:
        # Create test dataloader
        test_dataloader = create_dataloader_with_normalization(
            test_dataset,
            batch_size=trainer.training_config.eval_batch_size,
            shuffle=False,
            num_workers=trainer.training_config.num_workers,
            normalizer_method=trainer.training_config.normalization_method,
            apply_normalization=trainer.training_config.apply_normalization
        )
        
        # Temporarily replace eval dataloader
        original_eval_dataloader = trainer._eval_dataloader
        trainer._eval_dataloader = test_dataloader
        
        try:
            eval_result = trainer.evaluate(metric_key_prefix="test")
        finally:
            # Restore original eval dataloader
            trainer._eval_dataloader = original_eval_dataloader
    else:
        eval_result = trainer.evaluate(metric_key_prefix="test")
    
    logger.info("Model evaluation completed")
    for key, value in eval_result.items():
        if key.startswith("test_"):
            logger.info(f"{key}: {value:.6f}")
    
    return eval_result


def create_trainer(
    model_config_path: str = "configs/models/medium.json",
    data_dir: str = "./data",
    output_dir: str = "./outputs",
    **kwargs
) -> FluidTrainer:
    """
    Convenience function to create a trainer with minimal configuration.
    
    Args:
        model_config_path: Path to model configuration file
        data_dir: Path to data directory
        output_dir: Path to output directory
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured FluidTrainer
    """
    config = create_default_training_config(
        model_config_path=model_config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        **kwargs
    )
    
    setup_result = setup_training(config)
    return setup_result['trainer']


def quick_train(
    data_dir: str = "./data",
    output_dir: str = "./outputs",
    epochs: int = 5,
    batch_size: int = 16,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick training function for testing and development.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory  
        epochs: Number of training epochs
        batch_size: Training batch size
        **kwargs: Additional configuration parameters
        
    Returns:
        Training results
    """
    config = create_default_training_config(
        data_dir=data_dir,
        output_dir=output_dir,
        num_train_epochs=epochs,
        train_batch_size=batch_size,
        eval_batch_size=batch_size * 2,
        max_samples=20,  # Limited for quick testing
        eval_steps=50,
        save_steps=100,
        debug_mode=True,
        **kwargs
    )
    
    setup_result = setup_training(config)
    trainer = setup_result['trainer']
    
    # Run training
    results = run_training(trainer)
    
    return results