"""
Main training script for fluid dynamics models.

This script provides a command-line interface for training fluid dynamics models
with various configuration options.

Usage:
    python -m training.train --data_dir ./data --output_dir ./outputs --epochs 10
    python -m training.train --config config.json
    python -m training.train --quick_test  # For quick testing
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import json
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.utils import setup_training, run_training, evaluate_model
from training.config import TrainingConfig, create_default_training_config, create_quick_test_config, create_full_training_config
from data.compute_normalization_stats import compute_and_save_normalization_stats

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Fluid Dynamics Models")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Path to output directory")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, help="Evaluation batch size (default: 2x train batch)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="FluidDecoder", help="Model name")
    parser.add_argument("--sequence_length", type=int, default=3, help="Sequence length")
    
    # Evaluation and saving
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Save frequency")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints")
    
    # Data processing
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to load")
    parser.add_argument("--max_sequences_per_sample", type=int, help="Maximum sequences per sample")
    parser.add_argument("--normalization_method", type=str, default="standard", choices=["standard", "minmax", "none"], help="Normalization method")
    parser.add_argument("--compute_normalization_stats", action="store_true", help="Compute normalization statistics before training")
    
    # Training modes
    parser.add_argument("--quick_test", action="store_true", help="Run quick test with limited data")
    parser.add_argument("--full_training", action="store_true", help="Run full-scale training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Hardware and performance
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Logging and monitoring
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="fluid-dynamics", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity name")
    parser.add_argument("--run_name", type=str, help="Run name")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    
    # Evaluation
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation (requires trained model)")
    parser.add_argument("--model_path", type=str, help="Path to trained model for evaluation")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def load_config_from_args(args) -> TrainingConfig:
    """Load configuration from command line arguments."""
    
    # Load from config file if provided
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = TrainingConfig.from_file(args.config)
    else:
        # Create config based on training mode
        if args.quick_test:
            logger.info("Creating quick test configuration")
            config = create_quick_test_config()
        elif args.full_training:
            logger.info("Creating full training configuration")
            config = create_full_training_config()
        else:
            logger.info("Creating default configuration")
            config = create_default_training_config()
    
    # Override config with command line arguments
    arg_overrides = {}
    
    # Data parameters
    if args.data_dir != "./data":
        arg_overrides['data_dir'] = args.data_dir
    if args.output_dir != "./outputs":
        arg_overrides['output_dir'] = args.output_dir
    
    # Training parameters
    if args.epochs != 10:
        arg_overrides['num_train_epochs'] = args.epochs
    if args.batch_size != 32:
        arg_overrides['train_batch_size'] = args.batch_size
    if args.eval_batch_size:
        arg_overrides['eval_batch_size'] = args.eval_batch_size
    if args.learning_rate != 1e-4:
        arg_overrides['learning_rate'] = args.learning_rate
    if args.weight_decay != 1e-5:
        arg_overrides['weight_decay'] = args.weight_decay
    if args.warmup_ratio != 0.1:
        arg_overrides['warmup_ratio'] = args.warmup_ratio
    
    # Model parameters
    if args.model_name != "FluidDecoder":
        arg_overrides['model_name'] = args.model_name
    if args.sequence_length != 3:
        arg_overrides['sequence_length'] = args.sequence_length
    
    # Evaluation and saving
    if args.eval_steps != 500:
        arg_overrides['eval_steps'] = args.eval_steps
    if args.save_steps != 500:
        arg_overrides['save_steps'] = args.save_steps
    if args.save_total_limit != 3:
        arg_overrides['save_total_limit'] = args.save_total_limit
    
    # Data processing
    if args.max_samples:
        arg_overrides['max_samples'] = args.max_samples
    if args.max_sequences_per_sample:
        arg_overrides['max_sequences_per_sample'] = args.max_sequences_per_sample
    if args.normalization_method != "standard":
        arg_overrides['normalization_method'] = args.normalization_method
    if args.compute_normalization_stats:
        arg_overrides['compute_normalization_stats'] = True
    
    # Debug mode
    if args.debug:
        arg_overrides['debug_mode'] = True
    
    # Hardware and performance
    if args.device != "auto":
        arg_overrides['device'] = args.device
    if args.num_workers != 0:
        arg_overrides['num_workers'] = args.num_workers
    if args.mixed_precision:
        arg_overrides['mixed_precision'] = True
    if args.gradient_accumulation_steps != 1:
        arg_overrides['accumulation_steps'] = args.gradient_accumulation_steps
    
    # Logging and monitoring
    if args.use_wandb:
        arg_overrides['use_wandb'] = True
    if args.wandb_project != "fluid-dynamics":
        arg_overrides['wandb_project'] = args.wandb_project
    if args.wandb_entity:
        arg_overrides['wandb_entity'] = args.wandb_entity
    if args.run_name:
        arg_overrides['run_name'] = args.run_name
    
    # Reproducibility
    if args.seed != 42:
        arg_overrides['seed'] = args.seed
    
    # Apply overrides
    for key, value in arg_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
            logger.info(f"Override config: {key} = {value}")
        else:
            logger.warning(f"Unknown config parameter: {key}")
    
    return config


def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Fluid Dynamics Model Training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load configuration
        config = load_config_from_args(args)
        
        # Compute normalization statistics if requested
        if config.compute_normalization_stats:
            logger.info("Computing normalization statistics...")
            compute_and_save_normalization_stats(config.data_dir, config.normalization_method)
            logger.info("Normalization statistics computed and saved")
        
        # Evaluation only mode
        if args.eval_only:
            if not args.model_path:
                raise ValueError("--model_path is required for evaluation only mode")
            
            logger.info("Running evaluation only...")
            # TODO: Implement evaluation-only mode
            # This would load a trained model and run evaluation
            logger.error("Evaluation-only mode not yet implemented")
            return
        
        # Setup training
        logger.info("Setting up training environment...")
        setup_result = setup_training(config)
        
        trainer = setup_result['trainer']
        train_dataset = setup_result['train_dataset']
        val_dataset = setup_result['val_dataset']
        test_dataset = setup_result['test_dataset']
        
        # Log dataset information
        logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")
        if test_dataset:
            logger.info(f"Test samples: {len(test_dataset)}")
        
        # Run training
        logger.info("Starting training process...")
        training_results = run_training(trainer, resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Final evaluation on test set if available
        if test_dataset and len(test_dataset) > 0:
            logger.info("Running final test evaluation...")
            test_results = evaluate_model(trainer, test_dataset)
            training_results['test_results'] = test_results
        
        # Save final results
        results_path = os.path.join(config.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            # Convert tensors and other non-serializable objects to strings
            serializable_results = {}
            for key, value in training_results.items():
                if hasattr(value, 'item'):  # torch tensor with single value
                    serializable_results[key] = value.item()
                elif hasattr(value, 'tolist'):  # numpy array or torch tensor
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = str(value)
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Model saved to: {config.output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Model: {config.model_name}")
        print(f"Total epochs: {config.num_train_epochs}")
        print(f"Total steps: {training_results.get('total_steps', 'N/A')}")
        print(f"Best metric: {training_results.get('best_metric', 'N/A')}")
        print(f"Final eval loss: {training_results.get('eval_result', {}).get('eval_loss', 'N/A')}")
        if 'test_results' in training_results:
            print(f"Test loss: {training_results['test_results'].get('test_loss', 'N/A')}")
        print(f"Output directory: {config.output_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()