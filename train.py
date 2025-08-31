"""
Simplified training script for fluid dynamics models.

Usage:
    python train.py                                    # Use default config
    python train.py --config configs/quick_test.json   # Use specific config
    python train.py --config configs/full_training.json
    python train.py --config configs/production.json
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path

from training.utils import setup_training, run_training, evaluate_model
from training.config import TrainingConfig
from data.compute_normalization_stats import compute_and_save_stats

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Fluid Dynamics Models (Simplified)")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.json",
        help="Path to JSON config file (default: configs/default.json)"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--eval_only", 
        action="store_true", 
        help="Only run evaluation (requires --model_path)"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="Path to trained model for evaluation"
    )
    
    return parser.parse_args()




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
    
    # Load configuration from file
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        logger.info("Available configs:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.json"):
                logger.info(f"  - {config_file}")
        sys.exit(1)
    
    logger.info(f"Loading config from: {config_path}")
    config = TrainingConfig.from_file(config_path)
    
    # Log configuration summary
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    # Load model configuration
    model_config = config.load_model_config()
    
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"Model config: {config.model_config_path}")
    # Log model-specific configuration
    if hasattr(model_config, 'd_model'):
        # Decoder/Transformer model
        logger.info(f"Model size: d_model={model_config.d_model}, n_heads={model_config.n_heads}, n_layers={model_config.n_layers}")
    elif hasattr(model_config, 'hidden_channels'):
        # CNN model
        logger.info(f"Model size: hidden_channels={model_config.hidden_channels}, num_conv_layers={model_config.num_conv_layers}, kernel_sizes={model_config.kernel_sizes}")
    else:
        logger.info(f"Model configuration loaded from: {config.model_config_path}")
    logger.info(f"Data dir: {config.data_dir}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    logger.info(f"Batch size: {config.train_batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    logger.info(f"Use SwanLab: {config.use_swanlab}")  # Keep config name for backward compatibility
    logger.info(f"Debug mode: {config.debug_mode}")
    logger.info("=" * 60)
    
    # Compute normalization statistics if requested
    if config.compute_normalization_stats:
        logger.info("Computing normalization statistics...")
        compute_and_save_stats(config.data_dir, config.normalization_method)
        logger.info("Normalization statistics computed and saved")
    
    # Evaluation only mode
    if args.eval_only:
        if not args.model_path:
            raise ValueError("--model_path is required for evaluation only mode")
        
        logger.info("Running evaluation only...")
        # TODO: Implement evaluation-only mode
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
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Model saved to: {config.output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Model: {model_config.model_name}")
    print(f"Total epochs: {config.num_train_epochs}")
    print(f"Total steps: {training_results.get('total_steps', 'N/A')}")
    print(f"Best metric: {training_results.get('best_metric', 'N/A')}")
    print(f"Final eval loss: {training_results.get('eval_result', {}).get('eval_loss', 'N/A')}")
    if 'test_results' in training_results:
        print(f"Test loss: {training_results['test_results'].get('test_loss', 'N/A')}")
    print(f"Output directory: {config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()