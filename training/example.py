"""
Example training script demonstrating how to use the training package.

This script shows different ways to train fluid dynamics models:
1. Quick test training
2. Full training with custom configuration
3. Programmatic training setup
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.config import TrainingConfig, create_quick_test_config, create_full_training_config
from training.utils import setup_training, run_training
from models.config import DecoderConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_quick_test():
    """Example: Quick test training for development and debugging."""
    
    logger.info("=== Example 1: Quick Test Training ===")
    
    # Create quick test configuration
    config = create_quick_test_config(
        data_dir="./data",
        output_dir="./outputs/quick_test",
        num_train_epochs=2,
        max_samples=3,  # Very limited data for quick testing
        max_sequences_per_sample=5
    )
    
    # Setup training
    setup_result = setup_training(config)
    trainer = setup_result['trainer']
    
    # Run training
    results = run_training(trainer)
    
    logger.info(f"Quick test completed! Results: {results}")
    return results


def example_custom_configuration():
    """Example: Custom training configuration with specific model settings."""
    
    logger.info("=== Example 2: Custom Configuration Training ===")
    
    # Create custom model configuration
    model_config = DecoderConfig(
        d_model=256,        # Smaller model for faster training
        n_heads=8,
        n_layers=4,
        learning_rate=1e-4
    )
    
    # Create training configuration with custom model
    config = TrainingConfig(
        data_dir="./data",
        output_dir="./outputs/custom_config",
        num_train_epochs=5,
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        eval_steps=100,
        save_steps=200,
        model_config=model_config.to_dict(),
        max_samples=10,  # Limited for example
        debug_mode=True
    )
    
    # Save configuration for reference
    config.save_to_file("./outputs/example_config.json")
    logger.info("Configuration saved to ./outputs/example_config.json")
    
    # Setup and run training
    setup_result = setup_training(config)
    trainer = setup_result['trainer']
    results = run_training(trainer)
    
    logger.info(f"Custom configuration training completed! Results: {results}")
    return results


def example_full_training():
    """Example: Full-scale training configuration."""
    
    logger.info("=== Example 3: Full Training Configuration ===")
    
    # Create full training configuration
    config = create_full_training_config(
        data_dir="./data",
        output_dir="./outputs/full_training",
        num_train_epochs=20,  # Reduced for example
        use_wandb=False,  # Disable W&B for example
        mixed_precision=True
    )
    
    # Optional: Modify specific settings
    config.early_stopping_patience = 5
    config.max_samples = 50  # Limited for example
    
    # Setup and run training
    setup_result = setup_training(config)
    trainer = setup_result['trainer']
    results = run_training(trainer)
    
    logger.info(f"Full training completed! Results: {results}")
    return results


def example_evaluation():
    """Example: Evaluate a trained model."""
    
    logger.info("=== Example 4: Model Evaluation ===")
    
    # This would typically load a pre-trained model
    # For this example, we'll create a simple setup
    config = create_quick_test_config(
        data_dir="./data",
        output_dir="./outputs/evaluation",
        max_samples=5
    )
    
    setup_result = setup_training(config)
    trainer = setup_result['trainer']
    test_dataset = setup_result['test_dataset']
    
    # Run evaluation
    if test_dataset:
        from training.utils import evaluate_model
        eval_results = evaluate_model(trainer, test_dataset)
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    else:
        logger.warning("No test dataset available for evaluation")
        return None


def main():
    """Run all examples."""
    
    print("\n" + "="*60)
    print("FLUID DYNAMICS TRAINING EXAMPLES")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs("./outputs", exist_ok=True)
    
    try:
        # Example 1: Quick test
        print("\n🚀 Running Quick Test Example...")
        example_quick_test()
        
        # Example 2: Custom configuration  
        print("\n⚙️  Running Custom Configuration Example...")
        example_custom_configuration()
        
        # Uncomment for full training (takes longer)
        # print("\n🎯 Running Full Training Example...")
        # example_full_training()
        
        # Example 4: Evaluation
        print("\n📊 Running Evaluation Example...")
        example_evaluation()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nCheck the ./outputs/ directory for results.")
        print("Configuration files, logs, and model checkpoints are saved there.")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)