#!/usr/bin/env python3
"""
Main entry point for the Fluid Dynamics Transformer.

Provides command-line interface for training, validation, and inference.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from datetime import datetime
import json
import numpy as np

# Import our modules
from data import DataProcessor, create_data_loaders
from models import DecoderOnlyTransformer
from training import FluidDynamicsTrainer, FluidDynamicsLoss
from utils import FluidDynamicsEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fluid_dynamics_transformer.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fluid Dynamics Transformer - Gas Pipeline Network Prediction"
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir', 
        type=str, 
        required=True,
        help='Path to data directory containing train/test folders'
    )
    
    # Model arguments
    parser.add_argument(
        '--d-model', 
        type=int, 
        default=256,
        help='Model dimension (default: 256)'
    )
    parser.add_argument(
        '--nhead', 
        type=int, 
        default=8,
        help='Number of attention heads (default: 8)'
    )
    parser.add_argument(
        '--num-decoder-layers', 
        type=int, 
        default=6,
        help='Number of decoder layers (default: 6)'
    )
    parser.add_argument(
        '--dropout', 
        type=float, 
        default=0.1,
        help='Dropout rate (default: 0.1)'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--weight-decay', 
        type=float, 
        default=0.01,
        help='Weight decay (default: 0.01)'
    )
    parser.add_argument(
        '--patience', 
        type=int, 
        default=10,
        help='Early stopping patience (default: 10)'
    )
    
    # Loss function arguments
    parser.add_argument(
        '--mse-weight', 
        type=float, 
        default=1.0,
        help='Weight for MSE loss component (default: 1.0)'
    )
    parser.add_argument(
        '--smoothness-weight', 
        type=float, 
        default=0.1,
        help='Weight for temporal smoothness loss (default: 0.1)'
    )
    parser.add_argument(
        '--physical-weight', 
        type=float, 
        default=0.05,
        help='Weight for physical consistency loss (default: 0.05)'
    )
    
    # Data processing arguments
    parser.add_argument(
        '--sequence-length', 
        type=int, 
        default=1440,
        help='Sequence length in minutes (default: 1440 = 24 hours)'
    )
    parser.add_argument(
        '--scaler-type', 
        choices=['standard', 'minmax'], 
        default='standard',
        help='Data normalization method (default: standard)'
    )
    parser.add_argument(
        '--val-split', 
        type=float, 
        default=0.15,
        help='Validation split ratio (default: 0.15)'
    )
    
    # Mode and I/O arguments
    parser.add_argument(
        '--mode', 
        choices=['train', 'evaluate', 'predict', 'data-info'], 
        default='train',
        help='Execution mode (default: train)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory for models and results (default: outputs)'
    )
    parser.add_argument(
        '--checkpoint-path', 
        type=str,
        help='Path to model checkpoint for evaluation/prediction'
    )
    parser.add_argument(
        '--save-predictions', 
        action='store_true',
        help='Save predictions to CSV files'
    )
    
    # Device arguments
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        help='Device to use (cuda, cpu, or auto for automatic selection)'
    )
    parser.add_argument(
        '--num-workers', 
        type=int, 
        default=4,
        help='Number of data loader workers (default: 4)'
    )
    
    # Logging arguments
    parser.add_argument(
        '--log-interval', 
        type=int, 
        default=10,
        help='Log training progress every N batches (default: 10)'
    )
    parser.add_argument(
        '--save-interval', 
        type=int, 
        default=5,
        help='Save model checkpoint every N epochs (default: 5)'
    )
    
    # Advanced arguments
    parser.add_argument(
        '--accumulate-grad-batches', 
        type=int, 
        default=1,
        help='Gradient accumulation steps (default: 1)'
    )
    parser.add_argument(
        '--clip-grad-norm', 
        type=float, 
        default=1.0,
        help='Gradient clipping norm (default: 1.0)'
    )
    parser.add_argument(
        '--warmup-steps', 
        type=int, 
        default=1000,
        help='Learning rate warmup steps (default: 1000)'
    )
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """Setup compute device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS device")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using specified device: {device}")
    
    return device

def load_data(args) -> tuple:
    """Load and process data."""
    logger.info("Loading and processing data...")
    
    # Initialize data processor
    processor = DataProcessor(data_dir=args.data_dir)
    
    # Load data
    try:
        train_data, test_data = processor.load_all_data()
        logger.info(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    # Get equipment dimensions for model configuration
    equipment_dims = processor.get_equipment_dimensions()
    total_output_dim = sum(equipment_dims.values())
    
    logger.info("Equipment dimensions:")
    for eq_type, dim in equipment_dims.items():
        logger.info(f"  {eq_type}: {dim}")
    logger.info(f"Total output dimensions: {total_output_dim}")
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            val_split=args.val_split,
            scaler_type=args.scaler_type,
            num_workers=args.num_workers
        )
        logger.info("Data loaders created successfully")
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise
    
    return train_loader, val_loader, test_loader, equipment_dims, total_output_dim

def create_model(args, total_output_dim: int, device: torch.device) -> nn.Module:
    """Create and initialize model."""
    logger.info("Creating model...")
    
    model = DecoderOnlyTransformer(
        input_dim=539,  # Boundary conditions dimension
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        total_output_dim=total_output_dim,
        dropout=args.dropout
    )
    
    model = model.to(device)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    return model

def train_model(args, model, train_loader, val_loader, equipment_dims, device):
    """Train the model."""
    logger.info("Starting model training...")
    
    # Create loss function
    criterion = FluidDynamicsLoss(
        equipment_dims=equipment_dims,
        mse_weight=args.mse_weight,
        smoothness_weight=args.smoothness_weight,
        physical_weight=args.physical_weight
    )
    
    # Create trainer
    trainer = FluidDynamicsTrainer(
        model=model,
        criterion=criterion,
        equipment_dims=equipment_dims,
        device=device,
        log_dir=Path(args.output_dir) / 'tensorboard' / datetime.now().strftime('%Y%m%d_%H%M%S'),
        checkpoint_dir=Path(args.output_dir) / 'checkpoints'
    )
    
    # Training configuration
    training_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'patience': args.patience,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'clip_grad_norm': args.clip_grad_norm,
        'warmup_steps': args.warmup_steps,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval
    }
    
    # Start training
    try:
        best_model_path = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            **training_config
        )
        logger.info(f"Training completed. Best model saved at: {best_model_path}")
        return best_model_path
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def evaluate_model(args, model, test_loader, equipment_dims, device):
    """Evaluate the model."""
    logger.info("Evaluating model...")
    
    # Load checkpoint if specified
    if args.checkpoint_path:
        logger.info(f"Loading checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = FluidDynamicsEvaluator(equipment_dims=equipment_dims)
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            boundary_conditions = batch['boundary_conditions'].to(device)
            targets = batch['equipment_targets'].to(device)
            
            # Forward pass
            predictions = model(boundary_conditions)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Combine all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Evaluate
    results = evaluator.evaluate_predictions(predictions, targets, return_detailed=True)
    
    # Generate report
    report = evaluator.generate_report(predictions, targets)
    logger.info(f"\n{report}")
    
    # Save results
    output_file = Path(args.output_dir) / 'evaluation_results.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                   for k, v in value.items()}
            else:
                json_results[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
        
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {output_file}")
    
    # Save predictions if requested
    if args.save_predictions:
        pred_file = Path(args.output_dir) / 'predictions.npy'
        target_file = Path(args.output_dir) / 'targets.npy'
        
        np.save(pred_file, predictions)
        np.save(target_file, targets)
        
        logger.info(f"Predictions saved to: {pred_file}")
        logger.info(f"Targets saved to: {target_file}")

def show_data_info(args):
    """Show information about the dataset."""
    logger.info("Analyzing dataset...")
    
    try:
        processor = DataProcessor(data_dir=args.data_dir)
        train_data, test_data = processor.load_all_data()
        
        # Dataset statistics
        logger.info("Dataset Information:")
        logger.info(f"  Training samples: {len(train_data)}")
        logger.info(f"  Test samples: {len(test_data)}")
        
        # Equipment dimensions
        equipment_dims = processor.get_equipment_dimensions()
        logger.info("\nEquipment Dimensions:")
        total_dims = 0
        for eq_type, dim in equipment_dims.items():
            logger.info(f"  {eq_type}: {dim}")
            total_dims += dim
        logger.info(f"  Total: {total_dims}")
        
        # Data statistics
        if train_data:
            sample_stats = processor.get_data_statistics()
            logger.info(f"\nData Statistics:")
            logger.info(f"  Boundary conditions shape: {sample_stats.get('boundary_shape', 'Unknown')}")
            logger.info(f"  Equipment targets shape: {sample_stats.get('equipment_shape', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        raise

def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device(args.device)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    logger.info(f"Starting Fluid Dynamics Transformer - Mode: {args.mode}")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        if args.mode == 'data-info':
            show_data_info(args)
            
        elif args.mode == 'train':
            # Load data
            train_loader, val_loader, test_loader, equipment_dims, total_output_dim = load_data(args)
            
            # Create model
            model = create_model(args, total_output_dim, device)
            
            # Train model
            best_model_path = train_model(args, model, train_loader, val_loader, equipment_dims, device)
            
            if best_model_path:
                logger.info("Training completed successfully!")
                
                # Optionally evaluate on test set
                logger.info("Evaluating on test set...")
                args.checkpoint_path = str(best_model_path)
                evaluate_model(args, model, test_loader, equipment_dims, device)
        
        elif args.mode == 'evaluate':
            if not args.checkpoint_path:
                logger.error("Checkpoint path required for evaluation mode")
                return
            
            # Load data
            _, _, test_loader, equipment_dims, total_output_dim = load_data(args)
            
            # Create model
            model = create_model(args, total_output_dim, device)
            
            # Evaluate
            evaluate_model(args, model, test_loader, equipment_dims, device)
        
        elif args.mode == 'predict':
            logger.error("Prediction mode not yet implemented")
            # TODO: Implement prediction mode for new data
        
        logger.info("Execution completed successfully!")
    
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == '__main__':
    main()