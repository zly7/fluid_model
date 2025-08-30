"""
Custom training callbacks for fluid dynamics models.
"""

import torch
import logging
from typing import Dict, Any, Optional
import time
import os
from pathlib import Path

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import IntervalStrategy
import numpy as np

logger = logging.getLogger(__name__)


class FluidTrainingCallbacks(TrainerCallback):
    """
    Custom callbacks for fluid dynamics training.
    
    Provides:
    - Memory monitoring
    - Training progress logging
    - Custom model checkpointing
    - Performance metrics tracking
    """
    
    def __init__(self, 
                 log_memory_usage: bool = True,
                 save_predictions_every_n_steps: Optional[int] = None,
                 max_prediction_samples: int = 5):
        """
        Initialize callbacks.
        
        Args:
            log_memory_usage: Whether to log GPU memory usage
            save_predictions_every_n_steps: Save sample predictions every N steps
            max_prediction_samples: Maximum samples to save for predictions
        """
        self.log_memory_usage = log_memory_usage
        self.save_predictions_every_n_steps = save_predictions_every_n_steps
        self.max_prediction_samples = max_prediction_samples
        
        # Tracking variables
        self.training_start_time = None
        self.step_times = []
        self.best_eval_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        self.training_start_time = time.time()
        logger.info("=== Starting Fluid Dynamics Training ===")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Total epochs: {args.num_train_epochs}")
        logger.info(f"Train batch size: {args.per_device_train_batch_size}")
        logger.info(f"Eval batch size: {args.per_device_eval_batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Weight decay: {args.weight_decay}")
        
        if self.log_memory_usage and torch.cuda.is_available():
            self._log_memory_usage("Training start")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            logger.info(f"=== Training Completed ===")
            logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
            logger.info(f"Total steps: {state.global_step}")
            logger.info(f"Best eval loss: {self.best_eval_loss:.6f}")
            
            if self.step_times:
                avg_step_time = np.mean(self.step_times[-100:])  # Average of last 100 steps
                logger.info(f"Average step time (last 100): {avg_step_time:.3f} seconds")
        
        if self.log_memory_usage and torch.cuda.is_available():
            self._log_memory_usage("Training end")
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each epoch."""
        epoch_num = int(state.epoch) if state.epoch is not None else 0
        logger.info(f"--- Starting Epoch {epoch_num + 1}/{args.num_train_epochs} ---")
        
        if self.log_memory_usage and torch.cuda.is_available():
            self._log_memory_usage(f"Epoch {epoch_num + 1} start")
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each epoch."""
        epoch_num = int(state.epoch) if state.epoch is not None else 0
        logger.info(f"--- Completed Epoch {epoch_num + 1}/{args.num_train_epochs} ---")
        
        # Log epoch statistics
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            if 'train_loss' in latest_log:
                logger.info(f"Epoch {epoch_num + 1} train loss: {latest_log['train_loss']:.6f}")
            if 'eval_loss' in latest_log:
                logger.info(f"Epoch {epoch_num + 1} eval loss: {latest_log['eval_loss']:.6f}")
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each training step."""
        self.step_start_time = time.time()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        if hasattr(self, 'step_start_time'):
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
            
            # Keep only last 1000 step times to avoid memory issues
            if len(self.step_times) > 1000:
                self.step_times = self.step_times[-1000:]
        
        # Log progress periodically
        if state.global_step % 100 == 0:
            if self.step_times:
                avg_time = np.mean(self.step_times[-10:])
                logger.info(f"Step {state.global_step}: avg step time = {avg_time:.3f}s")
            
            if self.log_memory_usage and torch.cuda.is_available():
                self._log_memory_usage(f"Step {state.global_step}")
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, 
                   metrics: Dict[str, float] = None, **kwargs):
        """Called after evaluation."""
        if metrics:
            eval_loss = metrics.get('eval_loss')
            if eval_loss is not None:
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.epochs_without_improvement = 0
                    logger.info(f"🎉 New best eval loss: {eval_loss:.6f}")
                else:
                    self.epochs_without_improvement += 1
                
                # Log key metrics
                logger.info(f"Evaluation results (step {state.global_step}):")
                for key, value in metrics.items():
                    if key.startswith('eval_'):
                        logger.info(f"  {key}: {value:.6f}")
        
        # Save sample predictions if configured
        if (self.save_predictions_every_n_steps and 
            state.global_step % self.save_predictions_every_n_steps == 0):
            self._save_sample_predictions(args, state, **kwargs)
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when saving a checkpoint."""
        logger.info(f"💾 Checkpoint saved at step {state.global_step}")
        
        if self.log_memory_usage and torch.cuda.is_available():
            self._log_memory_usage("Checkpoint save")
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, 
              logs: Dict[str, float] = None, **kwargs):
        """Called when logging."""
        if logs and state.global_step % args.logging_steps == 0:
            # Enhanced logging with additional context
            step_info = f"Step {state.global_step}"
            if state.epoch is not None:
                step_info += f" (Epoch {state.epoch:.2f})"
            
            # Log training metrics with better formatting
            if 'loss' in logs:
                logger.info(f"{step_info}: loss = {logs['loss']:.6f}")
            if 'learning_rate' in logs:
                logger.info(f"{step_info}: lr = {logs['learning_rate']:.2e}")
    
    def _log_memory_usage(self, context: str):
        """Log GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_memory = torch.cuda.max_memory_allocated() / 1024**3   # GB
            
            logger.info(f"GPU Memory ({context}): "
                       f"Allocated={memory_allocated:.2f}GB, "
                       f"Reserved={memory_reserved:.2f}GB, "
                       f"Max={max_memory:.2f}GB")
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
    
    def _save_sample_predictions(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Save sample predictions for analysis."""
        try:
            model = kwargs.get('model')
            eval_dataloader = kwargs.get('eval_dataloader')
            
            if model is None or eval_dataloader is None:
                logger.warning("Model or eval_dataloader not available for saving predictions")
                return
            
            logger.info(f"Saving sample predictions at step {state.global_step}")
            
            # Get a few samples
            model.eval()
            predictions_dir = os.path.join(args.output_dir, "sample_predictions")
            os.makedirs(predictions_dir, exist_ok=True)
            
            sample_count = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_dataloader):
                    if sample_count >= self.max_prediction_samples:
                        break
                    
                    # Move to device
                    if hasattr(model, 'device'):
                        device = model.device
                    else:
                        device = next(model.parameters()).device
                    
                    # Prepare inputs
                    inputs = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.to(device)
                        else:
                            inputs[key] = value
                    
                    # Get predictions
                    outputs = model(input_ids=inputs['input'])
                    predictions = outputs['logits']
                    
                    # Save to file
                    sample_data = {
                        'step': state.global_step,
                        'batch_idx': batch_idx,
                        'input': inputs['input'].cpu().numpy(),
                        'target': inputs.get('target', torch.zeros_like(inputs['input'])).cpu().numpy(),
                        'predictions': predictions.cpu().numpy(),
                        'metadata': inputs.get('metadata', [])
                    }
                    
                    save_path = os.path.join(predictions_dir, f"step_{state.global_step}_sample_{sample_count}.npz")
                    np.savez_compressed(save_path, **sample_data)
                    
                    sample_count += 1
                    break  # Only save one sample per call to avoid overhead
                    
            logger.info(f"Saved {sample_count} sample predictions to {predictions_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save sample predictions: {e}")


class MemoryMonitorCallback(TrainerCallback):
    """Lightweight callback to monitor memory usage."""
    
    def __init__(self, log_every_n_steps: int = 500):
        self.log_every_n_steps = log_every_n_steps
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log memory usage every N steps."""
        if state.global_step % self.log_every_n_steps == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"Step {state.global_step}: GPU memory = {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class LearningRateMonitorCallback(TrainerCallback):
    """Monitor and log learning rate changes."""
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log learning rate periodically."""
        if state.global_step % args.logging_steps == 0:
            # Get current learning rate from optimizer
            optimizer = kwargs.get('optimizer')
            if optimizer and optimizer.param_groups:
                current_lr = optimizer.param_groups[0]['lr']
                logger.debug(f"Step {state.global_step}: Learning rate = {current_lr:.2e}")


def create_default_callbacks(
    log_memory_usage: bool = True,
    save_predictions_every_n_steps: Optional[int] = None,
    memory_monitor_steps: int = 500
) -> list:
    """
    Create default set of training callbacks.
    
    Args:
        log_memory_usage: Whether to log memory usage
        save_predictions_every_n_steps: Save predictions every N steps
        memory_monitor_steps: Monitor memory every N steps
        
    Returns:
        List of callback instances
    """
    callbacks = [
        FluidTrainingCallbacks(
            log_memory_usage=log_memory_usage,
            save_predictions_every_n_steps=save_predictions_every_n_steps
        )
    ]
    
    if torch.cuda.is_available():
        callbacks.append(MemoryMonitorCallback(log_every_n_steps=memory_monitor_steps))
    
    callbacks.append(LearningRateMonitorCallback())
    
    return callbacks