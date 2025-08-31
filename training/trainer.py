"""
Custom trainer for fluid dynamics models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union, Tuple
import logging
import os
from pathlib import Path
import json

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import EvalPrediction
import numpy as np

from .config import TrainingConfig
from data.normalizer import DataNormalizer

logger = logging.getLogger(__name__)


class FluidTrainer(Trainer):
    """
    Custom trainer for fluid dynamics models.
    
    Extends transformers.Trainer with:
    - Custom evaluation metrics
    - Denormalization for evaluation
    - Fluid-specific logging and callbacks
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        training_config: TrainingConfig,
        train_dataloader,
        eval_dataloader=None,
        normalizer: Optional[DataNormalizer] = None,
        **kwargs
    ):
        """
        Initialize FluidTrainer.
        
        Args:
            model: The fluid dynamics model
            args: Transformers training arguments
            training_config: Fluid-specific training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            normalizer: Data normalizer for denormalization during evaluation
            **kwargs: Additional arguments passed to Trainer
        """
        self.training_config = training_config
        self.normalizer = normalizer
        
        # No need for custom compute_metrics since model loss is sufficient
        # kwargs['compute_metrics'] = self._create_compute_metrics_fn()
        
        # Initialize parent trainer
        # Note: We pass train_dataset and eval_dataset as None since we provide dataloaders directly
        super().__init__(
            model=model,
            args=args,
            train_dataset=None,  # We use dataloaders instead
            eval_dataset=None,
            **kwargs
        )
        
        # Store dataloaders
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader
        
        # Add early stopping callback if configured
        if training_config.early_stopping_patience > 0:
            self.add_callback(EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience,
                early_stopping_threshold=training_config.early_stopping_threshold
            ))
        
        logger.info(f"FluidTrainer initialized with {len(self.get_train_dataloader())} training batches")
        if eval_dataloader:
            logger.info(f"Evaluation dataloader has {len(self.get_eval_dataloader())} batches")
    
    def get_train_dataloader(self):
        """Return the training dataloader."""
        return self._train_dataloader
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Return the evaluation dataloader."""
        return self._eval_dataloader if self._eval_dataloader is not None else None
    
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.
        
        Args:
            eval_dataset: Ignored, we use the dataloader
            ignore_keys: Metric keys to ignore
            metric_key_prefix: Prefix for metric keys
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Call parent evaluate method
        eval_results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Log key metrics
        if f"{metric_key_prefix}_loss" in eval_results:
            logger.info(f"Evaluation loss: {eval_results[f'{metric_key_prefix}_loss']:.6f}")
        
        return eval_results
    
    def log(self, logs: Dict[str, float]):
        """
        Custom logging with additional context.
        
        Args:
            logs: Dictionary of logs to record
        """
        # Add training configuration context to logs
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        
        if self.state.global_step is not None:
            logs["step"] = self.state.global_step
        
        # Add learning rate
        if hasattr(self.lr_scheduler, 'get_last_lr'):
            logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        
        # Call parent log method
        super().log(logs)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model with additional metadata.
        
        Args:
            output_dir: Directory to save to
            _internal_call: Whether this is an internal call
        """
        # Call parent save_model
        super().save_model(output_dir, _internal_call)
        
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Save training configuration
        config_path = os.path.join(output_dir, "training_config.json")
        self.training_config.save_to_file(config_path)
        
        # Save normalizer if available
        if self.normalizer is not None:
            normalizer_path = os.path.join(output_dir, "normalizer_stats.npz")
            self.normalizer.save_stats(normalizer_path)
            logger.info(f"Saved normalizer stats to {normalizer_path}")
        
        # Save additional metadata
        metadata = {
            "training_completed": self.state.epoch >= self.args.num_train_epochs if self.state.epoch else False,
            "total_steps": self.state.global_step,
            "epochs_completed": self.state.epoch,
            "best_metric": self.state.best_metric,
            "model_info": self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        }
        
        metadata_path = os.path.join(output_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model and metadata saved to {output_dir}")
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step.
        
        Args:
            model: The model to evaluate
            inputs: Model inputs
            prediction_loss_only: Whether to return only loss
            ignore_keys: Keys to ignore in inputs
            
        Returns:
            Tuple of (loss, predictions, labels)
        """
        # Ensure model is in eval mode
        model.eval()
        
        # Move inputs to device
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            # Forward pass
            if isinstance(inputs, dict) and 'target' in inputs:
                # Our custom format
                outputs = model(
                    input_ids=inputs['input'],
                    labels=inputs['target'],
                    prediction_mask=inputs.get('prediction_mask')
                )
                loss = outputs.get('loss')
                predictions = outputs.get('logits')
                labels = inputs['target']
            else:
                # Standard transformers format
                outputs = model(**inputs)
                loss = outputs.get('loss')
                predictions = outputs.get('logits')
                labels = inputs.get('labels')
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Detach tensors
        if loss is not None:
            loss = loss.detach()
        if predictions is not None:
            predictions = predictions.detach()
        if labels is not None:
            labels = labels.detach()
        
        return (loss, predictions, labels)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation.
        
        Args:
            model: The model
            inputs: Model inputs
            return_outputs: Whether to return outputs
            
        Returns:
            Loss tensor, optionally with outputs
        """
        # Handle our custom input format
        if isinstance(inputs, dict) and 'input' in inputs:
            outputs = model(
                input_ids=inputs['input'],
                labels=inputs.get('target'),
                prediction_mask=inputs.get('prediction_mask')
            )
        else:
            outputs = model(**inputs)
        
        loss = outputs.get('loss')
        
        if loss is None:
            raise ValueError("Model did not return a loss")
        
        return (loss, outputs) if return_outputs else loss


def create_fluid_trainer(
    model: nn.Module,
    training_config: TrainingConfig,
    train_dataloader,
    eval_dataloader=None,
    normalizer: Optional[DataNormalizer] = None,
    **trainer_kwargs
) -> FluidTrainer:
    """
    Create a FluidTrainer with proper configuration.
    
    Args:
        model: The fluid dynamics model
        training_config: Training configuration
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader (optional)
        normalizer: Data normalizer (optional)
        **trainer_kwargs: Additional trainer arguments
        
    Returns:
        Configured FluidTrainer instance
    """
    # Convert training config to transformers arguments
    training_args = TrainingArguments(**training_config.get_transformers_training_args())
    
    # Create trainer
    trainer = FluidTrainer(
        model=model,
        args=training_args,
        training_config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        normalizer=normalizer,
        **trainer_kwargs
    )
    
    logger.info("FluidTrainer created successfully")
    return trainer