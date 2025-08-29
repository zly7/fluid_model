import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
import io
from PIL import Image
import pandas as pd

logger = logging.getLogger(__name__)

class TensorBoardLogger:
    """
    Comprehensive TensorBoard logger for fluid dynamics transformer training.
    
    Provides logging for scalar metrics, histograms, model graphs, and predictions.
    """
    
    def __init__(self, log_dir: Union[str, Path], 
                 flush_secs: int = 120,
                 max_queue: int = 10):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            flush_secs: Frequency to flush logs to disk
            max_queue: Maximum number of outstanding log entries
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs,
            max_queue=max_queue
        )
        
        # Track logged scalars for summary
        self.scalar_history = {}
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)
        
        # Track for history
        if tag not in self.scalar_history:
            self.scalar_history[tag] = []
        self.scalar_history[tag].append((step, value))
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple related scalars."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
        # Track individual scalars
        for tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}"
            if full_tag not in self.scalar_history:
                self.scalar_history[full_tag] = []
            self.scalar_history[full_tag].append((step, value))
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """Log histogram of values."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, img: Union[torch.Tensor, np.ndarray, Image.Image], step: int):
        """Log an image."""
        if isinstance(img, Image.Image):
            # Convert PIL Image to numpy
            img = np.array(img)
        
        if isinstance(img, np.ndarray):
            # Convert numpy to torch tensor (H, W, C) -> (C, H, W)
            if img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
                img = torch.from_numpy(img).permute(2, 0, 1)
            elif img.ndim == 2:
                img = torch.from_numpy(img).unsqueeze(0)
        
        self.writer.add_image(tag, img, step)
    
    def log_figure(self, tag: str, figure: plt.Figure, step: int, close: bool = True):
        """Log a matplotlib figure."""
        # Convert figure to image
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to PIL Image then to tensor
        img = Image.open(buf)
        self.log_image(tag, img, step)
        
        buf.close()
        if close:
            plt.close(figure)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text data."""
        self.writer.add_text(tag, text, step)
    
    def log_model_graph(self, model: nn.Module, input_sample: torch.Tensor):
        """Log model computational graph."""
        try:
            self.writer.add_graph(model, input_sample)
            logger.info("Model graph logged successfully")
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
    
    def flush(self):
        """Flush pending logs."""
        self.writer.flush()

def log_training_metrics(logger: TensorBoardLogger, 
                        metrics: Dict[str, float], 
                        step: int, 
                        phase: str = 'train'):
    """
    Log training metrics with proper organization.
    
    Args:
        logger: TensorBoard logger instance
        metrics: Dictionary of metric name -> value
        step: Training step
        phase: Training phase ('train', 'val', 'test')
    """
    for metric_name, value in metrics.items():
        tag = f"{phase}/{metric_name}"
        logger.log_scalar(tag, value, step)

def log_loss_components(logger: TensorBoardLogger,
                       loss_dict: Dict[str, torch.Tensor],
                       step: int,
                       phase: str = 'train'):
    """
    Log loss components from FluidDynamicsLoss.
    
    Args:
        logger: TensorBoard logger instance
        loss_dict: Loss components dictionary
        step: Training step
        phase: Training phase
    """
    # Convert tensors to floats
    loss_values = {}
    for component, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            loss_values[component] = value.item()
        else:
            loss_values[component] = value
    
    # Log as grouped scalars
    logger.log_scalars(f"loss/{phase}", loss_values, step)

def log_equipment_metrics(logger: TensorBoardLogger,
                         equipment_metrics: Dict[str, Dict[str, float]],
                         step: int,
                         phase: str = 'val'):
    """
    Log equipment-specific metrics.
    
    Args:
        logger: TensorBoard logger instance
        equipment_metrics: Nested dict {metric_type: {equipment: value}}
        step: Training step
        phase: Training phase
    """
    for metric_type, equipment_dict in equipment_metrics.items():
        # Log as grouped scalars
        tag = f"{phase}_equipment/{metric_type}"
        logger.log_scalars(tag, equipment_dict, step)

def log_model_parameters(logger: TensorBoardLogger,
                        model: nn.Module,
                        step: int):
    """
    Log model parameter histograms and statistics.
    
    Args:
        logger: TensorBoard logger instance
        model: PyTorch model
        step: Training step
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Log parameter values
            logger.log_histogram(f"parameters/{name}", param, step)
            
            # Log gradient values
            logger.log_histogram(f"gradients/{name}", param.grad, step)
            
            # Log gradient norm
            grad_norm = param.grad.norm().item()
            logger.log_scalar(f"gradient_norms/{name}", grad_norm, step)

def log_learning_rate(logger: TensorBoardLogger,
                     optimizer: torch.optim.Optimizer,
                     step: int):
    """
    Log current learning rates.
    
    Args:
        logger: TensorBoard logger instance
        optimizer: PyTorch optimizer
        step: Training step
    """
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        logger.log_scalar(f"learning_rate/group_{i}", lr, step)

def log_prediction_samples(logger: TensorBoardLogger,
                          predictions: torch.Tensor,
                          targets: torch.Tensor,
                          equipment_dims: Dict[str, int],
                          step: int,
                          num_samples: int = 3,
                          equipment_types: Optional[List[str]] = None):
    """
    Log prediction vs target samples for visualization.
    
    Args:
        logger: TensorBoard logger instance
        predictions: Model predictions (batch_size, seq_len, dims)
        targets: Ground truth targets (batch_size, seq_len, dims)
        equipment_dims: Equipment dimension mapping
        step: Training step
        num_samples: Number of samples to visualize
        equipment_types: Specific equipment types to visualize
    """
    if equipment_types is None:
        equipment_types = ['B', 'C', 'P']  # Most important equipment
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    batch_size = min(num_samples, predictions.shape[0])
    
    # Calculate equipment indices
    equipment_indices = {}
    start_idx = 0
    equipment_order = ['B', 'C', 'H', 'N', 'P', 'R', 'T&E']
    
    for eq_type in equipment_order:
        if eq_type in equipment_dims:
            end_idx = start_idx + equipment_dims[eq_type]
            equipment_indices[eq_type] = (start_idx, end_idx)
            start_idx = end_idx
    
    for eq_type in equipment_types:
        if eq_type not in equipment_indices:
            continue
            
        start_idx, end_idx = equipment_indices[eq_type]
        
        for sample_idx in range(batch_size):
            # Extract equipment predictions and targets
            pred_eq = predictions[sample_idx, :, start_idx:end_idx]
            target_eq = targets[sample_idx, :, start_idx:end_idx]
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Equipment {eq_type} - Sample {sample_idx}')
            
            # Plot first few dimensions
            max_dims = min(4, pred_eq.shape[1])
            
            for dim_idx in range(max_dims):
                ax = axes[dim_idx // 2, dim_idx % 2]
                
                ax.plot(pred_eq[:, dim_idx], label='Prediction', alpha=0.7)
                ax.plot(target_eq[:, dim_idx], label='Target', alpha=0.7)
                ax.set_title(f'Dimension {dim_idx}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Log figure
            tag = f"predictions/{eq_type}/sample_{sample_idx}"
            logger.log_figure(tag, fig, step)

def create_correlation_heatmap(predictions: np.ndarray,
                             targets: np.ndarray,
                             equipment_dims: Dict[str, int],
                             title: str = "Prediction-Target Correlation") -> plt.Figure:
    """
    Create correlation heatmap between predictions and targets.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        equipment_dims: Equipment dimensions
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Flatten sequences for correlation
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    target_flat = targets.reshape(-1, targets.shape[-1])
    
    # Calculate correlations for each dimension
    correlations = []
    equipment_labels = []
    
    start_idx = 0
    equipment_order = ['B', 'C', 'H', 'N', 'P', 'R', 'T&E']
    
    for eq_type in equipment_order:
        if eq_type in equipment_dims:
            end_idx = start_idx + equipment_dims[eq_type]
            
            for dim_idx in range(start_idx, end_idx):
                pred_dim = pred_flat[:, dim_idx]
                target_dim = target_flat[:, dim_idx]
                
                # Calculate correlation
                if np.std(pred_dim) > 1e-8 and np.std(target_dim) > 1e-8:
                    corr = np.corrcoef(pred_dim, target_dim)[0, 1]
                else:
                    corr = 0.0
                
                correlations.append(corr)
                equipment_labels.append(f"{eq_type}_{dim_idx-start_idx}")
            
            start_idx = end_idx
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Reshape correlations for heatmap
    correlations = np.array(correlations).reshape(1, -1)
    
    sns.heatmap(correlations, 
                xticklabels=equipment_labels,
                yticklabels=['Correlation'],
                annot=False,
                cmap='RdYlBu_r',
                center=0,
                vmin=-1, vmax=1,
                ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel('Equipment Dimensions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def log_model_architecture(logger: TensorBoardLogger,
                          model: nn.Module,
                          input_sample: torch.Tensor):
    """
    Log detailed model architecture information.
    
    Args:
        logger: TensorBoard logger instance
        model: PyTorch model
        input_sample: Sample input for graph tracing
    """
    # Log computational graph
    logger.log_model_graph(model, input_sample)
    
    # Log model summary as text
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    architecture_info = f"""
    Model Architecture Summary:
    ==========================
    Total Parameters: {total_params:,}
    Trainable Parameters: {trainable_params:,}
    Non-trainable Parameters: {total_params - trainable_params:,}
    
    Model Structure:
    {str(model)}
    """
    
    logger.log_text("model/architecture", architecture_info, 0)

def log_training_progress(logger: TensorBoardLogger,
                         epoch: int,
                         train_loss: float,
                         val_loss: float,
                         val_metrics: Dict[str, float],
                         learning_rate: float,
                         epoch_time: float):
    """
    Log comprehensive training progress.
    
    Args:
        logger: TensorBoard logger instance
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        val_metrics: Validation metrics
        learning_rate: Current learning rate
        epoch_time: Time taken for epoch
    """
    # Log losses
    logger.log_scalars("loss", {
        "train": train_loss,
        "validation": val_loss
    }, epoch)
    
    # Log validation metrics
    for metric_name, value in val_metrics.items():
        logger.log_scalar(f"metrics/{metric_name}", value, epoch)
    
    # Log learning rate
    logger.log_scalar("training/learning_rate", learning_rate, epoch)
    
    # Log timing
    logger.log_scalar("training/epoch_time", epoch_time, epoch)

def create_prediction_comparison_plot(predictions: np.ndarray,
                                    targets: np.ndarray,
                                    equipment_type: str,
                                    sample_idx: int = 0,
                                    max_timesteps: int = 100) -> plt.Figure:
    """
    Create detailed prediction comparison plot.
    
    Args:
        predictions: Predictions for equipment
        targets: Targets for equipment
        equipment_type: Type of equipment
        sample_idx: Sample index to plot
        max_timesteps: Maximum timesteps to plot
        
    Returns:
        Matplotlib figure
    """
    # Limit timesteps for readability
    timesteps = min(max_timesteps, predictions.shape[0])
    pred_sample = predictions[:timesteps, sample_idx] if predictions.ndim > 1 else predictions[:timesteps]
    target_sample = targets[:timesteps, sample_idx] if targets.ndim > 1 else targets[:timesteps]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series comparison
    ax1.plot(pred_sample, label='Prediction', alpha=0.8, linewidth=2)
    ax1.plot(target_sample, label='Target', alpha=0.8, linewidth=2)
    ax1.set_title(f'{equipment_type} - Time Series Comparison')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2.scatter(target_sample, pred_sample, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(np.min(target_sample), np.min(pred_sample))
    max_val = max(np.max(target_sample), np.max(pred_sample))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    ax2.set_xlabel('Target Value')
    ax2.set_ylabel('Predicted Value')
    ax2.set_title('Prediction vs Target Scatter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig