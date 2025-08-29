"""
Base model class for fluid dynamics neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all fluid dynamics models.
    
    All models should:
    1. Inherit from this base class
    2. Implement the forward method
    3. Handle prediction masks correctly
    4. Support standard loss computation
    """
    
    def __init__(self, input_dim: int = 6712, output_dim: int = 6712, **kwargs):
        """
        Initialize base model.
        
        Args:
            input_dim: Input feature dimension (default: 6712)
            output_dim: Output feature dimension (default: 6712)
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.boundary_dims = 538  # First 538 dimensions are boundary conditions
        self.equipment_dims = 6174  # Remaining dimensions are equipment parameters
        
        # Model metadata
        self.model_name = self.__class__.__name__
        self.model_config = kwargs
        
    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            batch: Dictionary containing:
                - 'input': Input tensor [B, T, V=6712]
                - 'attention_mask': Attention mask [B, T, V] (optional)
                - 'prediction_mask': Prediction mask [B, V] (for loss computation)
                
        Returns:
            predictions: Output tensor [B, T, V=6712]
        """
        pass
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss with prediction masking.
        
        Args:
            batch: Input batch with 'target' and 'prediction_mask'
            predictions: Model predictions [B, T, V]
            
        Returns:
            Dictionary with loss components
        """
        targets = batch['target']  # [B, T, V]
        prediction_mask = batch['prediction_mask']  # [B, V]
        
        # Expand prediction mask to match time dimension
        mask = prediction_mask.unsqueeze(1)  # [B, 1, V]
        mask = mask.expand(-1, targets.size(1), -1)  # [B, T, V]
        
        # Compute MSE loss only for predicted variables (mask=1)
        mse_loss = F.mse_loss(predictions, targets, reduction='none')  # [B, T, V]
        masked_loss = mse_loss * mask.float()
        
        # Average over masked positions
        total_loss = masked_loss.sum() / mask.float().sum().clamp(min=1e-8)
        
        # Additional metrics
        with torch.no_grad():
            # Equipment-only loss (excluding boundary conditions)
            equipment_mask = mask[:, :, self.boundary_dims:]  # [B, T, equipment_dims]
            equipment_loss = (masked_loss[:, :, self.boundary_dims:] * equipment_mask.float()).sum() / equipment_mask.float().sum().clamp(min=1e-8)
            
            # MAE for interpretability
            mae_loss = F.l1_loss(predictions, targets, reduction='none')
            masked_mae = mae_loss * mask.float()
            total_mae = masked_mae.sum() / mask.float().sum().clamp(min=1e-8)
        
        return {
            'loss': total_loss,
            'mse_loss': total_loss,
            'equipment_loss': equipment_loss,
            'mae_loss': total_mae
        }
    
    def predict(self, batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Prediction interface for inference.
        
        Args:
            batch: Input batch
            **kwargs: Additional prediction parameters
            
        Returns:
            predictions: Model predictions [B, T, V]
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        param_count = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'total_parameters': param_count,
            'trainable_parameters': trainable_params,
            'config': self.model_config
        }
    
    def save_checkpoint(self, filepath: str, epoch: int = 0, optimizer_state: Optional[Dict] = None, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.state_dict(),
            'model_config': self.model_config,
            'epoch': epoch,
            'model_info': self.get_model_info()
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        # Add any additional info
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        logger.info(f"Model checkpoint loaded from {filepath}")
        return checkpoint
    
    def freeze_parameters(self, freeze_embeddings: bool = False):
        """Freeze model parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = False
            
        logger.info("Model parameters frozen")
    
    def unfreeze_parameters(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
            
        logger.info("Model parameters unfrozen")


class MaskedMSELoss(nn.Module):
    """Masked MSE Loss for equipment prediction."""
    
    def __init__(self, boundary_dims: int = 538):
        super().__init__()
        self.boundary_dims = boundary_dims
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, prediction_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked MSE loss.
        
        Args:
            predictions: [B, T, V]
            targets: [B, T, V]
            prediction_mask: [B, V] where 1=predict, 0=ignore
            
        Returns:
            Scalar loss
        """
        # Expand mask to time dimension
        mask = prediction_mask.unsqueeze(1)  # [B, 1, V]
        mask = mask.expand(-1, targets.size(1), -1)  # [B, T, V]
        
        # Compute MSE loss
        mse = F.mse_loss(predictions, targets, reduction='none')  # [B, T, V]
        
        # Apply mask and average
        masked_mse = mse * mask.float()
        loss = masked_mse.sum() / mask.float().sum().clamp(min=1e-8)
        
        return loss