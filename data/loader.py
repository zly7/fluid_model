import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any
import logging

from .dataset import FluidDataset

logger = logging.getLogger(__name__)

def custom_collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences and batch creation.
    
    Args:
        batch: List of sample dictionaries from FluidDataset
        
    Returns:
        Batched dictionary with properly collated tensors
    """
    # Extract components from batch
    boundary_tensors = []
    target_tensors = []
    sample_ids = []
    
    has_targets = False
    
    for sample in batch:
        boundary_tensors.append(sample['boundary'])
        sample_ids.append(sample['sample_id'])
        
        if 'targets' in sample and sample['targets'] is not None:
            target_tensors.append(sample['targets'])
            has_targets = True
    
    # Handle different sequence lengths by padding or truncating
    if len(boundary_tensors) > 1:
        # Find the minimum sequence length to avoid padding issues
        min_seq_len = min(tensor.shape[0] for tensor in boundary_tensors)
        
        # Truncate all tensors to minimum length
        boundary_tensors = [tensor[:min_seq_len] for tensor in boundary_tensors]
        
        if has_targets:
            target_tensors = [tensor[:min_seq_len] for tensor in target_tensors]
    
    # Stack into batches
    result = {
        'boundary': torch.stack(boundary_tensors, dim=0),  # (batch, seq_len, boundary_dims)
        'sample_ids': sample_ids
    }
    
    if has_targets:
        result['targets'] = torch.stack(target_tensors, dim=0)  # (batch, seq_len, equipment_dims)
    
    return result

def create_data_loaders(data_dir: str,
                       batch_size: int = 8,
                       num_workers: int = 4,
                       val_split: float = 0.2,
                       sequence_length: int = 1440,
                       normalize: bool = True,
                       scaler_type: str = 'standard',
                       max_samples: Optional[int] = None,
                       shuffle: bool = True,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        val_split: Fraction of training data to use for validation
        sequence_length: Length of time series sequences
        normalize: Whether to normalize the data
        scaler_type: Type of normalization ('standard' or 'minmax')
        max_samples: Maximum samples to load per split (for testing)
        shuffle: Whether to shuffle training data
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating data loaders: batch_size={batch_size}, val_split={val_split}")
    
    # Create datasets
    train_dataset = FluidDataset(
        data_dir=data_dir,
        split='train',
        sequence_length=sequence_length,
        normalize=normalize,
        scaler_type=scaler_type,
        max_samples=max_samples
    )
    
    val_dataset = FluidDataset(
        data_dir=data_dir,
        split='val',
        sequence_length=sequence_length,
        normalize=normalize,
        scaler_type=scaler_type,
        max_samples=max_samples
    )
    
    # Load scalers from training dataset to validation dataset
    if normalize:
        val_dataset.boundary_scaler = train_dataset.boundary_scaler
        val_dataset.equipment_scaler = train_dataset.equipment_scaler
    
    test_dataset = FluidDataset(
        data_dir=data_dir,
        split='test',
        sequence_length=sequence_length,
        normalize=normalize,
        scaler_type=scaler_type,
        max_samples=max_samples
    )
    
    # Load scalers from training dataset to test dataset
    if normalize:
        test_dataset.boundary_scaler = train_dataset.boundary_scaler
        test_dataset.equipment_scaler = train_dataset.equipment_scaler
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for test
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    
    # Log dataset statistics
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples") 
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def create_inference_loader(data_dir: str,
                           batch_size: int = 8,
                           num_workers: int = 4,
                           sequence_length: int = 1440,
                           scaler_path: Optional[str] = None) -> DataLoader:
    """
    Create data loader specifically for inference on test data.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for inference
        num_workers: Number of worker processes
        sequence_length: Length of time series sequences
        scaler_path: Path to saved scalers directory
        
    Returns:
        Test data loader configured for inference
    """
    test_dataset = FluidDataset(
        data_dir=data_dir,
        split='test',
        sequence_length=sequence_length,
        normalize=True,  # Usually want normalization for inference
        scaler_type='standard'
    )
    
    # Load pre-trained scalers if provided
    if scaler_path is not None:
        test_dataset.load_scalers(scaler_path)
        logger.info(f"Loaded scalers from {scaler_path} for inference")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    
    logger.info(f"Created inference loader: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return test_loader

def get_sample_batch(data_loader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Get a sample batch from data loader for testing/debugging.
    
    Args:
        data_loader: DataLoader to sample from
        
    Returns:
        Sample batch dictionary
    """
    try:
        batch = next(iter(data_loader))
        logger.info(f"Sample batch shapes:")
        logger.info(f"  Boundary: {batch['boundary'].shape}")
        
        if 'targets' in batch:
            logger.info(f"  Targets: {batch['targets'].shape}")
        else:
            logger.info(f"  Targets: None (test data)")
            
        logger.info(f"  Sample IDs: {len(batch['sample_ids'])}")
        
        return batch
        
    except Exception as e:
        logger.error(f"Error getting sample batch: {e}")
        raise

class AutoregressiveCollator:
    """
    Custom collator for autoregressive training.
    
    Creates input-target pairs where:
    - Input: boundary conditions + previous equipment state  
    - Target: next equipment state
    """
    
    def __init__(self, 
                 context_length: int = 10,
                 predict_length: int = 1):
        """
        Initialize autoregressive collator.
        
        Args:
            context_length: Length of historical context to use
            predict_length: Length of prediction horizon
        """
        self.context_length = context_length
        self.predict_length = predict_length
    
    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Create autoregressive input-target pairs.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Dictionary with autoregressive sequences
        """
        # First apply standard collation
        collated = custom_collate_fn(batch)
        
        if 'targets' not in collated:
            return collated  # Test data, no autoregressive structure needed
        
        boundary = collated['boundary']  # (batch, seq_len, boundary_dims)
        targets = collated['targets']    # (batch, seq_len, equipment_dims)
        
        batch_size, seq_len, _ = boundary.shape
        
        # Create autoregressive sequences
        inputs = []
        outputs = []
        
        for i in range(self.context_length, seq_len - self.predict_length):
            # Input: boundary conditions + historical equipment states
            context_boundary = boundary[:, i-self.context_length:i+1]  # Include current step
            context_equipment = targets[:, i-self.context_length:i]    # Exclude current step
            
            # Target: next equipment states
            target_equipment = targets[:, i:i+self.predict_length]
            
            inputs.append({
                'boundary': context_boundary,
                'previous_equipment': context_equipment
            })
            outputs.append(target_equipment)
        
        if len(inputs) == 0:
            # Fallback if sequence is too short
            return collated
        
        # Stack sequences
        result = {
            'boundary_sequences': torch.stack([inp['boundary'] for inp in inputs], dim=1),
            'equipment_sequences': torch.stack([inp['previous_equipment'] for inp in inputs], dim=1),
            'target_sequences': torch.stack(outputs, dim=1),
            'sample_ids': collated['sample_ids']
        }
        
        return result