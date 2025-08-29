import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import warnings
from tensordict import TensorDict

from .processor import DataProcessor

logger = logging.getLogger(__name__)

class FluidDataset(Dataset):
    """
    PyTorch Dataset for gas pipeline network fluid dynamics data.
    
    Data Format:
    - Input: [B, T, V] where B=batch, T=time_steps (default 3), V=variates (6712)
    - Target: [B, T, V] same format, time-shifted by 1 minute
    - Mask: [V] prediction mask, boundary=0, equipment=1
    
    Features:
    - Boundary conditions (538 dims) + Equipment predictions (6174 dims) = 6712 total
    - 30-min boundary data interpolated to 1-min intervals
    - Autoregressive structure: each minute predicts next minute
    - TensorDict packaging for structured data handling
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 sequence_length: int = 3,
                 normalize: bool = True,
                 scaler_type: str = 'standard',
                 cache_data: bool = True,
                 max_samples: Optional[int] = None,
                 max_sequences_per_sample: Optional[int] = None):
        """
        Initialize FluidDataset.
        
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            sequence_length: Length of time series sequences (default: 3 for 3 minutes)
            normalize: Whether to normalize the data
            scaler_type: 'standard' or 'minmax' normalization
            cache_data: Whether to cache loaded data in memory
            max_samples: Maximum number of samples to load (for testing)
            max_sequences_per_sample: Maximum sequences per sample (for memory control)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.cache_data = cache_data
        self.max_samples = max_samples
        self.max_sequences_per_sample = max_sequences_per_sample
        
        # Initialize data processor
        self.processor = DataProcessor(str(self.data_dir))
        
        # Data dimensions
        self.total_dims = 6712  # 538 boundary + 6174 equipment
        
        # Initialize scalers
        self.scaler = None
        
        # Cache for loaded data
        self.data_cache = {} if cache_data else None
        
        # Prediction mask (fixed across all samples)
        self.prediction_mask = None
        
        # Load and prepare data
        self._load_data()
        
        logger.info(f"FluidDataset initialized: {self.split} split, {len(self.all_sequences)} sequences, "
                   f"sequence_length={self.sequence_length}, total_dims={self.total_dims}")
    
    def _load_data(self):
        """Load and prepare dataset."""
        # Get sample directories
        if self.split == 'train':
            all_samples = self.processor.get_sample_directories('train')
            # Split train data: use first 80% for train, last 20% for validation
            train_size = int(0.8 * len(all_samples))
            self.sample_dirs = all_samples[:train_size]
        elif self.split == 'val':
            all_samples = self.processor.get_sample_directories('train')
            train_size = int(0.8 * len(all_samples))
            self.sample_dirs = all_samples[train_size:]
        elif self.split == 'test':
            self.sample_dirs = self.processor.get_sample_directories('test')
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Limit samples if specified
        if self.max_samples is not None:
            self.sample_dirs = self.sample_dirs[:self.max_samples]
        
        # Load all sequences from all samples
        self._load_all_sequences()
        
        # Fit scaler on training data
        if self.normalize and self.split == 'train':
            self._fit_scaler()
        
        # Validate that we have data
        if len(self.all_sequences) == 0:
            raise ValueError(f"No sequences found for split: {self.split}")
            
        logger.info(f"Loaded {len(self.all_sequences)} sequences from {len(self.sample_dirs)} samples for {self.split} split")
    
    def _load_all_sequences(self):
        """Load all sequences from all samples."""
        self.all_sequences = []
        self.sequence_metadata = []
        
        for sample_dir in self.sample_dirs:
            try:
                # Load sequences and prediction mask from this sample
                sequences, prediction_mask = self.processor.load_combined_sample_data(
                    sample_dir, self.sequence_length)
                
                if not sequences:
                    logger.warning(f"No sequences loaded from {sample_dir.name}")
                    continue
                
                # Store prediction mask (should be same for all samples)
                if self.prediction_mask is None:
                    self.prediction_mask = prediction_mask
                
                # Limit sequences per sample if specified
                if self.max_sequences_per_sample is not None:
                    sequences = sequences[:self.max_sequences_per_sample]
                
                # Add sequences with metadata
                for input_seq, target_seq, start_time, end_time in sequences:
                    self.all_sequences.append((input_seq, target_seq))
                    self.sequence_metadata.append({
                        'sample_id': sample_dir.name,
                        'start_time': start_time,
                        'end_time': end_time
                    })
                    
                logger.debug(f"Loaded {len(sequences)} sequences from {sample_dir.name}")
                
            except Exception as e:
                logger.error(f"Error loading sequences from {sample_dir.name}: {e}")
                continue
        
        # Create default mask if none was loaded
        if self.prediction_mask is None:
            logger.warning("No prediction mask loaded, creating default mask")
            self.prediction_mask = self.processor.create_prediction_mask()
    
    def _fit_scaler(self):
        """Fit normalization scaler on training data."""
        logger.info("Fitting scaler on training sequences...")
        
        # Collect all sequence data for scaler fitting
        all_data = []
        
        # Sample a subset for memory efficiency
        sample_size = min(100, len(self.all_sequences))
        sample_indices = np.linspace(0, len(self.all_sequences)-1, sample_size, dtype=int)
        
        for idx in sample_indices:
            input_seq, target_seq = self.all_sequences[idx]
            # Combine input and target for scaler fitting
            all_data.append(input_seq.reshape(-1, self.total_dims))
            all_data.append(target_seq.reshape(-1, self.total_dims))
        
        # Stack all data
        combined_data = np.vstack(all_data)  # [N_samples, 6713]
        
        # Fit scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.scaler.fit(combined_data)
        logger.info(f"Fitted {self.scaler_type} scaler on {combined_data.shape[0]} data points")
    
    def _normalize_data(self, data: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Normalize or denormalize data using fitted scaler.
        
        Args:
            data: Data to normalize/denormalize, shape [..., 6713]
            inverse: If True, perform inverse transformation
            
        Returns:
            Normalized/denormalized data
        """
        if self.scaler is None:
            return data
        
        # Handle different input shapes
        original_shape = data.shape
        
        if data.ndim == 3:  # [T, 6713] or similar
            data_2d = data.reshape(-1, data.shape[-1])
        elif data.ndim == 2:  # [T, 6713]
            data_2d = data
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        # Apply transformation
        if inverse:
            transformed = self.scaler.inverse_transform(data_2d)
        else:
            transformed = self.scaler.transform(data_2d)
        
        # Reshape back to original shape
        return transformed.reshape(original_shape)
    
    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.all_sequences)
    
    def __getitem__(self, idx: int) -> TensorDict:
        """
        Get a single sequence sample from the dataset.
        
        Args:
            idx: Sequence index
            
        Returns:
            TensorDict containing:
            - 'input': Input tensor [T, V=6712]
            - 'target': Target tensor [T, V=6712] 
            - 'mask': Prediction mask [V=6712]
            - 'metadata': Sample metadata dict
        """
        input_seq, target_seq = self.all_sequences[idx]
        metadata = self.sequence_metadata[idx]
        
        # Convert to numpy if needed
        if isinstance(input_seq, np.ndarray):
            input_data = input_seq.astype(np.float32)
        else:
            input_data = input_seq
            
        if isinstance(target_seq, np.ndarray):
            target_data = target_seq.astype(np.float32)
        else:
            target_data = target_seq
        
        # Normalize if enabled
        if self.normalize and self.scaler is not None:
            input_data = self._normalize_data(input_data)
            target_data = self._normalize_data(target_data)
        
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_data)    # [T, 6713]
        target_tensor = torch.FloatTensor(target_data)  # [T, 6713]
        mask_tensor = torch.from_numpy(self.prediction_mask).int()  # [6713]
        
        # Create TensorDict
        batch = TensorDict({
            'input': input_tensor,
            'target': target_tensor, 
            'mask': mask_tensor,
            'metadata': metadata
        }, batch_size=torch.Size([]))
        
        return batch
    
    def get_feature_info(self) -> Dict:
        """
        Get information about features and dimensions.
        
        Returns:
            Dictionary with feature information
        """
        return {
            'total_dims': self.total_dims,
            'boundary_dims': 538,
            'equipment_dims': 6174,
            'sequence_length': self.sequence_length,
            'prediction_mask_sum': int(np.sum(self.prediction_mask)) if self.prediction_mask is not None else 0,
            'equipment_breakdown': self.processor.equipment_info.copy()
        }
    
    def get_data_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'split': self.split,
            'num_sequences': len(self.all_sequences),
            'num_samples': len(self.sample_dirs),
            'total_dims': self.total_dims,
            'sequence_length': self.sequence_length,
            'normalized': self.normalize,
            'scaler_type': self.scaler_type
        }
    
    def save_scaler(self, save_dir: str):
        """Save fitted scaler to disk."""
        if not self.normalize or self.scaler is None:
            logger.warning("No scaler to save (normalize=False or scaler not fitted)")
            return
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / f'{self.scaler_type}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Scaler saved to {save_path}")
    
    def load_scaler(self, save_dir: str):
        """Load scaler from disk."""
        save_path = Path(save_dir)
        scaler_path = save_path / f'{self.scaler_type}_scaler.pkl'
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            logger.warning(f"Scaler file not found: {scaler_path}")
    
    def inverse_transform_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform normalized predictions back to original scale.
        
        Args:
            predictions: Normalized predictions tensor [..., 6712]
            
        Returns:
            Predictions in original scale
        """
        if not self.normalize or self.scaler is None:
            return predictions
        
        # Convert to numpy for scaler
        predictions_np = predictions.detach().cpu().numpy()
        
        # Apply inverse transformation
        predictions_original = self._normalize_data(predictions_np, inverse=True)
        
        # Convert back to tensor
        return torch.from_numpy(predictions_original).to(predictions.device)
    
    def get_sample_by_id(self, sample_id: str) -> List[int]:
        """
        Get all sequence indices for a specific sample ID.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            List of sequence indices belonging to this sample
        """
        indices = []
        for i, metadata in enumerate(self.sequence_metadata):
            if metadata['sample_id'] == sample_id:
                indices.append(i)
        return indices


def collate_fn(batch_list: List[TensorDict]) -> Dict[str, any]:
    """
    Custom collate function for TensorDict batches.
    
    Args:
        batch_list: List of TensorDict samples
        
    Returns:
        Regular dictionary with batched tensors
        - input: [B, T, V] 
        - target: [B, T, V]
        - prediction_mask: [B, V] - 预测变量mask (0=boundary, 1=equipment)
        - attention_mask: [B, T, V] - 因果注意力mask
    """
    # Stack tensor data
    inputs = torch.stack([item['input'] for item in batch_list])      # [B, T, V]
    targets = torch.stack([item['target'] for item in batch_list])    # [B, T, V]
    prediction_masks = torch.stack([item['mask'] for item in batch_list])  # [B, V]
    
    # 生成因果注意力mask [B, T, V]
    B, T, V = inputs.shape
    
    # 创建因果注意力mask
    attention_mask = torch.zeros((B, T, V), dtype=torch.float32)
    
    # 获取boundary和equipment的分界点 (假设前538为boundary，后面为equipment)
    boundary_dims = 538
    
    for b in range(B):
        for t in range(T):
            # Boundary变量：只和自己时间步交互，但不参与预测
            attention_mask[b, t, :boundary_dims] = 0.0
            
            # Equipment变量：因果mask，可以看到当前及之前时间步
            attention_mask[b, t, boundary_dims:] = 1.0
    
    # Collect metadata as list
    metadata = [item['metadata'] for item in batch_list]
    
    return {
        'input': inputs,                    # [B, T, V]
        'target': targets,                  # [B, T, V]  
        'prediction_mask': prediction_masks, # [B, V] - 哪些变量需要预测
        'attention_mask': attention_mask,   # [B, T, V] - decoder attention mask
        'metadata': metadata
    }