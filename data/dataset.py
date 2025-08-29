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
from .normalizer import DataNormalizer

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
                 cache_data: bool = True,
                 max_samples: Optional[int] = None,
                 max_sequences_per_sample: Optional[int] = None):
        """
        Initialize FluidDataset.
        
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            sequence_length: Length of time series sequences (default: 3 for 3 minutes)
            cache_data: Whether to cache loaded data in memory
            max_samples: Maximum number of samples to load (for testing)
            max_sequences_per_sample: Maximum sequences per sample (for memory control)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.cache_data = cache_data
        self.max_samples = max_samples
        self.max_sequences_per_sample = max_sequences_per_sample
        
        # Initialize data processor
        self.processor = DataProcessor(str(self.data_dir))
        
        # Data dimensions
        self.total_dims = 6712  # 538 boundary + 6174 equipment
        
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
        
        # Convert to numpy if needed and ensure float32 type
        if isinstance(input_seq, np.ndarray):
            input_data = input_seq.astype(np.float32)
        else:
            input_data = np.array(input_seq, dtype=np.float32)
            
        if isinstance(target_seq, np.ndarray):
            target_data = target_seq.astype(np.float32)
        else:
            target_data = np.array(target_seq, dtype=np.float32)
        
        # Convert to tensors - normalization will be handled in collate_fn
        input_tensor = torch.FloatTensor(input_data)    # [T, 6712]
        target_tensor = torch.FloatTensor(target_data)  # [T, 6712]
        mask_tensor = torch.from_numpy(self.prediction_mask).int()  # [6712]
        
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
            'sequence_length': self.sequence_length
        }
    
    
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


def collate_fn(batch_list: List[TensorDict], 
               normalizer: Optional[DataNormalizer] = None,
               apply_normalization: bool = True) -> Dict[str, any]:
    """
    Custom collate function for TensorDict batches with optional normalization.
    
    Args:
        batch_list: List of TensorDict samples
        normalizer: Optional DataNormalizer instance for data normalization
        apply_normalization: Whether to apply normalization (False for visualization)
        
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
    
    # Apply normalization if requested and normalizer is provided
    if apply_normalization and normalizer is not None and normalizer.fitted:
        inputs = normalizer.transform(inputs)
        targets = normalizer.transform(targets)
        logger.debug(f"Applied {normalizer.method} normalization to batch")
    
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
        'metadata': metadata,
        'normalized': apply_normalization and normalizer is not None  # 标记是否已归一化
    }


def create_collate_fn(normalizer: Optional[DataNormalizer] = None,
                     apply_normalization: bool = True):
    """
    创建带有归一化参数的collate函数。
    
    Args:
        normalizer: 数据归一化器
        apply_normalization: 是否应用归一化
        
    Returns:
        配置好的collate函数
    """
    def _collate_fn(batch_list: List[TensorDict]) -> Dict[str, any]:
        return collate_fn(batch_list, normalizer, apply_normalization)
    
    return _collate_fn


def load_normalizer(data_dir: str, method: str = 'standard') -> Optional[DataNormalizer]:
    """
    加载预计算的归一化器。
    
    Args:
        data_dir: 数据目录路径
        method: 归一化方法
        
    Returns:
        已加载的归一化器，如果加载失败返回None
    """
    try:
        normalizer = DataNormalizer(data_dir, method=method)
        if normalizer.load_stats():
            logger.info(f"Successfully loaded {method} normalizer from {data_dir}")
            return normalizer
        else:
            logger.warning(f"Failed to load {method} normalizer from {data_dir}")
            return None
    except Exception as e:
        logger.error(f"Error loading normalizer: {e}")
        return None


def create_dataloader_with_normalization(dataset: FluidDataset,
                                        batch_size: int = 32,
                                        shuffle: bool = False,
                                        num_workers: int = 0,
                                        normalizer_method: str = 'standard',
                                        apply_normalization: bool = True) -> torch.utils.data.DataLoader:
    """
    创建带有归一化功能的DataLoader。
    
    Args:
        dataset: FluidDataset实例
        batch_size: 批次大小
        shuffle: 是否随机打乱
        num_workers: 工作进程数
        normalizer_method: 归一化方法
        apply_normalization: 是否应用归一化（可视化时设为False）
        
    Returns:
        配置好的DataLoader
    """
    # 加载归一化器
    normalizer = None
    if apply_normalization:
        normalizer = load_normalizer(str(dataset.data_dir), normalizer_method)
        if normalizer is None:
            logger.warning("Normalizer not found. Consider running compute_normalization_stats.py first")
    
    # 创建collate函数
    collate_func = create_collate_fn(normalizer, apply_normalization)
    
    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_func,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created DataLoader: batch_size={batch_size}, normalization={apply_normalization}, "
               f"method={normalizer_method if normalizer else 'none'}")
    
    return dataloader