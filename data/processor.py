import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from scipy import interpolate

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Raw data processor for gas pipeline network fluid dynamics data.
    
    Handles:
    - Loading CSV files from train/test directories
    - Parsing boundary conditions and equipment outputs
    - Time series alignment (30-min boundary → 1-min predictions)
    - Data cleaning and validation
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize DataProcessor.
        
        Args:
            data_dir: Path to data directory containing dataset folder
        """
        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / "dataset"
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"
        
        # Equipment file types and their expected column counts (excluding TIME)
        self.equipment_info = {
            'B': 2058,     # Ball valves
            'C': 161,      # Compressors  
            'H': 192,      # Pipeline segments
            'N': 1716,     # Nodes
            'P': 1610,     # Pipelines
            'R': 50,       # Control valves
            'T&E': 387     # Gas sources and distribution points
        }
        
        # Total prediction dimensions: 6,174
        self.total_prediction_dims = sum(self.equipment_info.values())
        
        # Boundary conditions: 539 total columns (including TIME), 538 data columns  
        self.boundary_dims = 538  # Non-TIME columns only
        
        self._validate_directories()
    
    def _validate_directories(self) -> None:
        """Validate that required directories exist."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
            
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")
    
    def get_sample_directories(self, split: str = 'train') -> List[Path]:
        """
        Get list of sample directories for given split.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            List of sample directory paths
        """
        if split == 'train':
            base_dir = self.train_dir
        elif split == 'test':
            base_dir = self.test_dir
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
        
        sample_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        sample_dirs.sort()  # Sort by name for consistent ordering
        
        logger.info(f"Found {len(sample_dirs)} {split} samples")
        return sample_dirs
    
    def load_boundary_data(self, sample_dir: Path) -> pd.DataFrame:
        """
        Load boundary conditions from Boundary.csv.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            DataFrame with boundary conditions (TIME + 539 boundary parameters)
        """
        boundary_file = sample_dir / "Boundary.csv"
        if not boundary_file.exists():
            raise FileNotFoundError(f"Boundary.csv not found in {sample_dir}")
        
        df = pd.read_csv(boundary_file)
        
        # Validate dimensions (including TIME column)
        expected_cols = self.boundary_dims + 1  # +1 for TIME column
        if df.shape[1] != expected_cols:
            logger.warning(f"Expected {expected_cols} columns in {boundary_file}, got {df.shape[1]}")
        
        # Convert TIME to datetime if not already  
        if 'TIME' in df.columns:
            try:
                df['TIME'] = pd.to_datetime(df['TIME'], format='%Y/%m/%d %H:%M')
            except ValueError:
                # Try alternative format or let pandas infer
                df['TIME'] = pd.to_datetime(df['TIME'])
        
        logger.debug(f"Loaded boundary data: {df.shape} from {sample_dir.name}")
        return df
    
    def load_equipment_data(self, sample_dir: Path, equipment_type: str) -> Optional[pd.DataFrame]:
        """
        Load equipment prediction data from specific CSV file.
        
        Args:
            sample_dir: Path to sample directory
            equipment_type: Equipment type ('B', 'C', 'H', 'N', 'P', 'R', 'T&E')
            
        Returns:
            DataFrame with equipment predictions or None if file doesn't exist (test data)
        """
        csv_file = sample_dir / f"{equipment_type}.csv"
        
        # Test data may not have equipment files
        if not csv_file.exists():
            return None
        
        df = pd.read_csv(csv_file)
        
        # Validate dimensions
        expected_cols = self.equipment_info[equipment_type] + 1  # +1 for TIME
        if df.shape[1] != expected_cols:
            logger.warning(f"Expected {expected_cols} columns in {csv_file}, got {df.shape[1]}")
        
        # Convert TIME to datetime
        if 'TIME' in df.columns:
            try:
                df['TIME'] = pd.to_datetime(df['TIME'], format='%Y/%m/%d %H:%M')
            except ValueError:
                # Try alternative format or let pandas infer
                df['TIME'] = pd.to_datetime(df['TIME'])
        
        logger.debug(f"Loaded {equipment_type} data: {df.shape} from {sample_dir.name}")
        return df
    
    def load_all_equipment_data(self, sample_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load all equipment data for a sample.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            Dictionary mapping equipment type to DataFrame
        """
        equipment_data = {}
        
        for equipment_type in self.equipment_info.keys():
            data = self.load_equipment_data(sample_dir, equipment_type)
            if data is not None:
                equipment_data[equipment_type] = data
        
        return equipment_data
    
    def combine_equipment_data(self, equipment_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all equipment data into single DataFrame.
        
        Args:
            equipment_data: Dictionary of equipment DataFrames
            
        Returns:
            Combined DataFrame with all equipment predictions
        """
        if not equipment_data:
            return pd.DataFrame()
        
        # Get TIME from first available equipment data
        time_col = None
        for df in equipment_data.values():
            if 'TIME' in df.columns:
                time_col = df['TIME'].copy()
                break
        
        # Combine all non-TIME columns
        combined_data = {'TIME': time_col} if time_col is not None else {}
        
        for equipment_type, df in equipment_data.items():
            # Add all columns except TIME
            for col in df.columns:
                if col != 'TIME':
                    combined_data[col] = df[col]
        
        combined_df = pd.DataFrame(combined_data)
        
        # Validate total dimensions
        expected_cols = self.total_prediction_dims + 1  # +1 for TIME
        if combined_df.shape[1] != expected_cols:
            logger.warning(f"Expected {expected_cols} total columns, got {combined_df.shape[1]}")
        
        return combined_df
    
    def load_sample_data(self, sample_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load complete data for a single sample.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            Tuple of (boundary_data, equipment_data)
            equipment_data is None for test samples
        """
        # Load boundary conditions
        boundary_data = self.load_boundary_data(sample_dir)
        
        # Load equipment data (may be None for test data)
        equipment_dict = self.load_all_equipment_data(sample_dir)
        equipment_data = self.combine_equipment_data(equipment_dict) if equipment_dict else None
        
        return boundary_data, equipment_data
    
    def validate_data_consistency(self, boundary_data: pd.DataFrame, 
                                 equipment_data: Optional[pd.DataFrame]) -> bool:
        """
        Validate consistency between boundary and equipment data.
        
        Args:
            boundary_data: Boundary conditions DataFrame
            equipment_data: Equipment predictions DataFrame (may be None)
            
        Returns:
            True if data is consistent
        """
        if equipment_data is None:
            return True  # Test data doesn't have equipment data
        
        # Check time series length
        if len(boundary_data) != len(equipment_data):
            logger.error(f"Length mismatch: boundary={len(boundary_data)}, equipment={len(equipment_data)}")
            return False
        
        # Check time alignment if both have TIME columns
        if 'TIME' in boundary_data.columns and 'TIME' in equipment_data.columns:
            if not boundary_data['TIME'].equals(equipment_data['TIME']):
                logger.error("TIME columns are not aligned between boundary and equipment data")
                return False
        
        # Expected: 1440 time steps (24 hours * 60 minutes)
        expected_timesteps = 1440
        if len(boundary_data) != expected_timesteps:
            logger.warning(f"Expected {expected_timesteps} timesteps, got {len(boundary_data)}")
        
        return True
    
    def get_data_statistics(self, split: str = 'train') -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            Dictionary with dataset statistics
        """
        sample_dirs = self.get_sample_directories(split)
        
        stats = {
            'split': split,
            'num_samples': len(sample_dirs),
            'boundary_dims': self.boundary_dims,
            'total_prediction_dims': self.total_prediction_dims,
            'equipment_dims': self.equipment_info.copy(),
            'expected_timesteps': 1440,
            'samples': []
        }
        
        # Check first few samples for validation
        for i, sample_dir in enumerate(sample_dirs[:3]):
            try:
                boundary_data, equipment_data = self.load_sample_data(sample_dir)
                
                sample_stats = {
                    'name': sample_dir.name,
                    'boundary_shape': boundary_data.shape,
                    'equipment_shape': equipment_data.shape if equipment_data is not None else None,
                    'valid': self.validate_data_consistency(boundary_data, equipment_data)
                }
                
                stats['samples'].append(sample_stats)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample_dir.name}: {e}")
                stats['samples'].append({
                    'name': sample_dir.name,
                    'error': str(e)
                })
        
        return stats

    def interpolate_boundary_to_1min(self, boundary_df: pd.DataFrame) -> pd.DataFrame:
        """
        将30分钟间隔的boundary数据插值到1分钟间隔。
        
        Args:
            boundary_df: Boundary数据, shape [N, 539+1], TIME列 + 539个boundary参数
            
        Returns:
            插值后的数据, shape [M, 539+1], M为1分钟间隔的时间点数量
        """
        if 'TIME' not in boundary_df.columns:
            raise ValueError("Boundary data must contain TIME column")
        
        df = boundary_df.copy()
        
        # 确保TIME列是datetime格式
        df['TIME'] = pd.to_datetime(df['TIME'])
        
        # 创建1分钟间隔的时间索引
        start_time = df['TIME'].iloc[0]
        end_time = df['TIME'].iloc[-1] 
        minute_range = pd.date_range(start_time, end_time, freq='1min')
        
        # 设置TIME为索引进行插值
        df_indexed = df.set_index('TIME')
        
        # 线性插值到1分钟间隔
        interpolated = df_indexed.reindex(minute_range).interpolate(method='linear')
        
        # 重置索引，TIME重新作为列
        interpolated_df = interpolated.reset_index()
        interpolated_df.rename(columns={'index': 'TIME'}, inplace=True)
        
        logger.debug(f"Interpolated boundary data from {len(df)} to {len(interpolated_df)} time points")
        return interpolated_df
    
    def combine_all_data(self, boundary_df: pd.DataFrame, equipment_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并boundary和所有equipment数据为单一DataFrame。
        
        Args:
            boundary_df: 插值后的boundary数据, shape [T, 539+1]
            equipment_dict: 各设备类型的数据字典
            
        Returns:
            合并后的数据, shape [T, 6713+1], TIME列 + 6713个变量 (539 boundary + 6174 equipment)
        """
        if not equipment_dict:
            logger.warning("No equipment data provided")
            return boundary_df
        
        # 获取TIME列作为基础
        time_col = boundary_df['TIME'].copy()
        
        # 创建合并数据字典
        combined_data = {'TIME': time_col}
        
        # 添加boundary数据 (排除TIME列)
        boundary_numeric = boundary_df.drop(columns=['TIME'])
        for col in boundary_numeric.columns:
            combined_data[f'boundary_{col}'] = boundary_numeric[col]
        
        # 按预定义顺序添加equipment数据
        equipment_order = ['B', 'C', 'H', 'N', 'P', 'R', 'T&E']
        
        for equipment_type in equipment_order:
            if equipment_type in equipment_dict:
                eq_df = equipment_dict[equipment_type]
                # 添加除TIME外的所有列
                eq_numeric = eq_df.drop(columns=['TIME'], errors='ignore')
                for col in eq_numeric.columns:
                    combined_data[col] = eq_numeric[col]
        
        combined_df = pd.DataFrame(combined_data)
        
        # 验证维度
        expected_cols = 1 + self.boundary_dims + self.total_prediction_dims  # TIME + 538 + 6174 = 6713
        if combined_df.shape[1] != expected_cols:
            logger.warning(f"Expected {expected_cols} total columns, got {combined_df.shape[1]}")
        
        logger.debug(f"Combined data shape: {combined_df.shape}")
        return combined_df
    
    def create_prediction_mask(self) -> np.ndarray:
        """
        创建预测mask数组: boundary变量=0(不预测), equipment变量=1(需要预测)。
        
        Returns:
            mask数组, shape [6712], boundary部分为0，equipment部分为1
        """
        boundary_mask = np.zeros(self.boundary_dims, dtype=np.int32)  # 538个0
        equipment_mask = np.ones(self.total_prediction_dims, dtype=np.int32)  # 6174个1
        
        mask = np.concatenate([boundary_mask, equipment_mask])
        
        logger.debug(f"Created prediction mask: {len(mask)} total, {np.sum(mask)} predictable")
        return mask
    
    def create_causal_attention_mask(self, sequence_length: int) -> np.ndarray:
        """
        创建因果注意力mask用于decoder。
        
        Args:
            sequence_length: 时间步长 T
            
        Returns:
            attention_mask: shape [T, V], 每个时间步的变量attention mask
                - Boundary变量 (前538列): 只和自己时间步交互 (对角mask)  
                - Equipment变量 (后6174列): 因果mask，可以看到当前及之前时间步
        """
        T = sequence_length
        V = self.boundary_dims + self.total_prediction_dims  # 6712
        
        # 创建 [T, V] 的mask
        attention_mask = np.zeros((T, V), dtype=np.float32)
        
        # Boundary变量 (前538列): 对角mask，只和自己时间步交互
        for t in range(T):
            attention_mask[t, :self.boundary_dims] = 0.0  # Boundary不参与预测
        
        # Equipment变量 (后6174列): 因果mask
        for t in range(T):
            # 当前时间步可以看到当前及之前所有时间步的设备变量
            attention_mask[t, self.boundary_dims:] = 1.0  # 设备变量参与预测
        
        logger.debug(f"Created causal attention mask: [{T}, {V}]")
        return attention_mask
    
    def create_sequences(self, full_data: pd.DataFrame, sequence_length: int = 3) -> List[Tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]]:
        """
        从完整数据创建时序序列对 (input, target)。
        
        Args:
            full_data: 合并后的完整数据, shape [T, 6713] (TIME + 6712个变量)
            sequence_length: 序列长度，默认3分钟
            
        Returns:
            序列列表，每个元素为 (input_seq, target_seq, start_time, end_time)
            - input_seq: shape [sequence_length, 6712]
            - target_seq: shape [sequence_length, 6712] 
            - start_time, end_time: 序列的时间范围
        """
        if len(full_data) < sequence_length + 1:
            raise ValueError(f"Data length {len(full_data)} insufficient for sequence_length {sequence_length}")
        
        sequences = []
        
        # 提取数值数据 (排除TIME列)
        numeric_data = full_data.drop(columns=['TIME']).values  # [T, 6712]
        time_data = full_data['TIME'].values
        
        # 滑动窗口生成序列对
        for i in range(len(numeric_data) - sequence_length):
            input_seq = numeric_data[i:i+sequence_length]           # [T, 6712]
            target_seq = numeric_data[i+1:i+sequence_length+1]      # [T, 6712] - 向前滑动1分钟
            
            start_time = time_data[i]
            end_time = time_data[i+sequence_length-1]
            
            sequences.append((input_seq, target_seq, start_time, end_time))
        
        logger.debug(f"Created {len(sequences)} sequences of length {sequence_length}")
        return sequences
    
    def load_combined_sample_data(self, sample_dir: Path, sequence_length: int = 3) -> Tuple[List[Tuple], Optional[np.ndarray]]:
        """
        加载单个样本的完整合并数据并创建序列。
        
        Args:
            sample_dir: 样本目录路径
            sequence_length: 时序序列长度
            
        Returns:
            Tuple of (sequences, prediction_mask)
            - sequences: List of (input, target, start_time, end_time) tuples
            - prediction_mask: shape [6712] 预测mask数组
        """
        try:
            # 加载boundary数据
            boundary_data = self.load_boundary_data(sample_dir)
            
            # 插值boundary数据到1分钟间隔
            boundary_1min = self.interpolate_boundary_to_1min(boundary_data)
            
            # 加载所有equipment数据
            equipment_dict = self.load_all_equipment_data(sample_dir)
            
            # 验证数据存在性
            if not equipment_dict:
                # 测试数据可能没有equipment数据
                logger.info(f"No equipment data found for {sample_dir.name} (likely test data)")
                return [], None
            
            # 合并所有数据
            combined_data = self.combine_all_data(boundary_1min, equipment_dict)
            
            # 创建时序序列
            sequences = self.create_sequences(combined_data, sequence_length)
            
            # 创建预测mask
            prediction_mask = self.create_prediction_mask()
            
            logger.debug(f"Loaded {len(sequences)} sequences from {sample_dir.name}")
            return sequences, prediction_mask
            
        except Exception as e:
            logger.error(f"Error loading combined data from {sample_dir}: {e}")
            return [], None

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from TIME column.
        
        Args:
            df: DataFrame with TIME column
            
        Returns:
            DataFrame with additional time features
        """
        if 'TIME' not in df.columns:
            return df
        
        df = df.copy()
        
        # Extract time features
        df['hour'] = df['TIME'].dt.hour
        df['minute'] = df['TIME'].dt.minute
        df['day_of_week'] = df['TIME'].dt.dayofweek
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        
        return df