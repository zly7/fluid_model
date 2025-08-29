import numpy as np
import torch
from pathlib import Path
import pickle
import logging
from typing import Dict, Optional, Union, Tuple, List
import warnings

logger = logging.getLogger(__name__)

class DataNormalizer:
    """
    独立的数据归一化管理器，用于流体动力学数据的标准化处理。
    
    支持特性：
    - 多种归一化方法：standard (z-score), minmax, robust
    - 分维度统计：对6712维数据的每一维分别计算统计量
    - 持久化存储：保存和加载均值、标准差等统计量
    - 批量处理：高效处理大批量数据
    - 异常处理：处理NaN、常数列等边界情况
    """
    
    def __init__(self, 
                 data_dir: str,
                 method: str = 'standard',
                 boundary_dims: int = 538,
                 equipment_dims: int = 6174):
        """
        初始化数据归一化器。
        
        Args:
            data_dir: 数据根目录路径
            method: 归一化方法 ('standard', 'minmax', 'robust')
            boundary_dims: boundary变量维度数 (默认538)
            equipment_dims: equipment变量维度数 (默认6174)
        """
        self.data_dir = Path(data_dir)
        self.method = method.lower()
        self.boundary_dims = boundary_dims
        self.equipment_dims = equipment_dims
        self.total_dims = boundary_dims + equipment_dims  # 6712
        
        # 验证归一化方法
        if self.method not in ['standard', 'minmax', 'robust']:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # 统计量存储
        self.fitted = False
        self.mean_ = None      # shape [6712]
        self.std_ = None       # shape [6712]  
        self.min_ = None       # shape [6712] (for minmax)
        self.max_ = None       # shape [6712] (for minmax)
        self.q25_ = None       # shape [6712] (for robust)
        self.q75_ = None       # shape [6712] (for robust)
        self.median_ = None    # shape [6712] (for robust)
        
        # 创建存储目录
        self.stats_dir = self.data_dir / "normalization_stats"
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataNormalizer initialized: method={method}, dims={self.total_dims}")
    
    def fit(self, data_samples: List[np.ndarray], batch_size: int = 10000) -> None:
        """
        基于训练数据样本计算归一化统计量（使用流式计算避免内存溢出）。
        
        Args:
            data_samples: 数据样本列表，每个样本 shape [T, 6712]
            batch_size: 批处理大小，控制内存使用
        """
        logger.info(f"Fitting normalizer on {len(data_samples)} samples using streaming computation...")
        
        if not data_samples:
            raise ValueError("No data samples provided for fitting")
        
        # 计算总数据点数
        total_points = sum(sample.shape[0] for sample in data_samples)
        logger.info(f"Total data points: {total_points:,}")
        
        # 使用流式计算避免内存溢出
        if self.method == 'standard':
            self._fit_standard_streaming(data_samples, batch_size)
        elif self.method == 'minmax':
            self._fit_minmax_streaming(data_samples, batch_size)
        elif self.method == 'robust':
            self._fit_robust_streaming(data_samples, batch_size)
        
        self.fitted = True
        logger.info(f"Normalizer fitted using {self.method} method with streaming computation")
    
    def _fit_standard(self, data: np.ndarray) -> None:
        """计算标准化统计量 (z-score)。"""
        self.mean_ = np.mean(data, axis=0, dtype=np.float64)  # [6712]
        self.std_ = np.std(data, axis=0, dtype=np.float64, ddof=1)   # [6712]
        
        # 处理标准差为0的情况（常数列）
        zero_std_mask = self.std_ == 0
        if np.any(zero_std_mask):
            logger.warning(f"Found {np.sum(zero_std_mask)} constant columns, setting std=1")
            self.std_[zero_std_mask] = 1.0
        
        logger.debug(f"Standard stats - mean: [{self.mean_.min():.4f}, {self.mean_.max():.4f}], "
                    f"std: [{self.std_.min():.4f}, {self.std_.max():.4f}]")
    
    def _fit_minmax(self, data: np.ndarray) -> None:
        """计算最大最小值归一化统计量。"""
        self.min_ = np.min(data, axis=0, dtype=np.float64)  # [6712]
        self.max_ = np.max(data, axis=0, dtype=np.float64)  # [6712]
        
        # 处理最大值等于最小值的情况（常数列）
        const_mask = self.min_ == self.max_
        if np.any(const_mask):
            logger.warning(f"Found {np.sum(const_mask)} constant columns, setting range=1")
            self.max_[const_mask] = self.min_[const_mask] + 1.0
        
        logger.debug(f"MinMax stats - min: [{self.min_.min():.4f}, {self.min_.max():.4f}], "
                    f"max: [{self.max_.min():.4f}, {self.max_.max():.4f}]")
    
    def _fit_robust(self, data: np.ndarray) -> None:
        """计算鲁棒性归一化统计量 (基于分位数)。"""
        self.median_ = np.median(data, axis=0)  # [6712]
        self.q25_ = np.percentile(data, 25, axis=0)  # [6712]
        self.q75_ = np.percentile(data, 75, axis=0)  # [6712]
        
        # IQR = Q75 - Q25
        iqr = self.q75_ - self.q25_
        
        # 处理IQR为0的情况
        zero_iqr_mask = iqr == 0
        if np.any(zero_iqr_mask):
            logger.warning(f"Found {np.sum(zero_iqr_mask)} columns with zero IQR, setting IQR=1")
            self.q75_[zero_iqr_mask] = self.q25_[zero_iqr_mask] + 1.0
        
        logger.debug(f"Robust stats - median: [{self.median_.min():.4f}, {self.median_.max():.4f}], "
                    f"IQR: [{iqr.min():.4f}, {iqr.max():.4f}]")
    
    def _fit_standard_streaming(self, data_samples: List[np.ndarray], batch_size: int) -> None:
        """使用流式计算方式计算标准化统计量，避免内存溢出。"""
        logger.info("Computing standard normalization stats using streaming method...")
        
        # 初始化累计统计量
        n_total = 0
        sum_x = np.zeros(self.total_dims, dtype=np.float64)
        sum_x2 = np.zeros(self.total_dims, dtype=np.float64)
        
        # 流式计算均值和方差
        processed = 0
        for sample in data_samples:
            # 验证单个样本
            if sample.shape[1] != self.total_dims:
                raise ValueError(f"Expected {self.total_dims} features, got {sample.shape[1]}")
            
            # 处理NaN和inf
            sample_clean = sample.copy()
            nan_mask = np.isnan(sample_clean)
            if np.any(nan_mask):
                sample_clean[nan_mask] = 0.0
            inf_mask = np.isinf(sample_clean) 
            if np.any(inf_mask):
                sample_clean[inf_mask] = np.clip(sample_clean[inf_mask], -1e6, 1e6)
            
            # 分批处理大样本
            n_points = sample_clean.shape[0]
            for i in range(0, n_points, batch_size):
                end_idx = min(i + batch_size, n_points)
                batch = sample_clean[i:end_idx]
                
                # 累计统计
                n_batch = batch.shape[0]
                sum_x += np.sum(batch, axis=0)
                sum_x2 += np.sum(batch ** 2, axis=0)
                n_total += n_batch
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {processed}/{len(data_samples)} samples, total points: {n_total:,}")
        
        # 计算最终统计量
        self.mean_ = sum_x / n_total
        variance = (sum_x2 / n_total) - (self.mean_ ** 2)
        self.std_ = np.sqrt(variance)
        
        # 处理标准差为0的情况（常数列）
        zero_std_mask = self.std_ == 0
        if np.any(zero_std_mask):
            logger.warning(f"Found {np.sum(zero_std_mask)} constant columns, setting std=1")
            self.std_[zero_std_mask] = 1.0
        
        logger.info(f"Standard stats computed on {n_total:,} points - "
                   f"mean: [{self.mean_.min():.4f}, {self.mean_.max():.4f}], "
                   f"std: [{self.std_.min():.4f}, {self.std_.max():.4f}]")
    
    def _fit_minmax_streaming(self, data_samples: List[np.ndarray], batch_size: int) -> None:
        """使用流式计算方式计算最大最小值统计量。"""
        logger.info("Computing minmax normalization stats using streaming method...")
        
        # 初始化
        self.min_ = np.full(self.total_dims, np.inf, dtype=np.float64)
        self.max_ = np.full(self.total_dims, -np.inf, dtype=np.float64)
        
        processed = 0
        total_points = 0
        
        for sample in data_samples:
            if sample.shape[1] != self.total_dims:
                raise ValueError(f"Expected {self.total_dims} features, got {sample.shape[1]}")
            
            # 处理异常值
            sample_clean = sample.copy()
            nan_mask = np.isnan(sample_clean)
            if np.any(nan_mask):
                sample_clean[nan_mask] = 0.0
            inf_mask = np.isinf(sample_clean)
            if np.any(inf_mask):
                sample_clean[inf_mask] = np.clip(sample_clean[inf_mask], -1e6, 1e6)
            
            # 分批处理
            n_points = sample_clean.shape[0]
            for i in range(0, n_points, batch_size):
                end_idx = min(i + batch_size, n_points)
                batch = sample_clean[i:end_idx]
                
                # 更新最大最小值
                batch_min = np.min(batch, axis=0)
                batch_max = np.max(batch, axis=0)
                
                self.min_ = np.minimum(self.min_, batch_min)
                self.max_ = np.maximum(self.max_, batch_max)
                
                total_points += batch.shape[0]
            
            processed += 1
            if processed % 10 == 0:
                logger.info(f"Processed {processed}/{len(data_samples)} samples")
        
        # 处理最大值等于最小值的情况
        const_mask = self.min_ == self.max_
        if np.any(const_mask):
            logger.warning(f"Found {np.sum(const_mask)} constant columns, setting range=1")
            self.max_[const_mask] = self.min_[const_mask] + 1.0
        
        logger.info(f"MinMax stats computed on {total_points:,} points - "
                   f"min: [{self.min_.min():.4f}, {self.min_.max():.4f}], "
                   f"max: [{self.max_.min():.4f}, {self.max_.max():.4f}]")
    
    def _fit_robust_streaming(self, data_samples: List[np.ndarray], batch_size: int) -> None:
        """使用流式计算方式计算鲁棒性统计量（基于分位数）。"""
        logger.info("Computing robust normalization stats using streaming method...")
        
        # 对于鲁棒性统计量，我们需要收集所有数据来计算分位数
        # 但是我们可以分批收集，避免一次性加载所有数据
        logger.info("Collecting data for quantile computation...")
        
        all_data_list = []
        total_points = 0
        processed = 0
        
        for sample in data_samples:
            if sample.shape[1] != self.total_dims:
                raise ValueError(f"Expected {self.total_dims} features, got {sample.shape[1]}")
            
            # 处理异常值
            sample_clean = sample.copy()
            nan_mask = np.isnan(sample_clean)
            if np.any(nan_mask):
                sample_clean[nan_mask] = 0.0
            inf_mask = np.isinf(sample_clean)
            if np.any(inf_mask):
                sample_clean[inf_mask] = np.clip(sample_clean[inf_mask], -1e6, 1e6)
            
            all_data_list.append(sample_clean)
            total_points += sample_clean.shape[0]
            processed += 1
            
            if processed % 10 == 0:
                logger.info(f"Collected {processed}/{len(data_samples)} samples, {total_points:,} points")
        
        # 如果数据量太大，进行采样
        if total_points > 1000000:  # 超过100万个点时进行采样
            logger.warning(f"Data too large ({total_points:,} points), sampling 1M points for robust stats")
            
            # 计算采样比例
            sample_ratio = 1000000 / total_points
            sampled_data_list = []
            
            for sample in all_data_list:
                n_sample = max(1, int(sample.shape[0] * sample_ratio))
                indices = np.random.choice(sample.shape[0], n_sample, replace=False)
                sampled_data_list.append(sample[indices])
            
            combined_data = np.vstack(sampled_data_list)
        else:
            combined_data = np.vstack(all_data_list)
        
        logger.info(f"Computing quantiles on {combined_data.shape[0]:,} points...")
        
        # 计算分位数
        self.median_ = np.median(combined_data, axis=0)
        self.q25_ = np.percentile(combined_data, 25, axis=0)
        self.q75_ = np.percentile(combined_data, 75, axis=0)
        
        # 处理IQR为0的情况
        iqr = self.q75_ - self.q25_
        zero_iqr_mask = iqr == 0
        if np.any(zero_iqr_mask):
            logger.warning(f"Found {np.sum(zero_iqr_mask)} columns with zero IQR, setting IQR=1")
            self.q75_[zero_iqr_mask] = self.q25_[zero_iqr_mask] + 1.0
        
        logger.info(f"Robust stats computed - "
                   f"median: [{self.median_.min():.4f}, {self.median_.max():.4f}], "
                   f"IQR: [{iqr.min():.4f}, {iqr.max():.4f}]")
    
    def _validate_data(self, data: np.ndarray) -> None:
        """验证输入数据的有效性。"""
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data, got shape {data.shape}")
        
        if data.shape[1] != self.total_dims:
            raise ValueError(f"Expected {self.total_dims} features, got {data.shape[1]}")
        
        # 检查NaN值
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            logger.warning(f"Found {nan_count} NaN values in data, replacing with 0")
            data[nan_mask] = 0.0
        
        # 检查无穷值
        inf_mask = np.isinf(data)
        if np.any(inf_mask):
            inf_count = np.sum(inf_mask)
            logger.warning(f"Found {inf_count} infinite values in data, clipping")
            data[inf_mask] = np.clip(data[inf_mask], -1e6, 1e6)
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        对数据进行归一化变换。
        
        Args:
            data: 输入数据，shape [..., 6712]
            
        Returns:
            归一化后的数据，保持输入类型和形状
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        # 处理torch.Tensor
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
        
        # 保存原始形状
        original_shape = data_np.shape
        
        # 重塑为2D用于归一化
        if data_np.ndim > 2:
            data_2d = data_np.reshape(-1, self.total_dims)
        elif data_np.ndim == 2:
            data_2d = data_np
        elif data_np.ndim == 1:
            data_2d = data_np.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported data shape: {original_shape}")
        
        # 验证维度
        if data_2d.shape[1] != self.total_dims:
            raise ValueError(f"Expected {self.total_dims} features, got {data_2d.shape[1]}")
        
        # 应用归一化
        if self.method == 'standard':
            normalized = (data_2d - self.mean_) / self.std_
        elif self.method == 'minmax':
            normalized = (data_2d - self.min_) / (self.max_ - self.min_)
        elif self.method == 'robust':
            iqr = self.q75_ - self.q25_
            normalized = (data_2d - self.median_) / iqr
        
        # 重塑回原始形状
        normalized = normalized.reshape(original_shape)
        
        # 转换回torch.Tensor
        if is_tensor:
            normalized = torch.from_numpy(normalized.astype(np.float32)).to(device)
        
        return normalized
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        对归一化数据进行反变换，恢复原始尺度。
        
        Args:
            data: 归一化后的数据，shape [..., 6712]
            
        Returns:
            恢复原始尺度的数据
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        # 处理torch.Tensor
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
        
        # 保存原始形状
        original_shape = data_np.shape
        
        # 重塑为2D
        if data_np.ndim > 2:
            data_2d = data_np.reshape(-1, self.total_dims)
        elif data_np.ndim == 2:
            data_2d = data_np
        elif data_np.ndim == 1:
            data_2d = data_np.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported data shape: {original_shape}")
        
        # 应用反变换
        if self.method == 'standard':
            denormalized = data_2d * self.std_ + self.mean_
        elif self.method == 'minmax':
            denormalized = data_2d * (self.max_ - self.min_) + self.min_
        elif self.method == 'robust':
            iqr = self.q75_ - self.q25_
            denormalized = data_2d * iqr + self.median_
        
        # 重塑回原始形状
        denormalized = denormalized.reshape(original_shape)
        
        # 转换回torch.Tensor
        if is_tensor:
            denormalized = torch.from_numpy(denormalized.astype(np.float32)).to(device)
        
        return denormalized
    
    def save_stats(self, filename: Optional[str] = None) -> str:
        """
        保存归一化统计量到磁盘。
        
        Args:
            filename: 保存文件名（可选）
            
        Returns:
            保存文件的完整路径
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Nothing to save.")
        
        if filename is None:
            filename = f"{self.method}_stats.npz"
        
        save_path = self.stats_dir / filename
        
        # 准备保存数据
        save_data = {
            'method': self.method,
            'boundary_dims': self.boundary_dims,
            'equipment_dims': self.equipment_dims,
            'total_dims': self.total_dims,
            'fitted': self.fitted
        }
        
        # 根据方法添加相应统计量
        if self.method == 'standard':
            save_data.update({
                'mean': self.mean_,
                'std': self.std_
            })
        elif self.method == 'minmax':
            save_data.update({
                'min': self.min_,
                'max': self.max_
            })
        elif self.method == 'robust':
            save_data.update({
                'median': self.median_,
                'q25': self.q25_,
                'q75': self.q75_
            })
        
        # 保存到npz文件
        np.savez_compressed(save_path, **save_data)
        
        logger.info(f"Normalization stats saved to {save_path}")
        return str(save_path)
    
    def load_stats(self, filename: Optional[str] = None) -> bool:
        """
        从磁盘加载归一化统计量。
        
        Args:
            filename: 加载文件名（可选）
            
        Returns:
            加载是否成功
        """
        if filename is None:
            filename = f"{self.method}_stats.npz"
        
        load_path = self.stats_dir / filename
        
        if not load_path.exists():
            logger.error(f"Stats file not found: {load_path}")
            return False
        
        try:
            # 加载数据
            with np.load(load_path, allow_pickle=True) as data:
                # 验证方法匹配
                saved_method = str(data['method'])
                if saved_method != self.method:
                    logger.error(f"Method mismatch: expected {self.method}, got {saved_method}")
                    return False
                
                # 验证维度匹配
                saved_dims = int(data['total_dims'])
                if saved_dims != self.total_dims:
                    logger.error(f"Dimension mismatch: expected {self.total_dims}, got {saved_dims}")
                    return False
                
                # 加载统计量
                if self.method == 'standard':
                    self.mean_ = data['mean']
                    self.std_ = data['std']
                elif self.method == 'minmax':
                    self.min_ = data['min']
                    self.max_ = data['max']
                elif self.method == 'robust':
                    self.median_ = data['median']
                    self.q25_ = data['q25']
                    self.q75_ = data['q75']
                
                self.fitted = bool(data['fitted'])
            
            logger.info(f"Normalization stats loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading stats from {load_path}: {e}")
            return False
    
    def get_stats_summary(self) -> Dict:
        """
        获取归一化统计量的摘要信息。
        
        Returns:
            统计量摘要字典
        """
        if not self.fitted:
            return {'fitted': False, 'method': self.method}
        
        summary = {
            'fitted': True,
            'method': self.method,
            'total_dims': self.total_dims,
            'boundary_dims': self.boundary_dims,
            'equipment_dims': self.equipment_dims
        }
        
        if self.method == 'standard':
            summary.update({
                'mean_range': [float(self.mean_.min()), float(self.mean_.max())],
                'std_range': [float(self.std_.min()), float(self.std_.max())],
                'zero_std_cols': int(np.sum(self.std_ == 1.0))  # 检查设置为1的常数列
            })
        elif self.method == 'minmax':
            summary.update({
                'min_range': [float(self.min_.min()), float(self.min_.max())],
                'max_range': [float(self.max_.min()), float(self.max_.max())],
                'const_cols': int(np.sum((self.max_ - self.min_) == 1.0))  # 检查设置范围为1的常数列
            })
        elif self.method == 'robust':
            iqr = self.q75_ - self.q25_
            summary.update({
                'median_range': [float(self.median_.min()), float(self.median_.max())],
                'iqr_range': [float(iqr.min()), float(iqr.max())],
                'zero_iqr_cols': int(np.sum(iqr == 1.0))  # 检查设置为1的零IQR列
            })
        
        return summary