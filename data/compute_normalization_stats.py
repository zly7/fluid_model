"""
统计量预计算工具 - 用于计算数据归一化的统计量并保存到磁盘。

使用方法:
    python compute_normalization_stats.py --data_dir /path/to/data --method standard --max_samples 100

参数:
    --data_dir: 数据根目录
    --method: 归一化方法 (standard, minmax, robust)
    --max_samples: 最大样本数（用于控制内存使用）
    --sequence_length: 序列长度（默认3）
    --output_name: 输出文件名（可选）
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from typing import List
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.processor import DataProcessor
from data.normalizer import DataNormalizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_training_data(data_processor: DataProcessor,
                         max_samples: int = None,
                         sequence_length: int = 3) -> List[np.ndarray]:
    """
    收集训练和验证数据用于计算归一化统计量。
    
    Args:
        data_processor: 数据处理器
        max_samples: 最大样本数
        sequence_length: 序列长度
        
    Returns:
        数据样本列表，每个样本 shape [T, 6712]
    """
    logger.info("Collecting training data for normalization stats computation...")
    
    # 获取训练样本目录
    all_train_samples = data_processor.get_sample_directories('train')
    
    # 分割训练和验证集 (80%训练, 20%验证)
    train_size = int(0.8 * len(all_train_samples))
    train_samples = all_train_samples[:train_size]
    val_samples = all_train_samples[train_size:]
    
    # 合并训练和验证样本用于统计量计算
    all_samples = train_samples + val_samples
    
    # 限制样本数量
    if max_samples is not None and max_samples < len(all_samples):
        # 均匀采样
        indices = np.linspace(0, len(all_samples)-1, max_samples, dtype=int)
        all_samples = [all_samples[i] for i in indices]
        logger.info(f"Using {max_samples} samples out of {len(all_train_samples)} total")
    else:
        logger.info(f"Using all {len(all_samples)} samples")
    
    # 收集数据
    data_samples = []
    successful_loads = 0
    
    for i, sample_dir in enumerate(all_samples):
        try:
            # 加载单个样本的序列数据
            sequences, prediction_mask = data_processor.load_combined_sample_data(
                sample_dir, sequence_length)
            
            if not sequences:
                logger.warning(f"No sequences loaded from {sample_dir.name}")
                continue
            
            # 提取所有序列的input数据
            for input_seq, target_seq, start_time, end_time in sequences:
                # 只使用input序列进行统计量计算，避免与target的重复数据
                data_samples.append(input_seq)  # [T, 6712]
                
                # 注意: target_seq 是时间滑动的结果，与input_seq有重叠，
                # 为避免重复数据影响统计量，这里不加入target_seq
            
            successful_loads += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(all_samples)} samples, "
                           f"collected {len(data_samples)} sequences")
                
        except Exception as e:
            logger.error(f"Error processing sample {sample_dir.name}: {e}")
            continue
    
    logger.info(f"Successfully loaded {successful_loads}/{len(all_samples)} samples, "
               f"total sequences: {len(data_samples)}")
    
    return data_samples


def compute_and_save_stats(data_dir: str,
                          method: str = 'standard',
                          max_samples: int = None,
                          sequence_length: int = 3,
                          output_name: str = None,
                          batch_size: int = 10000) -> str:
    """
    计算并保存归一化统计量。
    
    Args:
        data_dir: 数据根目录
        method: 归一化方法
        max_samples: 最大样本数
        sequence_length: 序列长度
        output_name: 输出文件名
        batch_size: 批处理大小，控制内存使用
        
    Returns:
        保存文件的路径
    """
    logger.info(f"Computing normalization stats using {method} method...")
    
    # 初始化数据处理器
    processor = DataProcessor(data_dir)
    
    # 收集训练数据
    data_samples = collect_training_data(processor, max_samples, sequence_length)
    
    if not data_samples:
        raise ValueError("No data samples collected for stats computation")
    
    # 初始化归一化器
    normalizer = DataNormalizer(data_dir, method=method)
    
    # 计算统计量
    logger.info(f"Fitting normalizer with batch_size={batch_size}...")
    normalizer.fit(data_samples, batch_size=batch_size)
    
    # 打印统计摘要
    stats_summary = normalizer.get_stats_summary()
    logger.info(f"Normalization stats summary: {stats_summary}")
    
    # 保存统计量
    save_path = normalizer.save_stats(output_name)
    
    return save_path


def validate_stats(data_dir: str, method: str, stats_file: str = None) -> bool:
    """
    验证保存的统计量是否正确。
    
    Args:
        data_dir: 数据根目录
        method: 归一化方法
        stats_file: 统计量文件名
        
    Returns:
        验证是否通过
    """
    logger.info("Validating saved normalization stats...")
    
    try:
        # 加载统计量
        normalizer = DataNormalizer(data_dir, method=method)
        if not normalizer.load_stats(stats_file):
            logger.error("Failed to load stats file")
            return False
        
        # 检查统计量
        stats_summary = normalizer.get_stats_summary()
        logger.info(f"Loaded stats summary: {stats_summary}")
        
        # 基本验证
        if not normalizer.fitted:
            logger.error("Normalizer not properly fitted")
            return False
        
        # 测试变换
        test_data = np.random.randn(10, normalizer.total_dims).astype(np.float32)
        
        # 正向变换
        normalized = normalizer.transform(test_data)
        if normalized.shape != test_data.shape:
            logger.error(f"Transform shape mismatch: expected {test_data.shape}, got {normalized.shape}")
            return False
        
        # 反向变换
        denormalized = normalizer.inverse_transform(normalized)
        if not np.allclose(test_data, denormalized, rtol=1e-5, atol=1e-6):
            logger.error("Inverse transform failed - data not recovered correctly")
            return False
        
        logger.info("Stats validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"Stats validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Compute normalization statistics for fluid dynamics data')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--method', type=str, default='standard',
                       choices=['standard', 'minmax', 'robust'],
                       help='Normalization method')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (for memory control)')
    parser.add_argument('--sequence_length', type=int, default=3,
                       help='Sequence length for data loading')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Output filename (optional)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate saved stats after computation')
    parser.add_argument('--force', action='store_true',
                       help='Force recomputation even if stats file exists')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size for processing data (controls memory usage)')
    
    args = parser.parse_args()
    
    try:
        # 检查数据目录
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return 1
        
        # 检查是否已存在统计量文件
        normalizer = DataNormalizer(str(data_dir), method=args.method)
        stats_file = args.output_name or f"{args.method}_stats.npz"
        stats_path = normalizer.stats_dir / stats_file
        
        if stats_path.exists() and not args.force:
            logger.info(f"Stats file already exists: {stats_path}")
            logger.info("Use --force to recompute, or --validate to check existing file")
            
            if args.validate:
                if validate_stats(str(data_dir), args.method, args.output_name):
                    logger.info("Existing stats file is valid")
                    return 0
                else:
                    logger.error("Existing stats file is invalid")
                    return 1
            return 0
        
        # 计算统计量
        save_path = compute_and_save_stats(
            data_dir=str(data_dir),
            method=args.method,
            max_samples=args.max_samples,
            sequence_length=args.sequence_length,
            output_name=args.output_name,
            batch_size=args.batch_size
        )
        
        logger.info(f"Normalization stats saved to: {save_path}")
        
        # 可选验证
        if args.validate:
            if validate_stats(str(data_dir), args.method, args.output_name):
                logger.info("Stats validation successful!")
            else:
                logger.error("Stats validation failed!")
                return 1
        
        logger.info("Stats computation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during stats computation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)