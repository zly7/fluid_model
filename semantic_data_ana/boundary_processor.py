"""
边界数据处理器
用于加载和处理Boundary.csv文件中的边界条件数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import glob


class BoundaryDataProcessor:
    """
    边界数据处理器类
    专门用于处理气管网边界条件数据
    """
    
    def __init__(self):
        """初始化处理器"""
        # 定义气源变量列表
        self.gas_sources = [
            'T_002:SNQ', 'T_003:SNQ', 'T_004:SNQ'
        ]
        
        # 定义分输点变量列表 - 根据实际数据中找到的重要分输点
        self.distribution_points = [
            'E_001:SNQ', 'E_002:SNQ', 'E_003:SNQ', 'E_004:SNQ', 
            'E_005:SNQ', 'E_006:SNQ', 'E_007:SNQ', 'E_008:SNQ', 
            'E_009:SNQ', 'E_060:SNQ', 'E_061:SNQ', 'E_062:SNQ', 
            'E_109:SNQ'
        ]
        
    def find_all_cases(self, data_root: str) -> list:
        """
        查找所有可用的算例
        
        Args:
            data_root: 数据根目录
            
        Returns:
            算例名称列表
        """
        data_root = Path(data_root)
        cases = []
        
        # 搜索train和test目录
        for subset in ['train', 'test']:
            subset_dir = data_root / subset
            if subset_dir.exists():
                for case_dir in subset_dir.iterdir():
                    if case_dir.is_dir():
                        boundary_file = case_dir / 'Boundary.csv'
                        if boundary_file.exists():
                            cases.append(f"{subset}/{case_dir.name}")
        
        return sorted(cases)
    
    def load_boundary_data(self, data_root: str, case_name: str) -> pd.DataFrame:
        """
        加载指定算例的边界数据
        
        Args:
            data_root: 数据根目录
            case_name: 算例名称（例如：train/第001个算例）
            
        Returns:
            处理后的数据框
        """
        try:
            # 构建文件路径
            boundary_file = Path(data_root) / case_name / 'Boundary.csv'
            
            if not boundary_file.exists():
                print(f"边界文件不存在: {boundary_file}")
                return None
            
            # 读取数据
            df = pd.read_csv(boundary_file)
            
            # 处理时间列
            if 'TIME' in df.columns:
                df['TIME'] = pd.to_datetime(df['TIME'])
                df['time_index'] = range(len(df))  # 创建时间索引
            
            # 添加算例信息
            df['case_name'] = case_name
            df['case_id'] = case_name.split('/')[-1]
            
            # 验证必要的列是否存在
            missing_gas = [col for col in self.gas_sources if col not in df.columns]
            missing_dist = [col for col in self.distribution_points if col not in df.columns]
            
            if missing_gas:
                print(f"警告: 缺少气源列: {missing_gas}")
            if missing_dist:
                print(f"警告: 缺少分输点列: {missing_dist}")
            
            return df
            
        except Exception as e:
            print(f"加载边界数据时出错: {e}")
            return None
    
    def get_available_variables(self, df: pd.DataFrame) -> dict:
        """
        获取数据框中可用的变量
        
        Args:
            df: 数据框
            
        Returns:
            包含气源和分输点变量的字典
        """
        available_gas = [col for col in self.gas_sources if col in df.columns]
        available_dist = [col for col in self.distribution_points if col in df.columns]
        
        return {
            'gas_sources': available_gas,
            'distribution_points': available_dist
        }
    
    def filter_data_by_time_range(self, df: pd.DataFrame, start_idx: int = None, end_idx: int = None) -> pd.DataFrame:
        """
        按时间范围过滤数据
        
        Args:
            df: 数据框
            start_idx: 开始索引
            end_idx: 结束索引
            
        Returns:
            过滤后的数据框
        """
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(df)
            
        return df.iloc[start_idx:end_idx].copy()
    
    def get_data_statistics(self, df: pd.DataFrame, columns: list = None) -> dict:
        """
        获取数据统计信息
        
        Args:
            df: 数据框
            columns: 要统计的列名列表
            
        Returns:
            统计信息字典
        """
        if columns is None:
            columns = self.gas_sources + self.distribution_points
        
        # 过滤存在的列
        existing_columns = [col for col in columns if col in df.columns]
        
        if not existing_columns:
            return {}
        
        stats = {}
        for col in existing_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'count': df[col].count(),
                    'null_count': df[col].isnull().sum()
                }
        
        return stats
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        验证数据质量
        
        Args:
            df: 数据框
            
        Returns:
            验证结果字典
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 检查基本结构
        if df is None or df.empty:
            validation['is_valid'] = False
            validation['errors'].append("数据框为空")
            return validation
        
        # 检查时间列
        if 'TIME' not in df.columns:
            validation['warnings'].append("缺少TIME列")
        
        # 检查数据类型
        numeric_columns = self.gas_sources + self.distribution_points
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation['warnings'].append(f"列 {col} 不是数值类型")
                
                # 检查负值（对于SNQ可能不合理）
                if df[col].min() < 0:
                    validation['warnings'].append(f"列 {col} 包含负值")
        
        # 检查缺失值
        for col in numeric_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    validation['warnings'].append(f"列 {col} 有 {null_count} 个缺失值")
        
        return validation