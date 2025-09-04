"""
新的边界数据处理器
专门处理Boundary.csv中指定的气源和分输点的SNQ数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import glob


class BoundaryDataProcessor:
    """
    边界数据处理器
    专门用于提取和处理Boundary.csv中的特定气源和分输点数据
    """
    
    def __init__(self):
        """初始化处理器"""
        # 定义目标气源和分输点
        self.gas_sources = ["T_002:SNQ", "T_003:SNQ", "T_004:SNQ"]
        self.distribution_points = [
            "E_001:SNQ", "E_002:SNQ", "E_003:SNQ", "E_004:SNQ", 
            "E_005:SNQ", "E_006:SNQ", "E_007:SNQ", "E_008:SNQ", 
            "E_009:SNQ", "E_060:SNQ", "E_061:SNQ", "E_062:SNQ", 
            "E_109:SNQ"
        ]
        
        # 所有目标列
        self.target_columns = self.gas_sources + self.distribution_points
        
        # 数据存储
        self.processed_data = {}
    
    def find_all_cases(self, data_root: str) -> List[str]:
        """
        查找所有可用的算例
        
        Args:
            data_root: 数据根目录
            
        Returns:
            算例列表
        """
        data_path = Path(data_root)
        
        # 查找训练集
        train_cases = []
        train_path = data_path / "train"
        if train_path.exists():
            for case_dir in train_path.glob("第*个算例"):
                boundary_file = case_dir / "Boundary.csv"
                if boundary_file.exists():
                    train_cases.append(f"train_{case_dir.name}")
        
        # 查找测试集
        test_cases = []
        test_path = data_path / "test"
        if test_path.exists():
            for case_dir in test_path.glob("第*个算例"):
                boundary_file = case_dir / "Boundary.csv"
                if boundary_file.exists():
                    test_cases.append(f"test_{case_dir.name}")
        
        return sorted(train_cases + test_cases)
    
    def load_boundary_data(self, data_root: str, case_name: str) -> Optional[pd.DataFrame]:
        """
        加载指定算例的边界数据
        
        Args:
            data_root: 数据根目录
            case_name: 算例名称 (格式: train_第001个算例 或 test_第265个算例)
            
        Returns:
            处理后的数据框
        """
        try:
            # 解析算例路径
            if case_name.startswith("train_"):
                subset = "train"
                case_folder = case_name[6:]  # 去掉 "train_"
            elif case_name.startswith("test_"):
                subset = "test"
                case_folder = case_name[5:]  # 去掉 "test_"
            else:
                raise ValueError(f"Invalid case name format: {case_name}")
            
            # 构建文件路径
            data_path = Path(data_root)
            boundary_file = data_path / subset / case_folder / "Boundary.csv"
            
            if not boundary_file.exists():
                print(f"Warning: Boundary file not found: {boundary_file}")
                return None
            
            # 读取数据
            df = pd.read_csv(boundary_file)
            
            # 检查是否包含目标列
            missing_columns = [col for col in self.target_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns in {case_name}: {missing_columns}")
            
            # 提取目标列
            available_columns = [col for col in self.target_columns if col in df.columns]
            result_df = df[["TIME"] + available_columns].copy()
            
            # 转换时间列
            result_df['TIME'] = pd.to_datetime(result_df['TIME'])
            
            # 添加时间索引（30分钟间隔）
            result_df['time_index'] = range(len(result_df))
            
            # 添加算例信息
            result_df['case_name'] = case_name
            result_df['case_id'] = case_folder
            
            return result_df
            
        except Exception as e:
            print(f"Error loading {case_name}: {str(e)}")
            return None
    
    def process_all_cases(self, data_root: str, output_dir: str = None) -> Dict[str, pd.DataFrame]:
        """
        处理所有算例的边界数据
        
        Args:
            data_root: 数据根目录
            output_dir: 输出目录（可选）
            
        Returns:
            处理后的数据字典
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "processed_boundary_data"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 获取所有算例
        all_cases = self.find_all_cases(data_root)
        print(f"Found {len(all_cases)} cases to process")
        
        processed_data = {}
        successful_cases = []
        failed_cases = []
        
        # 处理每个算例
        for case_name in all_cases:
            print(f"Processing {case_name}...")
            
            df = self.load_boundary_data(data_root, case_name)
            if df is not None:
                processed_data[case_name] = df
                successful_cases.append(case_name)
                
                # 保存单独的CSV文件
                output_file = output_path / f"{case_name}_boundary_data.csv"
                df.to_csv(output_file, index=False)
                
            else:
                failed_cases.append(case_name)
        
        # 保存处理摘要
        summary = {
            'total_cases': len(all_cases),
            'successful_cases': len(successful_cases),
            'failed_cases': len(failed_cases),
            'target_columns': self.target_columns,
            'gas_sources': self.gas_sources,
            'distribution_points': self.distribution_points
        }
        
        summary_file = output_path / "processing_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("边界数据处理摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"总算例数: {summary['total_cases']}\n")
            f.write(f"成功处理: {summary['successful_cases']}\n")
            f.write(f"处理失败: {summary['failed_cases']}\n")
            f.write(f"\n目标气源: {', '.join(self.gas_sources)}\n")
            f.write(f"目标分输点: {', '.join(self.distribution_points)}\n")
            
            if failed_cases:
                f.write(f"\n失败的算例:\n")
                for case in failed_cases:
                    f.write(f"- {case}\n")
        
        self.processed_data = processed_data
        print(f"Processing completed. {len(successful_cases)}/{len(all_cases)} cases successful.")
        print(f"Results saved to: {output_path}")
        
        return processed_data
    
    def get_statistics(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算数据统计信息
        
        Args:
            data: 处理后的数据字典
            
        Returns:
            统计信息数据框
        """
        if not data:
            return pd.DataFrame()
        
        stats_list = []
        
        for case_name, df in data.items():
            case_stats = {
                'case_name': case_name,
                'data_points': len(df),
                'time_span_hours': len(df) * 0.5,  # 30分钟间隔
                'start_time': df['TIME'].min(),
                'end_time': df['TIME'].max()
            }
            
            # 每个列的统计信息
            for col in self.target_columns:
                if col in df.columns:
                    case_stats[f"{col}_mean"] = df[col].mean()
                    case_stats[f"{col}_std"] = df[col].std()
                    case_stats[f"{col}_min"] = df[col].min()
                    case_stats[f"{col}_max"] = df[col].max()
                else:
                    case_stats[f"{col}_mean"] = np.nan
                    case_stats[f"{col}_std"] = np.nan
                    case_stats[f"{col}_min"] = np.nan
                    case_stats[f"{col}_max"] = np.nan
            
            stats_list.append(case_stats)
        
        return pd.DataFrame(stats_list)


def main():
    """主函数"""
    # 初始化处理器
    processor = BoundaryDataProcessor()
    
    # 数据根目录
    data_root = "D:/ml_pro_master/chroes/fluid_model/data/dataset"
    
    # 输出目录
    output_dir = "D:/ml_pro_master/chroes/fluid_model/semantic_data_ana/processed_boundary_data"
    
    # 处理所有数据
    print("开始处理边界数据...")
    processed_data = processor.process_all_cases(data_root, output_dir)
    
    # 计算统计信息
    if processed_data:
        print("计算统计信息...")
        stats_df = processor.get_statistics(processed_data)
        
        # 保存统计信息
        stats_file = Path(output_dir) / "statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"统计信息已保存到: {stats_file}")
        
        # 打印基本信息
        print(f"\n处理完成:")
        print(f"- 成功处理 {len(processed_data)} 个算例")
        print(f"- 目标变量: {len(processor.target_columns)} 个")
        print(f"- 输出目录: {output_dir}")


if __name__ == "__main__":
    main()