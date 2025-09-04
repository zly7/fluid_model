"""
气源和分输点边界数据处理器
处理Boundary.csv文件中的气源(T系列)和分输点(E系列)数据
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re


class BoundaryDataProcessor:
    """
    边界数据处理器，专门处理气源和分输点的边界数据
    """
    
    def __init__(self, data_root: str):
        """
        初始化处理器
        
        Args:
            data_root: 数据根目录路径
        """
        self.data_root = Path(data_root)
        self.train_dir = self.data_root / "train"
        self.test_dir = self.data_root / "test"
        
        # 气源和分输点相关的列标识
        self.gas_source_pattern = re.compile(r'T_\d{3}:(SNQ|SP)')
        self.distribution_point_pattern = re.compile(r'E_\d{3}:SNQ')
        
    def find_all_boundary_files(self) -> List[Path]:
        """
        查找所有Boundary.csv文件
        
        Returns:
            包含所有Boundary.csv文件路径的列表
        """
        boundary_files = []
        
        # 查找训练集中的文件
        if self.train_dir.exists():
            boundary_files.extend(list(self.train_dir.glob("*/Boundary.csv")))
            
        # 查找测试集中的文件
        if self.test_dir.exists():
            boundary_files.extend(list(self.test_dir.glob("*/Boundary.csv")))
            
        return sorted(boundary_files)
    
    def extract_te_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        从Boundary.csv中提取气源(T)和分输点(E)相关的列
        
        Args:
            df: Boundary数据框
            
        Returns:
            包含T和E列的数据框, T系列列名列表, E系列列名列表
        """
        # 获取所有列名
        all_columns = df.columns.tolist()
        
        # 提取气源(T系列)列
        t_columns = []
        for col in all_columns:
            if self.gas_source_pattern.match(col):
                t_columns.append(col)
                
        # 提取分输点(E系列)列  
        e_columns = []
        for col in all_columns:
            if self.distribution_point_pattern.match(col):
                e_columns.append(col)
        
        # 保留TIME列和T、E相关列
        selected_columns = ['TIME'] + t_columns + e_columns
        te_df = df[selected_columns].copy()
        
        return te_df, t_columns, e_columns
    
    def expand_to_minute_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将半小时级别的边界数据扩展为分钟级别
        
        Args:
            df: 半小时级别的边界数据
            
        Returns:
            分钟级别的边界数据
        """
        expanded_rows = []
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            current_time = pd.to_datetime(current_row['TIME'])
            
            # 为当前半小时内的每一分钟创建数据行
            for minute_offset in range(30):
                minute_time = current_time + pd.Timedelta(minutes=minute_offset)
                
                # 创建新行
                new_row = current_row.copy()
                new_row['TIME'] = minute_time.strftime('%Y/%-m/%-d %H:%M')
                expanded_rows.append(new_row)
        
        expanded_df = pd.DataFrame(expanded_rows)
        return expanded_df.reset_index(drop=True)
    
    def process_single_boundary_file(self, file_path: Path) -> Dict[str, any]:
        """
        处理单个Boundary.csv文件
        
        Args:
            file_path: Boundary.csv文件路径
            
        Returns:
            处理结果字典
        """
        try:
            # 读取文件
            df = pd.read_csv(file_path)
            
            # 提取T和E相关列
            te_df, t_columns, e_columns = self.extract_te_columns(df)
            
            # 扩展为分钟级别
            minute_level_df = self.expand_to_minute_level(te_df)
            
            # 获取算例信息
            case_name = file_path.parent.name
            
            return {
                'case_name': case_name,
                'file_path': str(file_path),
                'original_shape': df.shape,
                'te_shape': te_df.shape,
                'minute_level_shape': minute_level_df.shape,
                'gas_source_columns': t_columns,
                'distribution_point_columns': e_columns,
                'minute_level_data': minute_level_df,
                'success': True
            }
            
        except Exception as e:
            return {
                'case_name': file_path.parent.name if file_path.parent else 'unknown',
                'file_path': str(file_path),
                'error': str(e),
                'success': False
            }
    
    def process_all_boundary_files(self) -> Dict[str, any]:
        """
        处理所有Boundary.csv文件
        
        Returns:
            处理结果汇总
        """
        boundary_files = self.find_all_boundary_files()
        
        if not boundary_files:
            return {
                'success': False,
                'message': f"未在 {self.data_root} 中找到任何Boundary.csv文件"
            }
        
        results = []
        successful_cases = []
        failed_cases = []
        
        print(f"找到 {len(boundary_files)} 个Boundary.csv文件")
        
        for i, file_path in enumerate(boundary_files):
            print(f"处理 {i+1}/{len(boundary_files)}: {file_path.parent.name}")
            
            result = self.process_single_boundary_file(file_path)
            results.append(result)
            
            if result['success']:
                successful_cases.append(result)
            else:
                failed_cases.append(result)
                print(f"  错误: {result['error']}")
        
        # 统计信息
        summary = {
            'total_files': len(boundary_files),
            'successful': len(successful_cases),
            'failed': len(failed_cases),
            'success_rate': len(successful_cases) / len(boundary_files) * 100,
            'successful_cases': successful_cases,
            'failed_cases': failed_cases,
            'results': results
        }
        
        # 如果有成功处理的文件，分析列信息
        if successful_cases:
            first_success = successful_cases[0]
            summary.update({
                'gas_source_count': len(first_success['gas_source_columns']),
                'distribution_point_count': len(first_success['distribution_point_columns']),
                'sample_gas_sources': first_success['gas_source_columns'][:10],
                'sample_distribution_points': first_success['distribution_point_columns'][:10]
            })
        
        return summary
    
    def save_processed_data(self, output_dir: str, case_limit: Optional[int] = None):
        """
        保存处理后的数据
        
        Args:
            output_dir: 输出目录
            case_limit: 限制处理的案例数量（用于测试）
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理所有边界文件
        summary = self.process_all_boundary_files()
        
        if not summary['successful_cases']:
            print("没有成功处理的文件，无法保存数据")
            return
        
        # 限制处理的案例数量
        cases_to_process = summary['successful_cases']
        if case_limit:
            cases_to_process = cases_to_process[:case_limit]
            print(f"限制处理案例数量为: {case_limit}")
        
        # 保存每个案例的处理后数据
        for i, case_result in enumerate(cases_to_process):
            case_name = case_result['case_name']
            minute_data = case_result['minute_level_data']
            
            # 保存为CSV
            output_file = output_path / f"{case_name}_TE_minute_level.csv"
            minute_data.to_csv(output_file, index=False)
            print(f"保存: {output_file}")
        
        # 保存处理汇总信息
        summary_file = output_path / "processing_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"边界数据处理汇总报告\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"总文件数: {summary['total_files']}\n")
            f.write(f"成功处理: {summary['successful']}\n")
            f.write(f"处理失败: {summary['failed']}\n")
            f.write(f"成功率: {summary['success_rate']:.1f}%\n\n")
            
            if 'gas_source_count' in summary:
                f.write(f"气源(T系列)数量: {summary['gas_source_count']}\n")
                f.write(f"分输点(E系列)数量: {summary['distribution_point_count']}\n\n")
                
                f.write("气源列示例:\n")
                for col in summary['sample_gas_sources']:
                    f.write(f"  {col}\n")
                    
                f.write("\n分输点列示例:\n")
                for col in summary['sample_distribution_points']:
                    f.write(f"  {col}\n")
            
            if summary['failed_cases']:
                f.write(f"\n失败案例:\n")
                for case in summary['failed_cases']:
                    f.write(f"  {case['case_name']}: {case['error']}\n")
        
        print(f"\n处理汇总报告已保存到: {summary_file}")
        print(f"成功处理 {len(cases_to_process)} 个案例的气源和分输点边界数据")


def main():
    """主函数"""
    # 设置数据路径
    data_root = "/home/chbds/zly/gaspipe/fluid_model/data/dataset"
    output_dir = "/home/chbds/zly/gaspipe/fluid_model/semantic_data_ana/processed_te_data"
    
    # 创建处理器
    processor = BoundaryDataProcessor(data_root)
    
    # 处理并保存数据（限制处理5个案例进行测试）
    processor.save_processed_data(output_dir, case_limit=5)


if __name__ == "__main__":
    main()