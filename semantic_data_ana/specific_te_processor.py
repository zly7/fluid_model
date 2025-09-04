"""
特定气源和分输点数据处理器
专门处理指定的气源和分输点在boundary.csv里面的SNQ数据
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re


class SpecificTEProcessor:
    """
    特定气源和分输点数据处理器
    处理指定的气源(T系列)和分输点(E系列)的SNQ数据
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
        
        # 指定需要处理的气源和分输点
        self.target_gas_sources = ['T_002:SNQ', 'T_003:SNQ', 'T_004:SNQ']
        self.target_distribution_points = [
            'E_001:SNQ', 'E_002:SNQ', 'E_003:SNQ', 'E_004:SNQ', 'E_005:SNQ',
            'E_006:SNQ', 'E_007:SNQ', 'E_008:SNQ', 'E_009:SNQ', 'E_060:SNQ',
            'E_061:SNQ', 'E_062:SNQ', 'E_109:SNQ'
        ]
        self.target_columns = ['TIME'] + self.target_gas_sources + self.target_distribution_points
        
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
    
    def extract_target_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        从Boundary.csv中提取目标气源和分输点列
        
        Args:
            df: Boundary数据框
            
        Returns:
            包含目标列的数据框, 实际存在的目标列列表
        """
        # 获取所有列名
        all_columns = df.columns.tolist()
        
        # 检查目标列是否存在
        existing_target_columns = []
        for col in self.target_columns:
            if col in all_columns:
                existing_target_columns.append(col)
        
        # 提取目标列数据
        target_df = df[existing_target_columns].copy()
        
        return target_df, existing_target_columns
    
    def process_single_boundary_file(self, file_path: Path) -> Dict[str, any]:
        """
        处理单个Boundary.csv文件，保持原始的半小时间隔
        
        Args:
            file_path: Boundary.csv文件路径
            
        Returns:
            处理结果字典
        """

        # 读取文件
        df = pd.read_csv(file_path)
        
        # 提取目标列
        target_df, existing_columns = self.extract_target_columns(df)
        
        # 获取算例信息
        case_name = file_path.parent.name
        
        # 解析时间并添加日期信息
        if 'TIME' in target_df.columns:
            target_df['TIME'] = pd.to_datetime(target_df['TIME'])
            target_df['date'] = target_df['TIME'].dt.date
            target_df['hour_minute'] = target_df['TIME'].dt.strftime('%H:%M')
            
            # 按天分组数据
            daily_data = {}
            for date, group in target_df.groupby('date'):
                daily_data[str(date)] = group.reset_index(drop=True)
        
        return {
            'case_name': case_name,
            'file_path': str(file_path),
            'original_shape': df.shape,
            'target_shape': target_df.shape,
            'existing_columns': existing_columns,
            'missing_columns': list(set(self.target_columns) - set(existing_columns)),
            'target_data': target_df,
            'daily_data': daily_data,
            'success': True
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
                raise ValueError(f"处理失败: {file_path}")

        # 统计信息
        summary = {
            'total_files': len(boundary_files),
            'successful': len(successful_cases),
            'failed': len(failed_cases),
            'success_rate': len(successful_cases) / len(boundary_files) * 100,
            'successful_cases': successful_cases,
            'failed_cases': failed_cases,
            'results': results,
            'target_gas_sources': self.target_gas_sources,
            'target_distribution_points': self.target_distribution_points
        }
        
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
            return summary
        
        # 限制处理的案例数量
        cases_to_process = summary['successful_cases']
        if case_limit:
            cases_to_process = cases_to_process[:case_limit]
            print(f"限制处理案例数量为: {case_limit}")
        
        # 保存每个案例的处理后数据
        for i, case_result in enumerate(cases_to_process):
            case_name = case_result['case_name']
            target_data = case_result['target_data']
            
            # 保存整体数据为CSV
            output_file = output_path / f"{case_name}_target_TE.csv"
            target_data.to_csv(output_file, index=False)
            print(f"保存: {output_file}")
            
            # 保存按日分组的数据
            daily_dir = output_path / f"{case_name}_daily"
            daily_dir.mkdir(exist_ok=True)
            
            for date, daily_df in case_result['daily_data'].items():
                daily_file = daily_dir / f"{date}.csv"
                daily_df.to_csv(daily_file, index=False)
        
        # 保存处理汇总信息
        summary_file = output_path / "specific_te_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"特定气源和分输点数据处理汇总报告\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"总文件数: {summary['total_files']}\n")
            f.write(f"成功处理: {summary['successful']}\n")
            f.write(f"处理失败: {summary['failed']}\n")
            f.write(f"成功率: {summary['success_rate']:.1f}%\n\n")
            
            f.write("目标气源(T系列):\n")
            for col in summary['target_gas_sources']:
                f.write(f"  {col}\n")
                
            f.write("\n目标分输点(E系列):\n")
            for col in summary['target_distribution_points']:
                f.write(f"  {col}\n")
            
            # 检查列缺失情况
            if summary['successful_cases']:
                first_success = summary['successful_cases'][0]
                if first_success['missing_columns']:
                    f.write(f"\n缺失的列:\n")
                    for col in first_success['missing_columns']:
                        f.write(f"  {col}\n")
            
            if summary['failed_cases']:
                f.write(f"\n失败案例:\n")
                for case in summary['failed_cases']:
                    f.write(f"  {case['case_name']}: {case['error']}\n")
        
        print(f"\n处理汇总报告已保存到: {summary_file}")
        print(f"成功处理 {len(cases_to_process)} 个案例的特定气源和分输点数据")
        
        return summary


def main():
    """主函数"""
    # 设置数据路径
    data_root = "/home/chbds/zly/gaspipe/fluid_model/data/dataset"
    output_dir = "/home/chbds/zly/gaspipe/fluid_model/semantic_data_ana/specific_te_data"
    
    # 创建处理器
    processor = SpecificTEProcessor(data_root)
    
    # 处理并保存数据（限制处理10个案例进行测试）
    summary = processor.save_processed_data(output_dir, case_limit=10)
    
    return summary


if __name__ == "__main__":
    summary = main()
    print(f"\n处理完成，总共处理了 {summary.get('successful', 0)} 个案例")