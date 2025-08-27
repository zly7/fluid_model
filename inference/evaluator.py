import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os
from scipy.stats import pearsonr

from ..data.dataset import DataProcessor

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, output_dir: str = 'evaluation_results'):
        """
        Args:
            output_dir: 评估结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.processor = DataProcessor("")
    
    def evaluate_predictions(self, 
                           predictions: Dict[str, np.ndarray],
                           ground_truth: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """评估预测结果
        
        Args:
            predictions: 预测结果字典
            ground_truth: 真实值字典
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        for output_name in predictions.keys():
            if output_name not in ground_truth:
                logger.warning(f"Ground truth not found for {output_name}")
                continue
            
            pred = predictions[output_name].flatten()
            true = ground_truth[output_name].flatten()
            
            # 基础回归指标
            mse = mean_squared_error(true, pred)
            mae = mean_absolute_error(true, pred)
            rmse = np.sqrt(mse)
            
            # R²分数
            r2 = r2_score(true, pred)
            
            # 相关系数
            corr, p_value = pearsonr(true, pred)
            
            # 平均绝对百分比误差
            mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100
            
            # 标准化指标
            normalized_mse = mse / (np.var(true) + 1e-8)
            normalized_mae = mae / (np.mean(np.abs(true)) + 1e-8)
            
            metrics[output_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'correlation': corr,
                'p_value': p_value,
                'mape': mape,
                'normalized_mse': normalized_mse,
                'normalized_mae': normalized_mae
            }
        
        return metrics
    
    def evaluate_from_files(self,
                          predictions_dir: str,
                          ground_truth_dir: str) -> Dict[str, Dict[str, float]]:
        """从文件评估预测结果
        
        Args:
            predictions_dir: 预测结果目录
            ground_truth_dir: 真实值目录
            
        Returns:
            评估指标字典
        """
        all_metrics = {}
        
        # 获取所有测试算例
        case_dirs = [d for d in os.listdir(predictions_dir)
                    if os.path.isdir(os.path.join(predictions_dir, d))]
        case_dirs.sort()
        
        # 收集所有预测和真实值
        all_predictions = {}
        all_ground_truth = {}
        
        for case_dir in case_dirs:
            pred_case_path = os.path.join(predictions_dir, case_dir)
            true_case_path = os.path.join(ground_truth_dir, case_dir)
            
            if not os.path.exists(true_case_path):
                logger.warning(f"Ground truth not found for {case_dir}")
                continue
            
            # 加载预测结果
            case_predictions = self._load_case_outputs(pred_case_path)
            case_ground_truth = self._load_case_outputs(true_case_path)
            
            # 累积结果
            for output_name, data in case_predictions.items():
                if output_name not in all_predictions:
                    all_predictions[output_name] = []
                all_predictions[output_name].append(data)
            
            for output_name, data in case_ground_truth.items():
                if output_name not in all_ground_truth:
                    all_ground_truth[output_name] = []
                all_ground_truth[output_name].append(data)
        
        # 合并所有数据
        merged_predictions = {}
        merged_ground_truth = {}
        
        for output_name in all_predictions.keys():
            merged_predictions[output_name] = np.concatenate(all_predictions[output_name])
            if output_name in all_ground_truth:
                merged_ground_truth[output_name] = np.concatenate(all_ground_truth[output_name])
        
        # 计算指标
        all_metrics = self.evaluate_predictions(merged_predictions, merged_ground_truth)
        
        return all_metrics
    
    def _load_case_outputs(self, case_path: str) -> Dict[str, np.ndarray]:
        """加载单个算例的输出文件"""
        outputs = {}
        output_files = ['B.csv', 'T&E.csv', 'H.csv', 'C.csv', 'N.csv', 'R.csv', 'P.csv']
        
        for output_file in output_files:
            file_path = os.path.join(case_path, output_file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # 移除TIME列
                if 'TIME' in df.columns:
                    df = df.drop(columns=['TIME'])
                outputs[output_file] = df.values
        
        return outputs
    
    def create_evaluation_report(self, 
                               metrics: Dict[str, Dict[str, float]],
                               save_path: Optional[str] = None) -> str:
        """创建评估报告
        
        Args:
            metrics: 评估指标字典
            save_path: 报告保存路径
            
        Returns:
            报告内容
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("模型评估报告")
        report_lines.append("=" * 80)
        report_lines.append()
        
        # 总体统计
        all_mse = [metric['mse'] for metric in metrics.values()]
        all_mae = [metric['mae'] for metric in metrics.values()]
        all_r2 = [metric['r2_score'] for metric in metrics.values()]
        
        report_lines.append("总体指标:")
        report_lines.append(f"  平均MSE: {np.mean(all_mse):.6f}")
        report_lines.append(f"  平均MAE: {np.mean(all_mae):.6f}")
        report_lines.append(f"  平均R²:  {np.mean(all_r2):.6f}")
        report_lines.append()
        
        # 各组件详细指标
        report_lines.append("各组件详细指标:")
        report_lines.append("-" * 80)
        
        for output_name, metric_dict in metrics.items():
            report_lines.append(f"{output_name}:")
            report_lines.append(f"  MSE:              {metric_dict['mse']:.6f}")
            report_lines.append(f"  MAE:              {metric_dict['mae']:.6f}")
            report_lines.append(f"  RMSE:             {metric_dict['rmse']:.6f}")
            report_lines.append(f"  R² Score:         {metric_dict['r2_score']:.6f}")
            report_lines.append(f"  Correlation:      {metric_dict['correlation']:.6f}")
            report_lines.append(f"  MAPE:             {metric_dict['mape']:.2f}%")
            report_lines.append(f"  Normalized MSE:   {metric_dict['normalized_mse']:.6f}")
            report_lines.append(f"  Normalized MAE:   {metric_dict['normalized_mae']:.6f}")
            report_lines.append()
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Evaluation report saved to {save_path}")
        return report_content
    
    def plot_predictions_vs_truth(self,
                                 predictions: Dict[str, np.ndarray],
                                 ground_truth: Dict[str, np.ndarray],
                                 max_points: int = 10000) -> None:
        """绘制预测值vs真实值散点图
        
        Args:
            predictions: 预测结果字典
            ground_truth: 真实值字典
            max_points: 最大绘制点数
        """
        n_outputs = len(predictions)
        cols = min(3, n_outputs)
        rows = (n_outputs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, output_name in enumerate(predictions.keys()):
            if output_name not in ground_truth:
                continue
            
            pred = predictions[output_name].flatten()
            true = ground_truth[output_name].flatten()
            
            # 随机采样以减少绘制点数
            if len(pred) > max_points:
                indices = np.random.choice(len(pred), max_points, replace=False)
                pred = pred[indices]
                true = true[indices]
            
            ax = axes[i]
            
            # 散点图
            ax.scatter(true, pred, alpha=0.5, s=1)
            
            # 理想线
            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # 设置标签和标题
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{output_name}')
            
            # 添加R²分数
            r2 = r2_score(true, pred)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(len(predictions), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'predictions_vs_truth.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_distribution(self,
                              predictions: Dict[str, np.ndarray],
                              ground_truth: Dict[str, np.ndarray]) -> None:
        """绘制误差分布图"""
        n_outputs = len(predictions)
        cols = min(3, n_outputs)
        rows = (n_outputs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, output_name in enumerate(predictions.keys()):
            if output_name not in ground_truth:
                continue
            
            pred = predictions[output_name].flatten()
            true = ground_truth[output_name].flatten()
            errors = pred - true
            
            ax = axes[i]
            
            # 直方图
            ax.hist(errors, bins=50, alpha=0.7, density=True)
            
            # 添加统计信息
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            
            ax.axvline(mean_error, color='red', linestyle='--', 
                      label=f'Mean: {mean_error:.6f}')
            ax.axvline(mean_error + std_error, color='orange', linestyle=':', 
                      alpha=0.7, label=f'±1σ: ±{std_error:.6f}')
            ax.axvline(mean_error - std_error, color='orange', linestyle=':', alpha=0.7)
            
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Density')
            ax.set_title(f'{output_name} - Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(predictions), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """绘制指标对比图"""
        output_names = list(metrics.keys())
        metric_names = ['mse', 'mae', 'r2_score', 'mape']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric_name in enumerate(metric_names):
            values = [metrics[name][metric_name] for name in output_names]
            
            ax = axes[i]
            bars = ax.bar(range(len(output_names)), values)
            
            # 设置标签
            ax.set_xlabel('Output Components')
            ax.set_ylabel(metric_name.upper())
            ax.set_title(f'{metric_name.upper()} Comparison')
            ax.set_xticks(range(len(output_names)))
            ax.set_xticklabels(output_names, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(self,
                                  predictions_dir: str,
                                  ground_truth_dir: str) -> Dict[str, Dict[str, float]]:
        """创建完整的评估报告
        
        Args:
            predictions_dir: 预测结果目录
            ground_truth_dir: 真实值目录
            
        Returns:
            评估指标字典
        """
        logger.info("Starting comprehensive evaluation...")
        
        # 计算评估指标
        metrics = self.evaluate_from_files(predictions_dir, ground_truth_dir)
        
        # 加载数据进行可视化
        case_dirs = [d for d in os.listdir(predictions_dir)
                    if os.path.isdir(os.path.join(predictions_dir, d))][:10]  # 限制数量
        
        all_predictions = {}
        all_ground_truth = {}
        
        for case_dir in case_dirs:
            pred_case_path = os.path.join(predictions_dir, case_dir)
            true_case_path = os.path.join(ground_truth_dir, case_dir)
            
            if not os.path.exists(true_case_path):
                continue
            
            case_predictions = self._load_case_outputs(pred_case_path)
            case_ground_truth = self._load_case_outputs(true_case_path)
            
            for output_name, data in case_predictions.items():
                if output_name not in all_predictions:
                    all_predictions[output_name] = []
                all_predictions[output_name].append(data)
            
            for output_name, data in case_ground_truth.items():
                if output_name not in all_ground_truth:
                    all_ground_truth[output_name] = []
                all_ground_truth[output_name].append(data)
        
        # 合并数据
        merged_predictions = {}
        merged_ground_truth = {}
        
        for output_name in all_predictions.keys():
            merged_predictions[output_name] = np.concatenate(all_predictions[output_name])
            if output_name in all_ground_truth:
                merged_ground_truth[output_name] = np.concatenate(all_ground_truth[output_name])
        
        # 生成报告和图表
        self.create_evaluation_report(metrics)
        self.plot_predictions_vs_truth(merged_predictions, merged_ground_truth)
        self.plot_error_distribution(merged_predictions, merged_ground_truth)
        self.plot_metrics_comparison(metrics)
        
        logger.info(f"Comprehensive evaluation completed. Results saved to {self.output_dir}")
        
        return metrics