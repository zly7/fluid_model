import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Union
import logging
from tqdm import tqdm

from ..models.utils import load_model
from ..data.dataset import DataProcessor

logger = logging.getLogger(__name__)


class FluidPredictor:
    """天然气管网预测器"""
    
    def __init__(self,
                 model: nn.Module,
                 model_path: Optional[str] = None,
                 device: str = 'cpu',
                 batch_size: int = 32):
        """
        Args:
            model: 模型实例
            model_path: 模型权重路径
            device: 设备
            batch_size: 批次大小
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        # 加载模型权重
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.model.to(device)
        self.model.eval()
        
        # 数据处理器
        self.processor = DataProcessor("")
    
    def load_model(self, model_path: str) -> Dict:
        """加载模型权重"""
        info = load_model(self.model, model_path, device=self.device)
        logger.info(f"Model loaded successfully from {model_path}")
        return info
    
    def predict_single(self, boundary_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """预测单个算例
        
        Args:
            boundary_data: 边界条件数据
            
        Returns:
            预测结果字典
        """
        # 预处理输入数据
        processed_input = self.processor.preprocess_boundary_data(boundary_data)
        
        # 转换为张量
        input_tensor = torch.from_numpy(processed_input).to(self.device)
        
        # 添加批次维度
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.dim() == 2 and input_tensor.shape[0] > 1:
            # 如果是时间序列数据，保持序列维度
            pass
        
        # 预测
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # 转换为numpy数组
        results = {}
        for output_name, output_tensor in predictions.items():
            results[output_name] = output_tensor.cpu().numpy()
        
        return results
    
    def predict_batch(self, boundary_data_list: List[pd.DataFrame]) -> List[Dict[str, np.ndarray]]:
        """批量预测
        
        Args:
            boundary_data_list: 边界条件数据列表
            
        Returns:
            预测结果列表
        """
        results = []
        
        # 分批处理
        for i in tqdm(range(0, len(boundary_data_list), self.batch_size), 
                      desc="Predicting"):
            batch_data = boundary_data_list[i:i+self.batch_size]
            
            # 预处理批次数据
            batch_inputs = []
            for boundary_data in batch_data:
                processed_input = self.processor.preprocess_boundary_data(boundary_data)
                batch_inputs.append(processed_input)
            
            # 堆叠为批次张量
            batch_tensor = torch.from_numpy(np.stack(batch_inputs)).to(self.device)
            
            # 预测
            with torch.no_grad():
                batch_predictions = self.model(batch_tensor)
            
            # 分解批次结果
            for j in range(len(batch_data)):
                sample_result = {}
                for output_name, output_tensor in batch_predictions.items():
                    sample_result[output_name] = output_tensor[j].cpu().numpy()
                results.append(sample_result)
        
        return results
    
    def predict_from_files(self, 
                          test_data_dir: str,
                          output_dir: str) -> None:
        """从文件预测并保存结果
        
        Args:
            test_data_dir: 测试数据目录
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有测试算例
        case_dirs = [d for d in os.listdir(test_data_dir) 
                    if d.startswith('第') and d.endswith('个算例')]
        case_dirs.sort()
        
        logger.info(f"Found {len(case_dirs)} test cases")
        
        # 逐个预测
        for case_dir in tqdm(case_dirs, desc="Processing test cases"):
            case_path = os.path.join(test_data_dir, case_dir)
            boundary_path = os.path.join(case_path, 'Boundary.csv')
            
            if not os.path.exists(boundary_path):
                logger.warning(f"Boundary.csv not found in {case_path}")
                continue
            
            # 加载边界数据
            boundary_data = pd.read_csv(boundary_path)
            
            # 预测
            predictions = self.predict_single(boundary_data)
            
            # 保存结果
            case_output_dir = os.path.join(output_dir, case_dir)
            os.makedirs(case_output_dir, exist_ok=True)
            
            self.save_predictions(predictions, case_output_dir)
    
    def save_predictions(self, 
                        predictions: Dict[str, np.ndarray], 
                        output_dir: str) -> None:
        """保存预测结果
        
        Args:
            predictions: 预测结果字典
            output_dir: 输出目录
        """
        for output_file, prediction in predictions.items():
            # 确保输出文件名正确
            if not output_file.endswith('.csv'):
                output_file = f"{output_file}.csv"
            
            output_path = os.path.join(output_dir, output_file)
            
            # 转换为DataFrame并保存
            if prediction.ndim == 1:
                # 单个时间步
                df = pd.DataFrame([prediction])
            else:
                # 多个时间步
                df = pd.DataFrame(prediction)
            
            # 添加TIME列（如果需要）
            if 'TIME' not in df.columns:
                time_steps = len(df)
                time_values = [f"2025/01/01 {i//2:02d}:{(i%2)*30:02d}:00" 
                              for i in range(time_steps)]
                df.insert(0, 'TIME', time_values[:time_steps])
            
            df.to_csv(output_path, index=False)
    
    def predict_with_uncertainty(self, 
                                boundary_data: pd.DataFrame,
                                num_samples: int = 100,
                                enable_dropout: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """带不确定性的预测（Monte Carlo Dropout）
        
        Args:
            boundary_data: 边界条件数据
            num_samples: 采样次数
            enable_dropout: 是否启用dropout
            
        Returns:
            预测结果字典，包含均值和标准差
        """
        if enable_dropout:
            # 启用dropout进行蒙特卡洛采样
            self.model.train()
        
        # 预处理输入数据
        processed_input = self.processor.preprocess_boundary_data(boundary_data)
        input_tensor = torch.from_numpy(processed_input).to(self.device)
        
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # 多次采样
        samples = {output_name: [] for output_name in self.model.output_files}
        
        with torch.no_grad():
            for _ in range(num_samples):
                predictions = self.model(input_tensor)
                
                for output_name, output_tensor in predictions.items():
                    samples[output_name].append(output_tensor.cpu().numpy())
        
        # 计算统计量
        results = {}
        for output_name, sample_list in samples.items():
            sample_array = np.stack(sample_list, axis=0)  # [num_samples, ...]
            
            results[output_name] = {
                'mean': np.mean(sample_array, axis=0),
                'std': np.std(sample_array, axis=0),
                'samples': sample_array
            }
        
        # 恢复eval模式
        self.model.eval()
        
        return results
    
    def explain_prediction(self, 
                          boundary_data: pd.DataFrame,
                          method: str = 'integrated_gradients') -> Dict[str, np.ndarray]:
        """解释预测结果
        
        Args:
            boundary_data: 边界条件数据
            method: 解释方法
            
        Returns:
            特征重要性字典
        """
        # 预处理输入数据
        processed_input = self.processor.preprocess_boundary_data(boundary_data)
        input_tensor = torch.from_numpy(processed_input).to(self.device)
        input_tensor.requires_grad_(True)
        
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        if method == 'gradient':
            return self._gradient_explanation(input_tensor)
        elif method == 'integrated_gradients':
            return self._integrated_gradients_explanation(input_tensor)
        else:
            raise ValueError(f"Unsupported explanation method: {method}")
    
    def _gradient_explanation(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """梯度解释"""
        explanations = {}
        
        predictions = self.model(input_tensor)
        
        for output_name, output_tensor in predictions.items():
            # 计算输出对输入的梯度
            grad_outputs = torch.ones_like(output_tensor)
            gradients = torch.autograd.grad(
                outputs=output_tensor,
                inputs=input_tensor,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=True
            )[0]
            
            explanations[output_name] = gradients.cpu().numpy()
        
        return explanations
    
    def _integrated_gradients_explanation(self, 
                                        input_tensor: torch.Tensor,
                                        steps: int = 50) -> Dict[str, np.ndarray]:
        """积分梯度解释"""
        baseline = torch.zeros_like(input_tensor)
        
        explanations = {}
        
        # 生成插值路径
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        gradients_list = {output_name: [] for output_name in self.model.output_files}
        
        for alpha in alphas:
            interpolated_input = baseline + alpha * (input_tensor - baseline)
            interpolated_input.requires_grad_(True)
            
            predictions = self.model(interpolated_input)
            
            for output_name, output_tensor in predictions.items():
                grad_outputs = torch.ones_like(output_tensor)
                gradients = torch.autograd.grad(
                    outputs=output_tensor,
                    inputs=interpolated_input,
                    grad_outputs=grad_outputs,
                    create_graph=False,
                    retain_graph=True
                )[0]
                
                gradients_list[output_name].append(gradients)
        
        # 计算积分梯度
        for output_name in self.model.output_files:
            avg_gradients = torch.stack(gradients_list[output_name]).mean(dim=0)
            integrated_gradients = (input_tensor - baseline) * avg_gradients
            explanations[output_name] = integrated_gradients.cpu().numpy()
        
        return explanations