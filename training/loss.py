import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
import torch.nn.functional as F


class FluidLoss(nn.Module):
    """天然气管网多任务损失函数"""
    
    def __init__(self, 
                 output_weights: Optional[Dict[str, float]] = None,
                 loss_type: str = 'mse',
                 reduction: str = 'mean'):
        """
        Args:
            output_weights: 各输出的权重字典
            loss_type: 损失类型 ('mse', 'mae', 'smooth_l1', 'huber')
            reduction: 减少方式 ('mean', 'sum', 'none')
        """
        super(FluidLoss, self).__init__()
        
        self.output_weights = output_weights or {}
        self.loss_type = loss_type
        self.reduction = reduction
        
        # 选择基础损失函数
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss(reduction='none')
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: 预测结果字典
            targets: 目标结果字典
            
        Returns:
            total_loss: 总损失
            component_losses: 各组件损失字典
        """
        component_losses = {}
        total_loss = 0.0
        
        for output_name in predictions.keys():
            if output_name not in targets:
                continue
            
            pred = predictions[output_name]
            target = targets[output_name]
            
            # 计算基础损失
            loss = self.base_loss(pred, target)
            
            # 应用reduction
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
            
            component_losses[output_name] = loss
            
            # 应用权重
            weight = self.output_weights.get(output_name, 1.0)
            total_loss += weight * loss
        
        return total_loss, component_losses


class PhysicsLoss(nn.Module):
    """物理约束损失函数"""
    
    def __init__(self, physics_weight: float = 1.0):
        """
        Args:
            physics_weight: 物理约束权重
        """
        super(PhysicsLoss, self).__init__()
        self.physics_weight = physics_weight
    
    def mass_conservation_loss(self, 
                             flow_in: torch.Tensor, 
                             flow_out: torch.Tensor) -> torch.Tensor:
        """质量守恒损失"""
        return torch.mean((flow_in - flow_out) ** 2)
    
    def momentum_conservation_loss(self,
                                 pressure_in: torch.Tensor,
                                 pressure_out: torch.Tensor,
                                 flow: torch.Tensor) -> torch.Tensor:
        """动量守恒损失（简化版）"""
        pressure_drop = pressure_in - pressure_out
        # 简化的压降-流量关系
        expected_pressure_drop = 0.01 * flow ** 2  # 简化假设
        return torch.mean((pressure_drop - expected_pressure_drop) ** 2)
    
    def energy_conservation_loss(self,
                               temp_in: torch.Tensor,
                               temp_out: torch.Tensor,
                               flow: torch.Tensor) -> torch.Tensor:
        """能量守恒损失"""
        temp_change = torch.abs(temp_in - temp_out)
        # 简化的温度变化约束
        max_temp_change = 0.1 * flow + 1.0  # 简化假设
        violation = torch.relu(temp_change - max_temp_change)
        return torch.mean(violation ** 2)
    
    def forward(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算物理约束损失"""
        physics_loss = 0.0
        
        # 提取相关预测值
        flows_in = []
        flows_out = []
        pressures_in = []
        pressures_out = []
        temps_in = []
        temps_out = []
        
        for output_name, output_tensor in predictions.items():
            # 这里需要根据实际的输出格式来解析
            # 假设输出张量的列对应不同的物理量
            if 'q_in' in output_name or 'flow_in' in output_name:
                flows_in.append(output_tensor)
            elif 'q_out' in output_name or 'flow_out' in output_name:
                flows_out.append(output_tensor)
            elif 'p_in' in output_name or 'pressure_in' in output_name:
                pressures_in.append(output_tensor)
            elif 'p_out' in output_name or 'pressure_out' in output_name:
                pressures_out.append(output_tensor)
            elif 't_in' in output_name or 'temp_in' in output_name:
                temps_in.append(output_tensor)
            elif 't_out' in output_name or 'temp_out' in output_name:
                temps_out.append(output_tensor)
        
        # 计算物理约束损失
        if flows_in and flows_out:
            for flow_in, flow_out in zip(flows_in, flows_out):
                physics_loss += self.mass_conservation_loss(flow_in, flow_out)
        
        if pressures_in and pressures_out and flows_in:
            for p_in, p_out, flow in zip(pressures_in, pressures_out, flows_in):
                physics_loss += self.momentum_conservation_loss(p_in, p_out, flow)
        
        if temps_in and temps_out and flows_in:
            for t_in, t_out, flow in zip(temps_in, temps_out, flows_in):
                physics_loss += self.energy_conservation_loss(t_in, t_out, flow)
        
        return self.physics_weight * physics_loss


class CombinedLoss(nn.Module):
    """组合损失函数：数据损失 + 物理约束损失"""
    
    def __init__(self,
                 data_loss: nn.Module,
                 physics_loss: nn.Module,
                 physics_weight: float = 0.1):
        """
        Args:
            data_loss: 数据拟合损失
            physics_loss: 物理约束损失
            physics_weight: 物理约束权重
        """
        super(CombinedLoss, self).__init__()
        self.data_loss = data_loss
        self.physics_loss = physics_loss
        self.physics_weight = physics_weight
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            total_loss: 总损失
            loss_components: 损失组件字典
        """
        # 计算数据损失
        if isinstance(self.data_loss, FluidLoss):
            data_loss, data_components = self.data_loss(predictions, targets)
        else:
            data_loss = self.data_loss(predictions, targets)
            data_components = {'data_loss': data_loss}
        
        # 计算物理损失
        physics_loss = self.physics_loss(predictions)
        
        # 总损失
        total_loss = data_loss + self.physics_weight * physics_loss
        
        # 损失组件
        loss_components = {
            **data_components,
            'physics_loss': physics_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_components


class FocalLoss(nn.Module):
    """Focal Loss用于处理不平衡数据"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测值
            targets: 目标值
        """
        mse_loss = F.mse_loss(inputs, targets, reduction='none')
        
        # 计算权重
        pt = torch.exp(-mse_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * mse_loss
        
        return focal_loss.mean()


class AdaptiveLoss(nn.Module):
    """自适应损失函数"""
    
    def __init__(self, num_tasks: int):
        """
        Args:
            num_tasks: 任务数量
        """
        super(AdaptiveLoss, self).__init__()
        self.num_tasks = num_tasks
        
        # 学习任务权重的对数
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: 各任务损失列表
        """
        total_loss = 0.0
        
        for i, loss in enumerate(losses):
            # 使用不确定性加权
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss


class RobustLoss(nn.Module):
    """鲁棒损失函数，对异常值不敏感"""
    
    def __init__(self, loss_type: str = 'huber', delta: float = 1.0):
        """
        Args:
            loss_type: 损失类型 ('huber', 'cauchy', 'geman_mcclure')
            delta: 鲁棒参数
        """
        super(RobustLoss, self).__init__()
        self.loss_type = loss_type
        self.delta = delta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: 预测值
            targets: 目标值
        """
        residual = predictions - targets
        
        if self.loss_type == 'huber':
            return F.huber_loss(predictions, targets, delta=self.delta)
        elif self.loss_type == 'cauchy':
            return torch.mean(torch.log(1 + (residual / self.delta) ** 2))
        elif self.loss_type == 'geman_mcclure':
            return torch.mean(residual ** 2 / (1 + residual ** 2 / self.delta ** 2))
        else:
            raise ValueError(f"Unsupported robust loss type: {self.loss_type}")