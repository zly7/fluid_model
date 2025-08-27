import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional
import math


def create_optimizer(model: torch.nn.Module,
                    optimizer_type: str = 'adamw',
                    learning_rate: float = 1e-3,
                    weight_decay: float = 1e-4,
                    **kwargs) -> torch.optim.Optimizer:
    """创建优化器
    
    Args:
        model: 模型
        optimizer_type: 优化器类型
        learning_rate: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他参数
        
    Returns:
        优化器实例
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_type.lower() == 'adagrad':
        return optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=kwargs.get('eps', 1e-10)
        )
    
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str = 'cosine',
                    **kwargs) -> Optional[_LRScheduler]:
    """创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        **kwargs: 其他参数
        
    Returns:
        学习率调度器实例
    """
    if scheduler_type is None or scheduler_type.lower() == 'none':
        return None
    
    elif scheduler_type.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
    
    elif scheduler_type.lower() == 'cosine_warm':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 0)
        )
    
    elif scheduler_type.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_type.lower() == 'multistep':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [50, 80]),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_type.lower() == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    
    elif scheduler_type.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-7)
        )
    
    elif scheduler_type.lower() == 'linear':
        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get('start_factor', 1.0),
            end_factor=kwargs.get('end_factor', 0.1),
            total_iters=kwargs.get('total_iters', 100)
        )
    
    elif scheduler_type.lower() == 'polynomial':
        return PolynomialLR(
            optimizer,
            total_iters=kwargs.get('total_iters', 100),
            power=kwargs.get('power', 1.0)
        )
    
    elif scheduler_type.lower() == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 10),
            total_epochs=kwargs.get('total_epochs', 100),
            min_lr=kwargs.get('min_lr', 0)
        )
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class PolynomialLR(_LRScheduler):
    """多项式学习率衰减"""
    
    def __init__(self, optimizer, total_iters, power=1.0, last_epoch=-1):
        self.total_iters = total_iters
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        decay_factor = (1 - self.last_epoch / self.total_iters) ** self.power
        return [base_lr * decay_factor for base_lr in self.base_lrs]


class WarmupCosineScheduler(_LRScheduler):
    """带warmup的余弦学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 余弦衰减阶段
            cosine_epochs = self.total_epochs - self.warmup_epochs
            cosine_progress = (self.last_epoch - self.warmup_epochs) / cosine_epochs
            cosine_progress = min(cosine_progress, 1.0)
            
            cos_factor = 0.5 * (1 + math.cos(math.pi * cosine_progress))
            return [self.min_lr + (base_lr - self.min_lr) * cos_factor 
                   for base_lr in self.base_lrs]


class CyclicLR(_LRScheduler):
    """循环学习率调度器"""
    
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, 
                 step_size_down=None, mode='triangular', gamma=1.0, 
                 scale_fn=None, scale_mode='cycle', last_epoch=-1):
        
        self.base_lrs = [base_lr] * len(optimizer.param_groups)
        self.max_lrs = [max_lr] * len(optimizer.param_groups)
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.total_size = self.step_size_up + self.step_size_down
        self.mode = mode
        self.gamma = gamma
        
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        
        super(CyclicLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle
        
        if x <= self.step_size_up / self.total_size:
            scale_factor = x / (self.step_size_up / self.total_size)
        else:
            scale_factor = (x - 1) / (self.step_size_down / self.total_size - 1)
        
        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            
            lrs.append(lr)
        
        return lrs


def get_optimizer_config(optimizer_name: str) -> Dict[str, Any]:
    """获取优化器的默认配置"""
    configs = {
        'adamw': {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        'adam': {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        'sgd': {
            'learning_rate': 1e-2,
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'nesterov': True
        },
        'rmsprop': {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'alpha': 0.99,
            'eps': 1e-8
        }
    }
    
    return configs.get(optimizer_name.lower(), configs['adamw'])


def get_scheduler_config(scheduler_name: str) -> Dict[str, Any]:
    """获取学习率调度器的默认配置"""
    configs = {
        'cosine': {
            'T_max': 100,
            'eta_min': 0
        },
        'cosine_warm': {
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 0
        },
        'step': {
            'step_size': 30,
            'gamma': 0.1
        },
        'multistep': {
            'milestones': [50, 80],
            'gamma': 0.1
        },
        'plateau': {
            'mode': 'min',
            'factor': 0.5,
            'patience': 10,
            'min_lr': 1e-7
        },
        'warmup_cosine': {
            'warmup_epochs': 10,
            'total_epochs': 100,
            'min_lr': 0
        }
    }
    
    return configs.get(scheduler_name.lower(), configs['cosine'])