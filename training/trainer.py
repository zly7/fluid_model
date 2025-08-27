import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import logging
import time
import os
from tqdm import tqdm
import numpy as np

from ..models.utils import save_model, EarlyStopping
from .loss import FluidLoss

logger = logging.getLogger(__name__)


class Trainer:
    """训练器类"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cpu',
                 save_dir: str = 'checkpoints',
                 patience: int = 10,
                 save_best_only: bool = True):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            save_dir: 模型保存目录
            patience: 早停耐心值
            save_best_only: 是否只保存最佳模型
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 早停机制
        self.early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # 最佳指标
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        component_losses = {}
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            
            # 将targets转移到设备
            if targets is not None:
                targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算损失
            if isinstance(self.criterion, FluidLoss):
                loss, losses_dict = self.criterion(outputs, targets)
                
                # 累计分量损失
                for key, value in losses_dict.items():
                    if key not in component_losses:
                        component_losses[key] = 0.0
                    component_losses[key] += value.item()
            else:
                loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item():.6f})
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_component_losses = {k: v / num_batches for k, v in component_losses.items()}
        
        return {'total_loss': avg_loss, **avg_component_losses}
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        component_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs = inputs.to(self.device)
                
                if targets is not None:
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                if isinstance(self.criterion, FluidLoss):
                    loss, losses_dict = self.criterion(outputs, targets)
                    
                    # 累计分量损失
                    for key, value in losses_dict.items():
                        if key not in component_losses:
                            component_losses[key] = 0.0
                        component_losses[key] += value.item()
                else:
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'val_loss': loss.item():.6f})
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_component_losses = {k: v / num_batches for k, v in component_losses.items()}
        
        return {'total_loss': avg_loss, **avg_component_losses}
    
    def train(self, 
              epochs: int, 
              log_interval: int = 10,
              save_interval: int = 10) -> Dict[str, List[float]]:
        """训练模型
        
        Args:
            epochs: 训练轮数
            log_interval: 日志输出间隔
            save_interval: 模型保存间隔
            
        Returns:
            训练历史字典
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['total_loss'])
            
            # 验证
            val_metrics = self.validate_epoch()
            val_loss = val_metrics.get('total_loss', float('inf'))
            if val_metrics:
                self.val_losses.append(val_loss)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            
            # 日志输出
            if epoch % log_interval == 0 or epoch == 1:
                log_msg = f"Epoch {epoch:3d}/{epochs} - "
                log_msg += f"Time: {epoch_time:.2f}s - "
                log_msg += f"Train Loss: {train_metrics['total_loss']:.6f}"
                
                if val_metrics:
                    log_msg += f" - Val Loss: {val_loss:.6f}"
                
                if self.scheduler:
                    log_msg += f" - LR: {current_lr:.2e}"
                
                logger.info(log_msg)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                
                if self.save_best_only:
                    best_model_path = os.path.join(self.save_dir, 'best_model.pth')
                    save_model(
                        self.model, self.optimizer, epoch, val_loss, best_model_path,
                        additional_info={
                            'train_loss': train_metrics['total_loss'],
                            'val_loss': val_loss,
                            'epoch': epoch
                        }
                    )
            
            # 定期保存检查点
            if epoch % save_interval == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
                save_model(
                    self.model, self.optimizer, epoch, val_loss, checkpoint_path,
                    additional_info={
                        'train_loss': train_metrics['total_loss'],
                        'val_loss': val_loss,
                        'epoch': epoch
                    }
                )
            
            # 早停检查
            if self.val_loader and self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        component_losses = {}
        all_predictions = {}
        all_targets = {}
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Evaluating")
            
            for inputs, targets in progress_bar:
                inputs = inputs.to(self.device)
                
                if targets is not None:
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                if targets is not None:
                    if isinstance(self.criterion, FluidLoss):
                        loss, losses_dict = self.criterion(outputs, targets)
                        
                        for key, value in losses_dict.items():
                            if key not in component_losses:
                                component_losses[key] = 0.0
                            component_losses[key] += value.item()
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                
                # 收集预测结果
                for output_name, output_tensor in outputs.items():
                    if output_name not in all_predictions:
                        all_predictions[output_name] = []
                    all_predictions[output_name].append(output_tensor.cpu().numpy())
                
                # 收集真实标签
                if targets is not None:
                    for target_name, target_tensor in targets.items():
                        if target_name not in all_targets:
                            all_targets[target_name] = []
                        all_targets[target_name].append(target_tensor.cpu().numpy())
                
                num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_component_losses = {k: v / num_batches for k, v in component_losses.items()}
        
        # 计算评估指标
        metrics = {'total_loss': avg_loss, **avg_component_losses}
        
        if all_targets:
            # 计算MSE、MAE等指标
            for output_name in all_predictions:
                if output_name in all_targets:
                    pred = np.concatenate(all_predictions[output_name])
                    target = np.concatenate(all_targets[output_name])
                    
                    mse = np.mean((pred - target) ** 2)
                    mae = np.mean(np.abs(pred - target))
                    
                    metrics[f'{output_name}_mse'] = mse
                    metrics[f'{output_name}_mae'] = mae
        
        return metrics, all_predictions