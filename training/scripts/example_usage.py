#!/usr/bin/env python3
"""
使用示例：展示如何使用 FluidDecoder 训练系统。

这个脚本展示了：
1. 创建模型配置
2. 设置训练参数
3. 使用 transformers.Trainer 进行训练
4. 推理和评估
"""

import os
import sys
from pathlib import Path
import torch
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# 导入项目模块
from training import (
    FluidDecoderForTraining,
    FluidDecoderConfig,
    FluidDataCollator,
    compute_fluid_metrics,
    create_training_callbacks,
    FluidInference
)
from data import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """主示例函数。"""
    logger.info("FluidDecoder 训练系统使用示例")
    
    # ====== 1. 模型配置 ======
    logger.info("创建模型配置...")
    config = FluidDecoderConfig(
        d_model=256,           # 隐藏层维度
        n_heads=8,            # 注意力头数  
        n_layers=6,           # Decoder层数
        d_ff=1024,            # 前馈网络维度
        input_dim=6712,       # 输入维度 (固定)
        output_dim=6712,      # 输出维度 (固定)
        boundary_dims=538,    # 边界条件维度
        dropout=0.1,          # Dropout概率
        activation="gelu"     # 激活函数
    )
    
    # 创建模型
    logger.info("创建模型...")
    model = FluidDecoderForTraining(config)
    logger.info(f"模型参数量: {model.num_parameters():,}")
    
    # ====== 2. 数据准备 ======
    logger.info("准备数据...")
    try:
        # 创建数据加载器
        train_dataloader, eval_dataloader = create_data_loaders(
            data_dir="data/dataset",
            batch_size=16,         # 较小的批次大小用于示例
            eval_batch_size=32,
            time_steps=3,
            num_workers=2,
            pin_memory=True
        )
        
        # 获取数据集
        train_dataset = train_dataloader.dataset
        eval_dataset = eval_dataloader.dataset
        
        logger.info(f"训练样本数: {len(train_dataset)}")
        logger.info(f"验证样本数: {len(eval_dataset)}")
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        logger.info("使用模拟数据集进行演示...")
        
        # 创建模拟数据集
        from torch.utils.data import TensorDataset
        
        # 模拟训练数据
        train_inputs = torch.randn(100, 3, 6712)     # [B, T, V]
        train_targets = torch.randn(100, 3, 6712)    # [B, T, V]
        train_masks = torch.ones(100, 6712)          # [B, V]
        train_masks[:, :538] = 0  # 边界条件掩码
        
        # 模拟验证数据
        eval_inputs = torch.randn(20, 3, 6712)
        eval_targets = torch.randn(20, 3, 6712)
        eval_masks = torch.ones(20, 6712)
        eval_masks[:, :538] = 0
        
        # 创建数据集 (简化格式)
        class MockDataset:
            def __init__(self, inputs, targets, masks):
                self.inputs = inputs
                self.targets = targets
                self.masks = masks
            
            def __len__(self):
                return len(self.inputs)
            
            def __getitem__(self, idx):
                return {
                    'input': self.inputs[idx],
                    'target': self.targets[idx], 
                    'prediction_mask': self.masks[idx]
                }
        
        train_dataset = MockDataset(train_inputs, train_targets, train_masks)
        eval_dataset = MockDataset(eval_inputs, eval_targets, eval_masks)
        
        logger.info("使用模拟数据集")
    
    # ====== 3. 数据整理器 ======
    data_collator = FluidDataCollator()
    
    # ====== 4. 训练参数 ======
    logger.info("设置训练参数...")
    training_args = TrainingArguments(
        # 基础设置
        output_dir="./example_results",
        run_name="fluid_example",
        overwrite_output_dir=True,
        
        # 训练控制 (示例用较少的epochs)
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        
        # 优化器
        optim="adamw_torch",
        learning_rate=1e-4,
        weight_decay=1e-5,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        
        # 性能设置
        fp16=torch.cuda.is_available(),  # 如果有GPU则使用混合精度
        dataloader_num_workers=2,
        
        # 评估和保存
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mse",
        greater_is_better=False,
        
        # 日志
        logging_steps=20,
        report_to=["tensorboard"],
        
        # 其他
        seed=42,
        remove_unused_columns=False,
    )
    
    # ====== 5. 创建 Trainer ======
    logger.info("创建 Trainer...")
    
    # 创建回调函数
    callbacks = create_training_callbacks(save_plots=True)
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_fluid_metrics,
        callbacks=callbacks,
    )
    
    # ====== 6. 开始训练 ======
    logger.info("开始训练...")
    
    try:
        # 训练模型
        train_result = trainer.train()
        
        logger.info("训练完成!")
        logger.info(f"最终训练损失: {train_result.training_loss:.6f}")
        logger.info(f"训练步数: {train_result.global_step}")
        
        # 保存模型
        trainer.save_model("./example_results/final_model")
        logger.info("模型已保存到 './example_results/final_model'")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        return
    
    # ====== 7. 最终评估 ======
    logger.info("执行最终评估...")
    eval_results = trainer.evaluate()
    logger.info("评估结果:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
    
    # ====== 8. 推理示例 ======
    logger.info("推理示例...")
    try:
        # 创建推理接口
        inference = FluidInference(
            model_path="./example_results/final_model",
            device="auto"
        )
        
        # 模型信息
        model_info = inference.get_model_info()
        logger.info(f"推理模型信息: {model_info}")
        
        # 单样本预测示例
        import numpy as np
        test_input = np.random.randn(3, 6712)  # [T=3, V=6712]
        
        result = inference.predict_single(
            test_input,
            denormalize=False  # 示例数据没有归一化
        )
        
        logger.info(f"单样本预测完成，输出形状: {result['predictions'].shape}")
        
        # 批量预测示例
        test_batch = np.random.randn(5, 3, 6712)  # [B=5, T=3, V=6712]
        
        batch_result = inference.predict_batch(
            test_batch,
            batch_size=2,
            denormalize=False
        )
        
        logger.info(f"批量预测完成，输出形状: {batch_result['predictions'].shape}")
        
        logger.info("推理示例完成!")
        
    except Exception as e:
        logger.error(f"推理示例失败: {e}")
    
    logger.info("="*50)
    logger.info("FluidDecoder 训练系统示例完成!")
    logger.info("检查 './example_results/' 目录查看输出文件")
    logger.info("="*50)


if __name__ == "__main__":
    main()