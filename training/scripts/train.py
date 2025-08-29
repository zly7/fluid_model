#!/usr/bin/env python3
"""
主训练脚本：使用 transformers.Trainer 训练 FluidDecoder 模型。

使用方法:
    python training/scripts/train.py --config config.yaml
    python training/scripts/train.py --model_size small --epochs 50 --batch_size 16
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import wandb
from datetime import datetime

# 添加项目路径到 Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import (
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    set_seed
)

# 导入项目模块
from training.hf_integration import (
    FluidDecoderForTraining,
    FluidDecoderConfig, 
    FluidDataCollator,
    compute_fluid_metrics
)
from data import create_data_loaders
from utils.metrics import FluidMetricsTracker

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train FluidDecoder with transformers.Trainer")
    
    # 模型配置
    parser.add_argument("--model_size", type=str, default="medium", 
                       choices=["small", "medium", "large"],
                       help="预定义模型大小")
    parser.add_argument("--d_model", type=int, default=None, help="模型隐藏维度")
    parser.add_argument("--n_heads", type=int, default=None, help="注意力头数")
    parser.add_argument("--n_layers", type=int, default=None, help="Decoder层数")
    parser.add_argument("--d_ff", type=int, default=None, help="前馈网络维度")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="评估批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=500, help="预热步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    
    # 数据配置
    parser.add_argument("--data_dir", type=str, default="data/dataset", help="数据目录")
    parser.add_argument("--time_steps", type=int, default=3, help="时间步长")
    parser.add_argument("--max_workers", type=int, default=4, help="数据加载工作进程数")
    
    # 训练控制
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fp16", action="store_true", help="使用混合精度训练")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="使用梯度检查点")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="早停耐心值")
    
    # 保存和日志
    parser.add_argument("--output_dir", type=str, default="./fluid_results", help="输出目录")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志记录步数")
    parser.add_argument("--eval_steps", type=int, default=500, help="评估步数")
    parser.add_argument("--save_steps", type=int, default=1000, help="保存步数")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存检查点数量限制")
    
    # 实验跟踪
    parser.add_argument("--wandb", action="store_true", help="使用 Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="fluid-dynamics", help="W&B 项目名")
    parser.add_argument("--run_name", type=str, default=None, help="实验运行名称")
    
    # 其他选项
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                       help="从检查点恢复训练")
    parser.add_argument("--do_eval_only", action="store_true", help="仅执行评估")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    return parser.parse_args()


def get_model_config(args) -> FluidDecoderConfig:
    """根据参数创建模型配置。"""
    
    # 预定义模型大小
    model_sizes = {
        "small": {
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 3,
            "d_ff": 512
        },
        "medium": {
            "d_model": 256, 
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 1024
        },
        "large": {
            "d_model": 512,
            "n_heads": 16, 
            "n_layers": 12,
            "d_ff": 2048
        }
    }
    
    # 获取预定义配置
    if args.model_size in model_sizes:
        config_dict = model_sizes[args.model_size].copy()
    else:
        config_dict = model_sizes["medium"].copy()
    
    # 命令行参数覆盖预定义配置
    if args.d_model is not None:
        config_dict["d_model"] = args.d_model
    if args.n_heads is not None:
        config_dict["n_heads"] = args.n_heads  
    if args.n_layers is not None:
        config_dict["n_layers"] = args.n_layers
    if args.d_ff is not None:
        config_dict["d_ff"] = args.d_ff
    
    # 创建配置
    config = FluidDecoderConfig(
        d_model=config_dict["d_model"],
        n_heads=config_dict["n_heads"], 
        n_layers=config_dict["n_layers"],
        d_ff=config_dict["d_ff"],
        input_dim=6712,
        output_dim=6712,
        boundary_dims=538,
        dropout=0.1,
        activation="gelu"
    )
    
    logger.info(f"Model config: {args.model_size} - d_model={config.d_model}, n_heads={config.n_heads}, n_layers={config.n_layers}")
    
    return config


def get_training_args(args) -> TrainingArguments:
    """创建训练参数。"""
    
    # 生成运行名称
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"fluid_decoder_{args.model_size}_{timestamp}"
    
    # 设置报告工具
    report_to = []
    if args.wandb:
        report_to.append("wandb")
        # 初始化 wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)
        )
    
    # 总是启用 tensorboard
    report_to.append("tensorboard")
    
    training_args = TrainingArguments(
        # 基础设置
        output_dir=args.output_dir,
        run_name=args.run_name,
        overwrite_output_dir=True,
        
        # 训练控制  
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # 优化器和学习率
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        
        # 性能优化
        fp16=args.fp16,
        bf16=False,  # 如果支持可以设为 True
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.max_workers,
        dataloader_pin_memory=True,
        
        # 评估和保存
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        eval_delay=0,
        save_strategy="steps", 
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mse",
        greater_is_better=False,
        
        # 日志和实验跟踪
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=report_to,
        
        # 其他设置
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        prediction_loss_only=False,
        ignore_data_skip=False,
    )
    
    return training_args


def main():
    """主训练函数。"""
    args = parse_args()
    
    # 设置调试模式
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*50)
    logger.info("开始 FluidDecoder 训练")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"模型大小: {args.model_size}")
    logger.info(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    logger.info("="*50)
    
    try:
        # 1. 创建模型配置和模型
        logger.info("创建模型...")
        config = get_model_config(args)
        model = FluidDecoderForTraining(config)
        
        logger.info(f"模型参数量: {model.num_parameters():,}")
        
        # 2. 加载数据
        logger.info("加载数据集...")
        train_dataloader, eval_dataloader = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            time_steps=args.time_steps,
            num_workers=args.max_workers,
            pin_memory=True
        )
        
        # 转换为 Dataset 对象 (HuggingFace 格式)
        train_dataset = train_dataloader.dataset
        eval_dataset = eval_dataloader.dataset
        
        logger.info(f"训练样本数: {len(train_dataset)}")
        logger.info(f"验证样本数: {len(eval_dataset)}")
        
        # 3. 创建数据整理器
        data_collator = FluidDataCollator()
        
        # 4. 设置训练参数
        training_args = get_training_args(args)
        
        # 5. 创建回调函数
        callbacks = []
        if args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_threshold=0.0001
                )
            )
        
        # 6. 创建 Trainer
        logger.info("创建 Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_fluid_metrics,
            callbacks=callbacks,
        )
        
        # 7. 训练前检查
        logger.info("训练配置检查:")
        logger.info(f"- 有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
        logger.info(f"- 总训练步数: {len(train_dataset) // args.batch_size * args.epochs}")
        logger.info(f"- 学习率调度: cosine")
        logger.info(f"- 优化器: adamw_torch")
        logger.info(f"- 混合精度: {args.fp16}")
        
        # 8. 开始训练或评估
        if args.do_eval_only:
            logger.info("执行模型评估...")
            if args.resume_from_checkpoint:
                logger.info(f"从检查点加载: {args.resume_from_checkpoint}")
            eval_results = trainer.evaluate()
            logger.info(f"评估结果: {eval_results}")
        else:
            logger.info("开始训练...")
            train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            
            # 保存最终模型
            logger.info("保存最终模型...")
            trainer.save_model()
            
            # 记录训练结果
            logger.info("训练完成!")
            logger.info(f"最终训练损失: {train_result.training_loss:.6f}")
            logger.info(f"训练步数: {train_result.global_step}")
            
            # 最终评估
            logger.info("执行最终评估...")
            eval_results = trainer.evaluate()
            logger.info(f"最终评估结果: {eval_results}")
            
            # 保存训练历史
            if trainer.state.log_history:
                import json
                with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
                    json.dump(trainer.state.log_history, f, indent=2)
    
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        raise
    
    finally:
        # 清理资源
        if args.wandb and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()