# Training 模块设计文档

## 任务概述
基于已完成的 @models (FluidDecoder) 和 @data (数据处理) 模块，设计完整的训练框架。

## 核心架构选择
**使用 🤗 Transformers Trainer** - 成熟稳定的训练框架，提供自动混合精度、分布式训练、实验跟踪等功能。

## 主要模块设计

### 1. HuggingFace 集成适配

#### 模型适配器 - FluidDecoderForTraining
让 FluidDecoder 兼容 HuggingFace 接口，支持标准的训练、评估和保存流程。

#### 数据整理器 - FluidDataCollator  
处理批次数据的组装，确保输入格式正确：[B, T, V=6712]

#### 指标计算 - compute_fluid_metrics
专门针对流体系统的评估指标：MSE、MAE、设备分组评估

### 2. 训练配置和控制
- **TrainingArguments** - 完整的训练参数配置
- **EarlyStoppingCallback** - 验证损失早停机制
- **实验跟踪** - WandB/TensorBoard 集成
- **混合精度训练** - FP16 自动优化
- **梯度裁剪和累积** - 稳定训练的关键技术

### 3. 评估指标系统
- **核心指标**: MSE、MAE、RMSE、R²
- **设备分组评估**: B(球阀)、C(压缩机)、H(管段)、N(节点)、P(管道)、R(调节阀)、T&E(气源分输)
- **掩码损失**: 只对设备变量计算损失，忽略边界条件

## 文件结构
```
training/
├── __init__.py                 # 包初始化
├── config.py                  # 训练配置类
├── inference.py               # 推理接口
├── scripts/                   # 训练脚本
│   ├── train.py              # 主训练脚本
│   └── evaluate.py           # 评估脚本
└── utils.py                   # 训练工具函数
```

## 核心训练流程
1. **数据准备** - 加载训练/验证数据集，应用归一化
2. **模型初始化** - 创建 FluidDecoderForTraining 实例
3. **训练配置** - 设置 TrainingArguments 参数
4. **训练执行** - Trainer 自动化训练循环
5. **模型保存** - 保存最佳检查点和最终模型

## 使用示例
```python
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from training.hf_integration import FluidDecoderForTraining, FluidDecoderConfig, FluidDataCollator, compute_fluid_metrics
from data import create_datasets

# 1. 模型配置
config = FluidDecoderConfig(d_model=256, n_heads=8, n_layers=6)
model = FluidDecoderForTraining(config)

# 2. 训练配置
training_args = TrainingArguments(
    output_dir="./fluid_results",
    num_train_epochs=100,
    per_device_train_batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-5,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_mse",
    greater_is_better=False,
    fp16=True,
    report_to=["wandb"],
)

# 3. 训练执行
train_dataset, eval_dataset = create_datasets()
data_collator = FluidDataCollator()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_fluid_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

trainer.train()
trainer.save_model("./fluid_final_model")
```

## 关键技术要点
- **掩码损失**: 使用 prediction_mask 只对设备变量计算损失
- **维度处理**: [B, T, V] → [B, T*V, d_model] → [B, T, V] 
- **数据归一化**: 训练时标准化，推理时需要反归一化
- **梯度裁剪**: 防止梯度爆炸 (max_grad_norm=1.0)
- **早停机制**: 验证损失不再下降时停止